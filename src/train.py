import os
import sys
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

sys.path.append("/content/BrainTumorSeg/src")
from model_nnunet import NNUNet3D
from losses import CombinedLoss
from dataset import get_loaders

# ── Paths ──────────────────────────────────────────────
DATA_DIR   = "/content/drive/MyDrive/BrainTumorFYP/data"
MODEL_DIR  = "/content/drive/MyDrive/BrainTumorFYP/models"
OUTPUT_DIR = "/content/drive/MyDrive/BrainTumorFYP/outputs"

# ── Config ─────────────────────────────────────────────
EPOCHS      = 50
BATCH_SIZE  = 2
LR          = 1e-4
NUM_CLASSES = 4

def dice_score(pred, target, num_classes=4):
    pred_labels = pred.argmax(dim=1)
    scores = []
    for c in range(1, num_classes):
        p = (pred_labels == c).float()
        t = (target      == c).float()
        intersection = (p * t).sum()
        union        = p.sum() + t.sum()
        score = (2 * intersection + 1e-5) / (union + 1e-5)
        scores.append(score.item())
    return np.mean(scores)


def save_checkpoint(epoch, model, optimizer, scheduler,
                    scaler, best_dice, train_losses, val_dices, filename):
    torch.save({
        "epoch"        : epoch,
        "model_state"  : model.state_dict(),
        "optimizer"    : optimizer.state_dict(),
        "scheduler"    : scheduler.state_dict(),
        "scaler"       : scaler.state_dict(),
        "best_dice"    : best_dice,
        "train_losses" : train_losses,
        "val_dices"    : val_dices,
    }, filename)


def load_checkpoint(filename, model, optimizer, scheduler, scaler):
    print(f"Loading checkpoint: {filename}")
    ckpt = torch.load(filename, map_location="cuda"
                      if torch.cuda.is_available() else "cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch  = ckpt["epoch"] + 1
    best_dice    = ckpt["best_dice"]
    train_losses = ckpt["train_losses"]
    val_dices    = ckpt["val_dices"]
    print(f"✅ Resumed from epoch {ckpt['epoch']} | Best Dice so far: {best_dice:.4f}")
    return start_epoch, best_dice, train_losses, val_dices


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data
    train_loader, val_loader = get_loaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Model + optimizer + scheduler + scaler
    model     = NNUNet3D(in_channels=4, out_channels=NUM_CLASSES).to(device)
    criterion = CombinedLoss(num_classes=NUM_CLASSES)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler("cuda")

    # ── Check if resume checkpoint exists ──────────────
    resume_path = f"{MODEL_DIR}/last_checkpoint.pth"
    best_path   = f"{MODEL_DIR}/best_model.pth"

    start_epoch  = 1
    best_dice    = 0.0
    train_losses = []
    val_dices    = []

    if os.path.exists(resume_path):
        print("\n🔄 Found existing checkpoint — resuming training...")
        start_epoch, best_dice, train_losses, val_dices = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler
        )
    else:
        print("\n🚀 No checkpoint found — starting fresh training...")

    # ── Training loop ──────────────────────────────────
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f"\n── Epoch {epoch}/{EPOCHS} ──────────────────────")

        # Training
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()

            with autocast("cuda"):
                preds = model(images)
                loss  = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        # Validation
        model.eval()
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks  = masks.to(device)
                with autocast("cuda"):
                    preds = model(images)
                val_dice += dice_score(preds, masks)

        avg_dice = val_dice / len(val_loader)
        val_dices.append(avg_dice)

        print(f"  Train Loss : {avg_loss:.4f}")
        print(f"  Val Dice   : {avg_dice:.4f}")

        # Save last checkpoint (always — every epoch)
        save_checkpoint(epoch, model, optimizer, scheduler,
                        scaler, best_dice, train_losses, val_dices,
                        resume_path)
        print(f"  💾 Last checkpoint saved (epoch {epoch})")

        # Save best model separately
        if avg_dice > best_dice:
            best_dice = avg_dice
            save_checkpoint(epoch, model, optimizer, scheduler,
                            scaler, best_dice, train_losses, val_dices,
                            best_path)
            print(f"  ✅ Best model saved! Dice = {best_dice:.4f}")

        # Save checkpoint every 10 epochs as extra backup
        if epoch % 10 == 0:
            backup_path = f"{MODEL_DIR}/checkpoint_epoch_{epoch}.pth"
            save_checkpoint(epoch, model, optimizer, scheduler,
                            scaler, best_dice, train_losses, val_dices,
                            backup_path)
            print(f"  📦 Backup checkpoint saved at epoch {epoch}")

        # Save training history as numpy arrays
        np.save(f"{OUTPUT_DIR}/train_losses.npy", np.array(train_losses))
        np.save(f"{OUTPUT_DIR}/val_dices.npy",    np.array(val_dices))

    print(f"\n🎉 Training complete! Best Dice: {best_dice:.4f}")
    return model, train_losses, val_dices


if __name__ == "__main__":
    model, train_losses, val_dices = train()
