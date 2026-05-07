import os, sys, base64, io
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
from scipy.ndimage import zoom

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model_nnunet import NNUNet3D

app = Flask(__name__)

# ── Load model once at startup ──────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__),
             "..", "models", "best_model.pth")

device = torch.device("cpu")
model  = NNUNet3D().to(device)
ckpt   = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"✅ Model loaded | Epoch {ckpt['epoch']} | Dice {ckpt['best_dice']:.4f}")

LABEL_NAMES = {0: "Background", 1: "NCR", 2: "Edema", 3: "Enhancing Tumor"}
COLORS      = {0: "None", 1: "Red", 2: "Yellow", 3: "Dark Red"}

def preprocess_volume(vol):
    """Normalize and resize to 128x128x128"""
    vol  = vol.astype(np.float32)
    vol  = zoom(vol, [128/s for s in vol.shape], order=1)
    mask = vol > 0
    if mask.sum() > 0:
        vol = (vol - vol[mask].mean()) / (vol[mask].std() + 1e-8)
    vol[~mask] = 0
    return vol

def array_to_base64(arr, cmap="gray", vmin=None, vmax=None):
    """Convert numpy array to base64 PNG string"""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap=cmap,
              vmin=vmin if vmin is not None else arr.min(),
              vmax=vmax if vmax is not None else arr.max())
    ax.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0, dpi=100)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        f    = request.files["mri"]
        path = f"/tmp/{f.filename}"
        f.save(path)

        # Load and preprocess
        vol  = nib.load(path).get_fdata()
        vol  = preprocess_volume(vol)
        tensor = torch.FloatTensor(vol).unsqueeze(0).unsqueeze(0)
        tensor = tensor.repeat(1, 4, 1, 1, 1)  # repeat as 4 channels

        # Predict
        with torch.no_grad():
            pred = model(tensor).argmax(dim=1)[0].numpy()

        # Middle slice
        sl = vol.shape[2] // 2

        # Convert to images
        input_img = array_to_base64(vol[:,:,sl], cmap="gray")
        pred_img  = array_to_base64(pred[:,:,sl], cmap="jet", vmin=0, vmax=3)

        # Overlay
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(vol[:,:,sl], cmap="gray")
        ax.imshow(pred[:,:,sl], cmap="jet", alpha=0.4, vmin=0, vmax=3)
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0, dpi=100)
        plt.close()
        buf.seek(0)
        overlay_img = base64.b64encode(buf.read()).decode()

        # Tumor stats
        total_voxels = float(pred.size)
        stats = {
            "Whole Tumor"     : f"{float((pred > 0).sum()) / total_voxels * 100:.1f}%",
            "Tumor Core"      : f"{float(((pred==1)|(pred==3)).sum()) / total_voxels * 100:.1f}%",
            "Enhancing Tumor" : f"{float((pred==3).sum()) / total_voxels * 100:.1f}%",
        }

        return jsonify({
            "success" : True,
            "input"   : input_img,
            "pred"    : pred_img,
            "overlay" : overlay_img,
            "stats"   : stats,
            "info"    : f"Best model (Epoch {ckpt['epoch']}, Dice {ckpt['best_dice']:.4f})"
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
