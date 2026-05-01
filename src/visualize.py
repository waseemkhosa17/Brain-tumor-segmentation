import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR   = '/content/drive/MyDrive/BrainTumorFYP/data'
OUTPUT_DIR = '/content/drive/MyDrive/BrainTumorFYP/outputs'

cases = sorted([f for f in os.listdir(DATA_DIR) if f.startswith('BraTS')])

def visualize_sample(case_name):
    case_path = os.path.join(DATA_DIR, case_name)
    flair = nib.load(f'{case_path}/{case_name}_flair.nii.gz').get_fdata()
    t1    = nib.load(f'{case_path}/{case_name}_t1.nii.gz').get_fdata()
    t1ce  = nib.load(f'{case_path}/{case_name}_t1ce.nii.gz').get_fdata()
    t2    = nib.load(f'{case_path}/{case_name}_t2.nii.gz').get_fdata()
    seg   = nib.load(f'{case_path}/{case_name}_seg.nii.gz').get_fdata()

    sl = flair.shape[2] // 2
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'Case: {case_name}  |  Slice: {sl}', fontsize=13)
    axes[0].imshow(flair[:,:,sl], cmap='gray'); axes[0].set_title('FLAIR')
    axes[1].imshow(t1[:,:,sl],    cmap='gray'); axes[1].set_title('T1')
    axes[2].imshow(t1ce[:,:,sl],  cmap='gray'); axes[2].set_title('T1ce')
    axes[3].imshow(t2[:,:,sl],    cmap='gray'); axes[3].set_title('T2')
    axes[4].imshow(seg[:,:,sl],   cmap='jet', vmin=0, vmax=4)
    axes[4].set_title('Segmentation')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/sample_visualization.png', bbox_inches='tight', dpi=100)
    plt.show()
    print(f"Shape: {flair.shape}, Labels: {np.unique(seg)}")

visualize_sample(cases[0])
