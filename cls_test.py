import torch
import sys
sys.path.append('/well/rittscher/users/ycr745/nnUNet/nnunetv2')
from test import load_and_modify_model, CTDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torchio as tio
from torchvision import transforms
from tqdm import tqdm
import os
import pandas as pd
import nibabel as nib
import torch.nn.functional as F
import re

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the validation dataset and compute metrics.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Get class predictions
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Flatten lists
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    report = classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(len(np.unique(all_labels)))])
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return report, conf_matrix

def test_model(model, device):
    # Path to the test images directory
    test_images_dir = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/imagesTs"
    
    # Set model to evaluation mode
    model.eval()
    
    # Store results
    results = []

    with torch.no_grad():
        for file_name in tqdm(os.listdir(test_images_dir)):
            if not file_name.endswith(".nii.gz"):
                continue

            # Full path to the image
            img_path = os.path.join(test_images_dir, file_name)

            # Load the image data
            img = nib.load(img_path).get_fdata()
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add channel and batch dimensions
            img = F.interpolate(img, size=(128, 128, 128), mode='trilinear', align_corners=False)
            # Predict using the model
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            subtype = predicted.item()

            modified_name = re.sub(r'_0000', '', file_name)
            # Extract the numeric part from the file name for sorting
            numeric_part = int(re.search(r'\d+', modified_name).group())

            # Add the results to the list
            results.append({"Names": modified_name, "Subtype": subtype, "SortKey": numeric_part})
    
    # Sort results by the numeric part of the file name
    results = sorted(results, key=lambda x: x["SortKey"])

    # Remove the SortKey before saving to CSV
    for item in results:
        item.pop("SortKey")

    # Save results to CSV
    results_dir = '/well/rittscher/users/ycr745/nnUNet/nnunetv2/cls_results'
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(results_dir, "subtype_results.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(f"Results saved to {results_csv_path}")

  


def main():
    data_dir = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/imagesTr"
    split_file = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/splits_final.json"
    label_file = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/image_to_subtype.json"
    checkpoint_path = '/well/rittscher/users/ycr745/nnUNet/nnunetv2/cls_results/best.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_and_modify_model(checkpoint_path=None, num_classes=3, dropout_rate=0.5).to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    # val_transforms = tio.Compose([
    #     NormalizeTransform(),
    # ])
    val_transforms = transforms.Compose([
        transforms.Normalize(mean=0.0, std=1.0),
        ]) 
    
    val_dataset = CTDataset(data_dir, split_file, label_file, mode="val", transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # Evaluate the model on the validation set
    report, conf_matrix = evaluate_model(model, val_loader, device)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(conf_matrix)

    test_model(model, device)

if __name__ == "__main__":
    main()
