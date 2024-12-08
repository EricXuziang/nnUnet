from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import nibabel as nib
import json
from pathlib import Path
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchio as tio
from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
import os
import numpy as np
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
# from nnunetv2.dataloader_own import get_tr_and_val_datasets, get_dataloaders

class MultiScaleClassificationHead(nn.Module):
    def __init__(self, feature_dims, num_classes=3, dropout_rate=0.5):
        """
        feature_dims: List of feature dimensions for each scale (e.g., [32, 64, 128, 256, 320, 320])
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        """
        super(MultiScaleClassificationHead, self).__init__()
        self.gap = nn.ModuleList([nn.AdaptiveAvgPool3d(1) for _ in feature_dims])  # GAP for each scale
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(sum(feature_dims), num_classes)  # Concatenate all GAP outputs
        )

    def forward(self, features):
        """
        features: List of multi-scale feature maps from the encoder
        """
        pooled_features = [self.gap[i](features[i]).squeeze(-1).squeeze(-1).squeeze(-1) for i in range(len(features))]
        concatenated_features = torch.cat(pooled_features, dim=1)  # Concatenate along channel dimension
        return self.fc(concatenated_features)
    
class MultiTaskModel(nn.Module):
    def __init__(self, full_model,feature_dims, num_classes=3, dropout_rate=0.5):
        super(MultiTaskModel, self).__init__()
        # Full segmentation model
        self.encoder = nn.Sequential(
            full_model.encoder
        )
        self.segmentation_head = full_model.decoder
        self.classification_head = MultiScaleClassificationHead(feature_dims, num_classes, dropout_rate)

    def forward(self, x):
        # Shared encoder features
        features = self.encoder(x)
        # Segmentation output
        segmentation_output = self.segmentation_head(features)
        # segmentation_output = [seg_layer(features) for seg_layer in self.segmentation_head.seg_layers]
        
        # Classification output
        classification_output = self.classification_head(features)
        
        return classification_output, segmentation_output

def load_and_modify_multitask_model(checkpoint_path=None, num_classes=3, dropout_rate=0.5):
    # Load the full model (e.g., ResidualEncoderUNet)
    full_model = get_network_from_plans(
        arch_class_name="dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        arch_kwargs={
            "n_stages": 6,
            "features_per_stage": [32, 64, 128, 256, 320, 320],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-05, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=1,
        output_channels=3,
        allow_init=True,
        deep_supervision=True,
    )
    
    # if checkpoint_path is not None:
    #     print(f"Loading checkpoint from: {checkpoint_path}")
    #     checkpoint = torch.load(checkpoint_path)
    #     full_model.load_state_dict(checkpoint['network_weights'])
    # else:
    #     print("No checkpoint provided. Model initialized without pre-trained weights.")

    # Create the multi-task wrapper model
    feature_dims = [32, 64, 128, 256, 320, 320]
    model = MultiTaskModel(full_model, feature_dims=feature_dims, num_classes=num_classes, dropout_rate=dropout_rate)
    print("Multi-task model created with shared encoder, segmentation model, and classification head.")
    return model

class MultiTaskDataset(Dataset):
    def __init__(self, image_dir, seg_dir, label_file, split_file, split, transform=None):
        """
        image_dir: Path to the directory containing input images.
        seg_dir: Path to the directory containing segmentation masks.
        label_file: Path to the JSON file containing classification labels.
        split_file: Path to the JSON file containing train/val splits.
        split: Either 'train' or 'val' to select the appropriate data split.
        transform: TorchIO transformations to apply to the data.
        """
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.transform = transform

        # Load classification labels
        with open(label_file, 'r') as f:
            self.cls_labels = json.load(f)
        self.cls_labels = self.cls_labels[split]
        # Load train/val split
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.case_ids = splits[0][split]

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        # Get case ID
        case_id = self.case_ids[idx]

        # Load image (e.g., case_002_0000.nii.gz)
        image_path = os.path.join(self.image_dir, f"{case_id}_0000.nii.gz")
        image = nib.load(image_path).get_fdata(dtype=np.float32)
        image = np.expand_dims(image, axis=0)  # Add channel dimension (C, D, H, W)

        # Load segmentation label (e.g., case_002.nii.gz)
        seg_path = os.path.join(self.seg_dir, f"{case_id}.nii.gz")
        segmentation = nib.load(seg_path).get_fdata(dtype=np.float32).astype(np.int64)

        target_shape = (128, 128, 128)  # Target size (D, H, W)
        image = F.interpolate(torch.from_numpy(image).unsqueeze(0), size=target_shape, mode='trilinear').squeeze(0)
        segmentation = F.interpolate(
            torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0).float(),
            size=target_shape,
            mode='nearest'
            ).squeeze(0).squeeze(0).long()
        # Load classification label
        cls_label = self.cls_labels[case_id]

        # Apply transformations
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                segmentation=tio.LabelMap(tensor=segmentation[None, ...])
            )
            transformed = self.transform(subject)
            image = transformed['image'].data
            segmentation = transformed['segmentation'].data.squeeze(0)  # Remove channel for segmentation

        # Classification label tensor
        cls_label = torch.tensor(cls_label, dtype=torch.long)

        return {
            "image": image,  # Tensor of shape (C, D, H, W)
            "segmentation": segmentation,  # Tensor of shape (D, H, W)
            "cls_label": cls_label  # Scalar tensor
        }
    
def get_transforms(mode):
    if mode == 'train':
        return tio.Compose([
        tio.RescaleIntensity((-1, 1)),  # Normalize intensity values
        tio.RandomFlip(axes=(0, 1, 2)),  # Random flips
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),  # Random scaling and rotation
        tio.RandomNoise(mean=0.0, std=(0.01, 0.1)),  # Add random noise
        ]) 
    else:
        return tio.Compose([
        tio.RescaleIntensity((-1, 1)),  # Normalize intensity values
        ]) 

class MultiResolutionSegLoss(nn.Module):
    def __init__(self, dice_kwargs, ce_kwargs, weights=None):
        """
        Multi-resolution Segmentation Loss Function
        :param dice_kwargs: Parameters for the Dice loss.
        :param ce_kwargs: Parameters for the Cross-Entropy (CE) loss.
        :param weights: Weights for outputs at different resolutions, defaulting to equal weighting for each output layer.
        """
        super(MultiResolutionSegLoss, self).__init__()
        self.dice_loss = MultiClassDiceLoss(**dice_kwargs)
        self.ce_loss = WeightedCELoss(**ce_kwargs)
        self.weights = weights

    def forward(self, outputs, targets):
        """
        :param outputs: List containing segmentation outputs at multiple resolutions [B, C, D, H, W].
         :param targets: Segmentation labels with shape [B, D, H, W]. Used for multi-class Dice loss.
        """
        if self.weights is None:
            self.weights = [1.0 / len(outputs)] * len(outputs)  
        assert len(outputs) == len(self.weights), 

        total_loss = 0
        for output, weight in zip(outputs, self.weights):

            resized_targets = F.interpolate(
                targets.unsqueeze(1).float(),
                size=output.shape[2:],  
                mode='nearest'
            ).squeeze(1).long()

            dice_loss = self.dice_loss(output, resized_targets)
            ce_loss = self.ce_loss(output, resized_targets)

            total_loss += weight * (dice_loss + ce_loss)

        return total_loss
    
class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, include_background=False):
        """
        Multi class Dice loss
        :param smooth: A small constant to prevent division by zero.
        :param include_background: Whether to include the background class in the loss computation.
        """
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits, targets):
        """
        :param logits: Network outputs with shape (B, C, D, H, W), before applying softmax.
        :param targets: Ground truth labels with shape (B, D, H, W), containing class indices 0, 1, 2.
        """

        probs = torch.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets.long(), num_classes=logits.size(1)).permute(0, 4, 1, 2, 3)

        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3, 4))
        union = torch.sum(probs, dim=(2, 3, 4)) + torch.sum(targets_one_hot, dim=(2, 3, 4))
        dice_score = (2 * intersection + self.smooth) / (union + self.smooth)

        if not self.include_background:
            dice_score = dice_score[:, 1:]

        return 1 - dice_score.mean()
    
class WeightedCELoss(nn.Module):
    def __init__(self, class_weights=None):
        """
        :param class_weights: Weights for each class.
        """
        super(WeightedCELoss, self).__init__()
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        """
        :param logits: Network outputs with shape (B, C, D, H, W).
        :param targets: Ground truth labels with shape (B, D, H, W). 
        """
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=self.class_weights)
        
class MultiTaskLoss(nn.Module):
    def __init__(self, dice_kwargs, ce_kwargs, cls_loss_weight=1.0, seg_loss_weight=1.0, seg_weights=None):
        """
        Multi-task loss combining multi-resolution segmentation (Dice + CE) and classification loss.
        :param dice_kwargs: Parameters for the Dice loss.
        :param ce_kwargs: Parameters for the CE loss.
        :param cls_loss_weight: Weight for the classification loss.
        :param seg_loss_weight: Weight for the segmentation loss.
        :param seg_weights: Weights for outputs at different resolutions.
        """
        super(MultiTaskLoss, self).__init__()
        self.cls_loss_weight = cls_loss_weight
        self.seg_loss_weight = seg_loss_weight

        self.seg_loss = MultiResolutionSegLoss(dice_kwargs, ce_kwargs, weights=seg_weights)
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, seg_outputs, cls_logits, seg_targets, cls_targets):
        """
        :param seg_outputs: Outputs for the segmentation task, a list where each element has shape [B, C, D, H, W].
        :param cls_logits: Outputs for the classification task with shape [B, num_classes].
        :param seg_targets: Ground truth labels for the segmentation task with shape [B, D, H, W].
        :param cls_targets: Ground truth labels for the classification task with shape [B].
        """
        seg_loss = self.seg_loss(seg_outputs, seg_targets)
        cls_loss = self.cls_loss(cls_logits, cls_targets)

        total_loss = self.seg_loss_weight * seg_loss + self.cls_loss_weight * cls_loss
        return total_loss, seg_loss, cls_loss 

def load_json(path):
    import json 
    with open(path, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)
    return load_dict

def create_dataloader(image_dir, seg_dir, label_file, split_file, split, mode, batch_size=4, shuffle=True, num_workers=4):
    """
    image_dir: Path to directory containing input images
    seg_dir: Path to directory containing segmentation masks
    label_file: Path to JSON file containing classification labels
    split_file: Path to JSON file containing train/val splits
    split: Either 'train' or 'val'
    mode: Mode for data augmentation ('train' or 'val')
    """
    dataset = MultiTaskDataset(
        image_dir=image_dir,
        seg_dir=seg_dir,
        label_file=label_file,
        split_file=split_file,
        split=split,
        transform=get_transforms(mode)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), num_workers=num_workers)

def plot_individual_loss_curves(train_total_losses, val_total_losses,
                                train_seg_losses, val_seg_losses,
                                train_cls_losses, val_cls_losses,
                                save_path):
    """
    Plot loss
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot total loss
    plt.figure()
    plt.plot(train_total_losses, label='Train Total Loss')
    plt.plot(val_total_losses, label='Val Total Loss')
    plt.title('Training and Validation Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path / 'total_loss_curve.png')
    plt.close()

    # Plot segmentation loss
    plt.figure()
    plt.plot(train_seg_losses, label='Train Segmentation Loss')
    plt.plot(val_seg_losses, label='Val Segmentation Loss')
    plt.title('Training and Validation Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path / 'segmentation_loss_curve.png')
    plt.close()

    # Plot classification loss
    plt.figure()
    plt.plot(train_cls_losses, label='Train Classification Loss')
    plt.plot(val_cls_losses, label='Val Classification Loss')
    plt.title('Training and Validation Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path / 'classification_loss_curve.png')
    plt.close()

def train_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs, 
    optimizer, 
    scheduler, 
    loss_fn, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model.to(device)
    best_val_loss = float("inf")
    train_total_losses,val_total_losses,train_seg_losses,val_seg_losses,train_cls_losses,val_cls_losses  = [],[],[],[],[],[]
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_seg_loss, train_cls_loss = 0.0, 0.0, 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            segmentations = batch["segmentation"].to(device)
            cls_labels = batch["cls_label"].to(device)

            optimizer.zero_grad()
            cls_output, seg_output = model(images)

            total_loss, seg_loss, cls_loss = loss_fn(seg_output, cls_output, segmentations, cls_labels)

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            train_seg_loss += seg_loss.item()
            train_cls_loss += cls_loss.item()

        train_loss /= len(train_loader)
        train_seg_loss /= len(train_loader)
        train_cls_loss /= len(train_loader)

        train_total_losses.append(train_loss)
        train_seg_losses.append(train_seg_loss)
        train_cls_losses.append(train_cls_loss)

        # Validation
        model.eval()
        val_loss, val_seg_loss, val_cls_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                segmentations = batch["segmentation"].to(device)
                cls_labels = batch["cls_label"].to(device)

                cls_output, seg_output = model(images)
                total_loss, seg_loss, cls_loss = loss_fn(seg_output, cls_output, segmentations, cls_labels)

                val_loss += total_loss.item()
                val_seg_loss += seg_loss.item()
                val_cls_loss += cls_loss.item()

        val_loss /= len(val_loader)
        val_seg_loss /= len(val_loader)
        val_cls_loss /= len(val_loader)

        val_total_losses.append(val_loss)
        val_seg_losses.append(val_seg_loss)
        val_cls_losses.append(val_cls_loss)
        # Update learning rate
        scheduler.step(val_loss)

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_multitask_model.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}]:")
        print(f"  Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Cls: {train_cls_loss:.4f})")
        print(f"  Val Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Cls: {val_cls_loss:.4f})")

    plot_individual_loss_curves(
    train_total_losses,
    val_total_losses,
    train_seg_losses,
    val_seg_losses,
    train_cls_losses,
    val_cls_losses,
    save_path="/well/rittscher/users/ycr745/nnUNet/nnunetv2/multi_results"
    )       
def main():
    # Paths
    image_dir = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/imagesTr"
    seg_dir = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/gt_segmentations"
    label_file = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/image_to_subtype.json"
    split_file = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/splits_final.json"
    # preprocessed_dataset_folder = '/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed'

    save_path = Path("/well/rittscher/users/ycr745/nnUNet/nnunetv2/multi_results")
    save_path.mkdir(parents=True, exist_ok=True)

    # Create Train DataLoader
    train_loader = create_dataloader(
        image_dir=image_dir,
        seg_dir=seg_dir,
        label_file=label_file,
        split_file=split_file,
        split="train",
        mode="train",
        batch_size=1
    )

    # Create Validation DataLoader
    val_loader = create_dataloader(
        image_dir=image_dir,
        seg_dir=seg_dir,
        label_file=label_file,
        split_file=split_file,
        split="val",
        mode="val",
        batch_size=1
    )
    # for batch in train_loader:
    #     images = batch['image']  # Tensor of shape [B, 1, D, H, W]
    #     segmentations = batch['segmentation']  # Tensor of shape [B, D, H, W]
    #     cls_labels = batch['cls_label']  # Tensor of shape [B]
        
    #     print(f"Images: {images.shape}, Segmentations: {segmentations.shape}, Classification Labels: {cls_labels}")
    #     break
    loss_fn = MultiTaskLoss(
    dice_kwargs={"smooth": 1e-5, "include_background": False},
    ce_kwargs={"class_weights": torch.tensor([1.0, 0.5, 0.8])},
    cls_loss_weight=0.5,
    seg_loss_weight=0.5,
    seg_weights=[0.5, 0.35, 0.25, 0.15, 0.1] 
    )

    model = load_and_modify_multitask_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=500,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        )
if __name__ == "__main__":
    main()