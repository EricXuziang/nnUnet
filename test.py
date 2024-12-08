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
import numpy as np

from scipy import ndimage
from torchvision.models.video import r3d_18

# Custom dataset for 3D CT data
class CTDataset(Dataset):
    def __init__(self, data_dir, split_file, label_file, mode="train", transform=None):
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.data_dir = Path(data_dir)
        self.cases = splits[0][mode]
        with open(label_file, 'r') as f:
            self.labels = json.load(f)[mode]
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case_id = self.cases[idx]
        img_path = self.data_dir / f"{case_id}_0000.nii.gz"
        label = self.labels[case_id]
        img = nib.load(str(img_path)).get_fdata()
        # img = normalize(img)
        # img = resize_volume(img)
        # img = np.clip(img, 0, 1)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # print(f"Image shape: {img.shape}")
        img = F.interpolate(img.unsqueeze(0), size=(128, 128, 128), mode='trilinear', align_corners=False).squeeze(0)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# Classification head with dropout
class ClassificationHead(nn.Module):
    def __init__(self, input_features, num_classes=3, dropout_rate=0.5):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_features, num_classes)
        )

    def forward(self, x):
        x = x.mean(dim=[2, 3, 4])  # Global Average Pooling
        return self.fc(x)

# EncoderClassifier model
class EncoderClassifier(nn.Module):
    def __init__(self, encoder, num_classes=3, dropout_rate=0.5):
        super(EncoderClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = ClassificationHead(320, num_classes, dropout_rate)  # Adjust input features if necessary

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
def plot_loss_curve(train_losses, val_losses, save_path):
    """
    绘制训练和验证损失曲线并保存图像。
    
    Args:
        train_losses (list): 训练损失列表。
        val_losses (list): 验证损失列表。
        save_path (str or Path): 保存图像的路径。
    """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(Path(save_path) / 'loss_curve_new1.png')
    plt.close()

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, save_path):
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print(f"Epoch {epoch + 1}/{num_epochs} - {phase} Loss: {epoch_loss:.4f}")

            if phase == 'train':
                train_losses.append(epoch_loss)
            else:
                val_losses.append(epoch_loss)
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), Path(save_path) / 'best_new1.pth')
    
    torch.save(model.state_dict(), Path(save_path) / 'final_new1.pth')
    plot_loss_curve(train_losses, val_losses, save_path)

def load_and_modify_model(checkpoint_path=None, num_classes=3, dropout_rate=0.5):
    # Load the full model checkpoint
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
    if checkpoint_path is not None:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        full_model.load_state_dict(checkpoint['network_weights'])
    else:
        print("No checkpoint provided. Model initialized without pre-trained weights.")

    # Extract the encoder from the full model
    encoder = full_model.encoder  # Modify if the checkpoint structure differs
    encoder = nn.Sequential(encoder.stem, encoder.stages)  # Encoder consists of stem + stages

    # Create the modified model
    model = EncoderClassifier(encoder, num_classes=num_classes, dropout_rate=dropout_rate)
    print("Model successfully modified with classification head.")
    return model

def normalize(volume):
    """对图像数据进行归一化处理"""
    # 设置最小和最大阈值
    min_val = -1000
    max_val = 400
    # 将低于最小阈值的数据设置为最小阈值
    volume[volume < min_val] = min_val
    # 将高于最大阈值的数据设置为最大阈值
    volume[volume > max_val] = max_val
    # 进行归一化
    volume = (volume - min_val) / (max_val - min_val)
    # 转换数据类型为float32
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def main():
    # Paths
    data_dir = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/imagesTr"
    split_file = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/splits_final.json"
    label_file = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/image_to_subtype.json"
    save_path = Path("/well/rittscher/users/ycr745/nnUNet/nnunetv2/cls_results")
    save_path.mkdir(parents=True, exist_ok=True)

    # Load pretrained encoder
    checkpoint_path = '/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_results/Dataset001_quiz/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/checkpoint_best.pth'

    model = load_and_modify_model(checkpoint_path, num_classes=3, dropout_rate=0.5).to("cuda" if torch.cuda.is_available() else "cpu")

    
    # local_weights_path = '/well/rittscher/users/ycr745/nnUNet/nnunetv2/cls_results/r3d_18-b3b3357e.pth'
    # model = r3d_18(weights=None)
    # state_dict = torch.load(local_weights_path)
    # model.load_state_dict(state_dict)
    # model.fc = nn.Linear(model.fc.in_features, 3)
    # origc = model.stem[0]
    # model.stem[0] = torch.nn.Conv3d(1, origc.out_channels, kernel_size=origc.kernel_size, stride=origc.stride, padding=origc.padding, bias=origc.bias)
    # with torch.no_grad():
    #     model.stem[0].weight.data = origc.weight.data.sum(dim=1, keepdim=True)
    # model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets and Dataloaders
    train_transforms = tio.Compose([
        transforms.Normalize(mean=0.0, std=1.0),
        # NormalizeTransform(),
        tio.transforms.RandomAffine(scales=(1, 1), degrees=(10, 10, 10)),
        tio.transforms.RescaleIntensity((0, 1)),
        tio.transforms.RandomNoise(mean=0, std=0.05),
    ])

    val_transforms = tio.Compose([
        # NormalizeTransform(),
        transforms.Normalize(mean=0.0, std=1.0), 
    ])

    train_dataset = CTDataset(data_dir, split_file, label_file, mode="train", transform=train_transforms)
    val_dataset = CTDataset(data_dir, split_file, label_file, mode="val", transform=val_transforms)
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    }

    print(f"Trainset: {len(train_dataset)}, Valset: {len(val_dataset)}")

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    train_model(model, dataloaders, criterion, optimizer, num_epochs=200, device="cuda", save_path=save_path)


if __name__ == "__main__":
    main()