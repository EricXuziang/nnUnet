import os
import shutil

# make train/val dataset
source_train = "/well/rittscher/projects/3d_ziang/UHN-MedImg3D-ML-quiz/train"
source_val = "/well/rittscher/projects/3d_ziang/UHN-MedImg3D-ML-quiz/validation"
target_imagesTr = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/imagesTr"
target_labelsTr = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/labelsTr"
target_imagesVal = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/imagesVal"
target_labelsVal = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/labelsVal"


os.makedirs(target_imagesTr, exist_ok=True)
os.makedirs(target_labelsTr, exist_ok=True)
os.makedirs(target_imagesVal, exist_ok=True)
os.makedirs(target_labelsVal, exist_ok=True)

subtypes = ['subtype0', 'subtype1', 'subtype2']

def make_dataset(image_path, label_path, mode = 'train'):
    for subtype in subtypes:
        subtype_path_train = os.path.join(source_train, subtype)
        subtype_path_val = os.path.join(source_val, subtype)
        if mode == 'train':
            # Combine training and validation files for training mode
            subtype_files = sorted([f for f in os.listdir(subtype_path_train) if not f.startswith(".")]) + \
                            sorted([f for f in os.listdir(subtype_path_val) if not f.startswith(".")])
        elif mode == 'val':
            # Only validation files for validation mode
            subtype_files = sorted([f for f in os.listdir(subtype_path_val) if not f.startswith(".")])
        else:
            raise ValueError("Invalid mode. Use 'train' or 'val'.")
        
        for file in subtype_files:
            if file.endswith("_0000.nii.gz"):  
                print('image:',file)
                new_image_name = f"case_{file[7:10]}_0000.nii.gz"
                new_label_name = f"case_{file[7:10]}.nii.gz"
                label_file = file.replace("_0000.nii.gz", ".nii.gz")

                if file in os.listdir(subtype_path_train):
                    current_path = subtype_path_train
                else:
                    current_path = subtype_path_val

                shutil.copy(
                    os.path.join(current_path, file),
                    os.path.join(image_path, new_image_name)
                )
                print(f"Copied image: {new_image_name}")
                shutil.copy(
                    os.path.join(current_path, label_file),
                    os.path.join(label_path, new_label_name)
                )
                print(f"Copied label: {new_label_name}")

make_dataset(target_imagesVal, target_labelsVal, mode ='val')

# make dataset.json
# import json
# nnUNet_dir = '/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/' #

# def sts_json():
#     info = {
#         "channel_names": {
#             "0": "CT"
#         },
#         "labels": {
#             "background": 0,
#             "pancreas": 1,
#             "lesion": 2
#         },
#         "numTraining": 252,
#         "file_ending": ".nii.gz"
#     }
#     with open(nnUNet_dir + 'nnUNet_raw/Dataset001_quiz/dataset.json',
#               'w') as f:
#         json.dump(info, f, indent=4)

# sts_json()


# 修复标签文件
# import nibabel as nib
# import numpy as np
# import os

# # input label path
# labels_dir = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_raw/Dataset001_quiz/labelsTs"

# for label_file in os.listdir(labels_dir):
#     if label_file.endswith(".nii.gz"):
#         label_path = os.path.join(labels_dir, label_file)
        
#         img = nib.load(label_path)
#         data = img.get_fdata()
#         # float test
#         if not np.all(np.isin(data, [0, 1, 2])):
#             print(f"Fixing {label_file}")
                      
#             fixed_data = np.round(data).astype(np.int16)
            
#             fixed_img = nib.Nifti1Image(fixed_data, img.affine, img.header)
#             nib.save(fixed_img, label_path)


# make own train/val json
# from batchgenerators.utilities.file_and_folder_operations import load_json
# import json

# def get_case_ids(source_path):
#     case_ids = []
#     for subtype in subtypes:
#         subtype_path = os.path.join(source_path, subtype)
#         if os.path.exists(subtype_path):
#             for file in os.listdir(subtype_path):
#                 if file.endswith("_0000.nii.gz"):
#                     case_id = file.split("_")[2]
#                     if case_id not in case_ids:
#                         case_ids.append('case_' + case_id)
#     return sorted(case_ids)

# def generate_image_subtype_mapping(source_path, subtypes):
#     image_subtype_mapping = {}
#     for subtype_index, subtype in enumerate(subtypes):
#         subtype_path = os.path.join(source_path, subtype)
#         if os.path.exists(subtype_path):
#             for file in os.listdir(subtype_path):
#                 if file.endswith("_0000.nii.gz"):  
#                     case_id = file.split("_")[2]
#                     image_subtype_mapping['case_' + case_id] = subtype_index
#     return image_subtype_mapping

# make own train/val split(splits_final.json)
# train_cases = get_case_ids(source_train)
# val_cases = get_case_ids(source_val)

# data = [{
#     "train":  train_cases,
#     "val":  val_cases
# }]
# output_file = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/splits_final.json"
# with open(output_file, "w") as f:
#     json.dump(data, f, indent=4)

# make own train/val to subtype(image_to_subtype.json)
# train_image_subtype = generate_image_subtype_mapping(source_train, subtypes)
# val_image_subtype = generate_image_subtype_mapping(source_val, subtypes)
# image_subtype_mapping = {
#     "train": train_image_subtype,
#     "val": val_image_subtype
# }
# output_mapping_file = "/well/rittscher/users/ycr745/nnUNet/nnunetv2/dataset/nnUNet_preprocessed/Dataset001_quiz/image_to_subtype.json"
# with open(output_mapping_file, "w") as f:
#     json.dump(image_subtype_mapping, f, indent=4)
