# import torch
# import numpy as np
# import os
# from torch.utils.data import Dataset
# from torchvision.datasets import ImageFolder


# class CustomDataset(Dataset):
#     def __init__(self, feature_dir, label_dir):
#         self.feature_dir = feature_dir
#         self.label_dir = label_dir
#         self.flip = 'flip' in self.feature_dir

#         aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
#         aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
#         if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
#             self.aug_feature_dir = aug_feature_dir
#             self.aug_label_dir = aug_label_dir
#         else:
#             self.aug_feature_dir = None
#             self.aug_label_dir = None

#         # self.feature_files = sorted(os.listdir(feature_dir))
#         # self.label_files = sorted(os.listdir(label_dir))
#         # TODO: make it configurable
#         self.feature_files = [f"{i}.npy" for i in range(1281167)]
#         self.label_files = [f"{i}.npy" for i in range(1281167)]

#     def __len__(self):
#         assert len(self.feature_files) == len(self.label_files), \
#             "Number of feature files and label files should be same"
#         return len(self.feature_files)

#     def __getitem__(self, idx):
#         if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
#             feature_dir = self.aug_feature_dir
#             label_dir = self.aug_label_dir
#         else:
#             feature_dir = self.feature_dir
#             label_dir = self.label_dir
                   
#         feature_file = self.feature_files[idx]
#         label_file = self.label_files[idx]

#         features = np.load(os.path.join(feature_dir, feature_file))
#         if self.flip:
#             aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
#             features = features[:, aug_idx]
#         labels = np.load(os.path.join(label_dir, label_file))
#         return torch.from_numpy(features), torch.from_numpy(labels)


# def build_imagenet(args, transform):
#     return ImageFolder(args.data_path, transform=transform)

# def build_imagenet_code(args):
#     feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
#     label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
#     assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
#         f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
#     return CustomDataset(feature_dir, label_dir)

import torch
import numpy as np
import os
from glob import glob
import h5py
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import glob

class CustomDataset(Dataset):
    # Your existing CustomDataset class remains unchanged
    def __init__(self, feature_dir, label_dir):
        self.feature_dir = feature_dir
        self.label_dir = label_dir
        self.flip = 'flip' in self.feature_dir

        aug_feature_dir = feature_dir.replace('ten_crop/', 'ten_crop_105/')
        aug_label_dir = label_dir.replace('ten_crop/', 'ten_crop_105/')
        if os.path.exists(aug_feature_dir) and os.path.exists(aug_label_dir):
            self.aug_feature_dir = aug_feature_dir
            self.aug_label_dir = aug_label_dir
        else:
            self.aug_feature_dir = None
            self.aug_label_dir = None

        self.feature_files = [f"{i}.npy" for i in range(1281167)]
        self.label_files = [f"{i}.npy" for i in range(1281167)]

    def __len__(self):
        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files should be same"
        return len(self.feature_files)

    def __getitem__(self, idx):
        if self.aug_feature_dir is not None and torch.rand(1) < 0.5:
            feature_dir = self.aug_feature_dir
            label_dir = self.aug_label_dir
        else:
            feature_dir = self.feature_dir
            label_dir = self.label_dir
                   
        feature_file = self.feature_files[idx]
        label_file = self.label_files[idx]

        features = np.load(os.path.join(feature_dir, feature_file))
        if self.flip:
            aug_idx = torch.randint(low=0, high=features.shape[1], size=(1,)).item()
            features = features[:, aug_idx]
        labels = np.load(os.path.join(label_dir, label_file))
        return torch.from_numpy(features), torch.from_numpy(labels)


class CosmosTokenDataset(Dataset):
    def __init__(self, h5_root_path="/fs/cml-projects/yet-another-diffusion/cosmos-imagenet-shards"):
        self.h5_root_path = h5_root_path
        
        # Add these attributes to match original dataset interface
        self.flip = False
        self.feature_dir = h5_root_path  # Used in logging
        self.aug_feature_dir = None
        
        # Rest of your initialization code
        self.h5_files = []
        for folder in sorted(glob.glob(os.path.join(h5_root_path, "folders_*"))):
            self.h5_files.extend(sorted(glob.glob(os.path.join(folder, "*.h5"))))
        
        # Create class mapping from actual ImageNet folders
        imagenet_path = "/fs/cml-datasets/ImageNet/ILSVRC2012/train"
        class_folders = sorted(os.listdir(imagenet_path))
        self.class_to_idx = {folder: idx for idx, folder in enumerate(class_folders)}
        print(f"Found {len(self.class_to_idx)} ImageNet classes")
        
        # Pre-compute file indices
        self.file_indices = []
        self.total_samples = 0
        
        for h5_file in self.h5_files:
            with h5py.File(h5_file, 'r') as f:
                num_samples = len(f['latents'])
                self.file_indices.append((h5_file, self.total_samples, self.total_samples + num_samples))
                self.total_samples += num_samples
        
        print(f"Found {self.total_samples} samples across {len(self.h5_files)} files")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Find which file contains this index
        for h5_file, start_idx, end_idx in self.file_indices:
            if start_idx <= idx < end_idx:
                with h5py.File(h5_file, 'r') as f:
                    file_idx = idx - start_idx
                    tokens = torch.from_numpy(f['latents'][file_idx]).long()  # [32, 32]
                    path = f['paths'][file_idx].decode('utf-8')
                    
                    # Get class folder name from path
                    class_name = path.split('/')[-2]
                    class_id = self.class_to_idx[class_name]
                    
                    return tokens.flatten(), torch.tensor(class_id)


def build_imagenet(args, transform):
    return ImageFolder(args.data_path, transform=transform)

def build_imagenet_code(args):
    if hasattr(args, 'cosmos_path') and args.cosmos_path is not None:
        return CosmosTokenDataset(args.cosmos_path)  # This version is correct
    else:
        # Original code path
        feature_dir = f"{args.code_path}/imagenet{args.image_size}_codes"
        label_dir = f"{args.code_path}/imagenet{args.image_size}_labels"
        assert os.path.exists(feature_dir) and os.path.exists(label_dir), \
            f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
        return CustomDataset(feature_dir, label_dir)