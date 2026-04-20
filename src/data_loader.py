import nibabel as nib
import numpy as np
from torch.utils.data import Dataset

from preprocessing import normalize, resize

class BraTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        """
        image_paths: list of MRI file paths (.nii)
        mask_paths: list of mask file paths (.nii)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load MRI and mask
        image = nib.load(self.image_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # Take middle slice (2D approach from your thesis)
        image = image[:, :, image.shape[2] // 2]
        mask = mask[:, :, mask.shape[2] // 2]

        # Normalize and resize
        image = normalize(image)
        image = resize(image)

        mask = resize(mask)

        # Expand dims for CNN (1 channel)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return image.astype(np.float32), mask.astype(np.float32)