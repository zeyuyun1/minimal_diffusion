import pytorch_lightning as pl
import torch
import math
import os
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder, STL10
from datasets import load_dataset
from scipy.io import loadmat

import os
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class HFTinyImageNetDataset(Dataset):
    """
    Wraps a Hugging Face Dataset to return (image, label) tuples 
    and apply standard torchvision transforms.
    """
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']

        # Ensure image is RGB (some rare images in subsets might be grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "/home/zeyu/data", batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()
        
        # Tiny-ImageNet is natively 64x64. Standard normalizations can be added here.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        # Download and cache the dataset from Hugging Face
        load_dataset("zh-plus/tiny-imagenet", cache_dir=self.hparams.data_dir)

    def setup(self, stage: str = None):
        # Load from cache
        dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir=self.hparams.data_dir)

        # 1. Training & Validation
        if stage == "fit" or stage is None:
            self.tiny_train = HFTinyImageNetDataset(dataset['train'], transform=self.transform)
            # HF's Tiny-ImageNet usually provides 'valid' instead of 'val'
            self.tiny_val = HFTinyImageNetDataset(dataset['valid'], transform=self.transform)

        # 2. Test Dataset
        if stage == "test" or stage is None:
            # Tiny ImageNet test labels are not publicly available in the standard format, 
            # so the validation set is conventionally used as the test set in most implementations.
            # If the HF dataset has a specific 'test' split, we use it; otherwise fallback to 'valid'.
            if 'test' in dataset:
                self.tiny_test = HFTinyImageNetDataset(dataset['test'], transform=self.transform)
            else:
                self.tiny_test = HFTinyImageNetDataset(dataset['valid'], transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.tiny_train, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.tiny_val, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.tiny_test, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers,
            pin_memory=True
        )
        

class SubsetWithTransform(Dataset):
    """Wraps a Subset so you can apply a different transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 3,
        img_size: int = 64,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
        grayscale: bool = False,
        no_resize: bool = False,
        random_crop: bool = False,
    ):
        super().__init__()
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.img_size    = img_size
        self.val_split   = val_split
        self.test_split  = test_split
        self.seed        = seed
        self.grayscale   = grayscale
        self.no_resize   = no_resize
        self.random_crop = random_crop

        # define transforms
        if self.no_resize:
            common_pre = []
        else:
            # If random_crop is enabled, we still resize the shorter side to img_size
            # to ensure the crop fits, then apply RandomCrop.
            if self.random_crop:
                common_pre = [
                    # transforms.Resize(self.img_size),
                    transforms.RandomCrop(self.img_size),
                ]
            else:
                common_pre = [
                    transforms.Resize(self.img_size),
                    transforms.CenterCrop(self.img_size),
                ]
        grayscale_tf = [transforms.Grayscale(num_output_channels=1)] if self.grayscale else []
        to_tensor_tf = [transforms.ToTensor()]

        self.train_transform = transforms.Compose(common_pre + grayscale_tf + to_tensor_tf)
        self.val_transform   = transforms.Compose(common_pre + grayscale_tf + to_tensor_tf)

    def prepare_data(self):
        # no download step for a local ImageFolder
        pass

    def setup(self, stage=None):
        # only split once
        if not hasattr(self, "train_dataset"):
            full = ImageFolder(self.data_dir, transform=None)
            total = len(full)
            val_len  = int(total * self.val_split)
            test_len = int(total * self.test_split)
            train_len = total - val_len - test_len

            generator = torch.Generator().manual_seed(self.seed)
            train_sub, val_sub, test_sub = random_split(
                full,
                [train_len, val_len, test_len],
                generator=generator
            )

            # wrap subsets to apply the correct transform
            self.train_dataset = SubsetWithTransform(train_sub, self.train_transform)
            self.val_dataset   = SubsetWithTransform(val_sub,   self.val_transform)
            self.test_dataset  = SubsetWithTransform(test_sub,  self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True,num_workers=31)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class STL10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()
        
        # Standard Normalization for STL-10
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ])

    def prepare_data(self):
        # Download all necessary splits
        STL10(self.hparams.data_dir, split='train', download=True)
        STL10(self.hparams.data_dir, split='test', download=True)
        STL10(self.hparams.data_dir, split='unlabeled', download=True)

    def setup(self, stage: str = None):
        # 1. Labeled Training & Validation
        if stage == "fit" or stage is None:
            stl_full = STL10(self.hparams.data_dir, split='train', transform=self.transform)
            self.stl_train, self.stl_val = random_split(stl_full, [4500, 500])
            
            # 2. Unlabeled Dataset (100,000 images)
            self.stl_unlabeled = STL10(
                self.hparams.data_dir, 
                split='unlabeled', 
                transform=self.transform
            )

        # 3. Test Dataset
        if stage == "test" or stage is None:
            self.stl_test = STL10(self.hparams.data_dir, split='test', transform=self.transform)

    def train_dataloader(self):
        # Use unlabeled (100k) for denoising training; labeled train is only 5k
        return DataLoader(
            self.stl_unlabeled,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def unlabeled_dataloader(self):
        # Alias for the same 100k unlabeled images (for scripts that call this explicitly)
        return self.train_dataloader()

    def val_dataloader(self):
        return DataLoader(self.stl_val, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.stl_test, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)


class CleanPacManDataset(Dataset):
    """Toy 1D Pac-Man manifold vectors used in the notebook experiments."""

    def __init__(self, num_samples=5000, d_dim=128, chunk_length=0):
        self.num_samples = num_samples
        self.d_dim = d_dim
        self.chunk_length = chunk_length

        self.grid = torch.linspace(0, 2 * math.pi, d_dim + 1)[:-1]
        mu = torch.rand(num_samples, 1) * 2 * math.pi
        mouth_size = torch.rand(num_samples, 1)
        self.manifold_coords = torch.cat([mu, mouth_size], dim=1)

        freq = 4.0 - 2.5 * mouth_size
        sigma = 0.8 / freq
        local_x = ((self.grid - mu + math.pi) % (2 * math.pi)) - math.pi
        wave = -(
            1.0 * torch.cos(1 * freq * local_x)
            + 0.5 * torch.cos(2 * freq * local_x)
            + 0.25 * torch.cos(3 * freq * local_x)
        ) / 1.75
        envelope = torch.exp(-(local_x ** 2) / (2 * sigma ** 2))
        self.data = wave * envelope

    def get_grid_samples(self, mu_steps=36, size_steps=10):
        mu_vals = torch.linspace(0, 2 * math.pi, mu_steps + 1)[:-1]
        size_vals = torch.linspace(0, 1, size_steps)
        mu_grid, size_grid = torch.meshgrid(mu_vals, size_vals, indexing="ij")
        mu_flat = mu_grid.flatten().unsqueeze(1)
        size_flat = size_grid.flatten().unsqueeze(1)
        grid_coords = torch.cat([mu_flat, size_flat], dim=1)

        freq = 4.0 - 2.5 * size_flat
        sigma = 0.8 / freq
        local_x = ((self.grid - mu_flat + math.pi) % (2 * math.pi)) - math.pi
        wave = -(
            1.0 * torch.cos(1 * freq * local_x)
            + 0.5 * torch.cos(2 * freq * local_x)
            + 0.25 * torch.cos(3 * freq * local_x)
        ) / 1.75
        envelope = torch.exp(-(local_x ** 2) / (2 * sigma ** 2))
        grid_data = wave * envelope
        return grid_data, grid_coords

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.chunk_length == 0:
            return self.data[idx], self.manifold_coords[idx]
        indices = [(idx + i) % self.num_samples for i in range(self.chunk_length)]
        return self.data[indices], self.manifold_coords[indices]


class ToyPacManDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_samples: int = 10000,
        d_dim: int = 128,
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.d_dim = d_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

    def setup(self, stage=None):
        if not hasattr(self, "train_dataset"):
            full = CleanPacManDataset(num_samples=self.num_samples, d_dim=self.d_dim, chunk_length=0)
            total = len(full)
            val_len = int(total * self.val_split)
            test_len = int(total * self.test_split)
            train_len = total - val_len - test_len
            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                full,
                [train_len, val_len, test_len],
                generator=generator,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )


def patchify(patch_size: int, images: torch.Tensor, stride: int):
    """Create [B, N, C, P, P] patches from [B, C, H, W] images."""
    if images.ndim != 4:
        raise ValueError(f"images must be rank-4 [B,C,H,W], got {images.shape}")
    b, c, h, w = images.shape
    p = int(patch_size)
    if p <= 0:
        raise ValueError("patch_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if h < p or w < p:
        raise ValueError(f"patch_size={p} must fit image shape HxW={h}x{w}")

    patches = images.unfold(2, p, stride).unfold(3, p, stride)  # [B, C, nH, nW, P, P]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, nH, nW, C, P, P]
    return patches.view(b, -1, c, p, p)


class FieldDataset(Dataset):
    def __init__(self, root: str, patch_size: int = 8, stride: int = None, filter_threshold: float = 0.01):
        self.P = patch_size
        self.C=1
        if stride is None:
            stride = patch_size

        root = os.path.expanduser(root)
        os.makedirs(root, exist_ok=True)
        
        if not os.path.exists(f"{root}/field.mat"):
            # Using standard python requests or urllib is often safer than os.system
            os.system(f"wget -P {root} https://rctn.org/bruno/sparsenet/IMAGES.mat")
            os.system(f"mv {root}/IMAGES.mat {root}/field.mat")
            
        self.images = torch.tensor(loadmat(f"{root}/field.mat")["IMAGES"]).float()
        self.images = self.images - self.images.mean() 
        
        # 2. Reshape to [B, C, H, W]
        self.images = self.images.permute(2, 0, 1).unsqueeze(1) 

        # 3. Patchify
        patches = patchify(patch_size, self.images, stride) # [B, N, C, P, P]
        patches = patches.reshape(-1, self.C, self.P, self.P)

        # 4. FILTER EMPTY PATCHES
        # Calculate variance for each patch. If it's too flat, drop it.
        patch_vars = patches.var(dim=(1, 2, 3)) 
        self.patches = patches[patch_vars > filter_threshold]

        # 5. NORMALIZE RANGE
        # For EDM/Diffusion, it is critical that the data std is roughly 1.0 
        # or matches your sigma_data (0.25).
        self.patches = (self.patches - self.patches.mean()) / (self.patches.std() + 1e-8)
        self.patches = self.patches * 0.25 # Scale to match your sigma_data

        print(f"Dataset initialized. Remaining patches after filtering: {len(self.patches)}")

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        return self.patches[idx], torch.tensor(0, dtype=torch.long)