"""Image dataset for the driver state classification baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


@dataclass(slots=True)
class SampleRecord:
    """Metadata for one image sample."""

    path: Path
    label_name: str
    label_index: int


class DriverStateImageDataset(Dataset[dict[str, Any]]):
    """Directory-based image dataset for four-class driver state classification."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        class_names: list[str],
        image_size: int,
        mean: list[float],
        std: list[float],
        augment: bool = False,
        augmentation_config: dict[str, Any] | None = None,
        return_path: bool = True,
        allowed_extensions: list[str] | None = None,
    ) -> None:
        """Initialize the dataset from a split directory."""
        self.root_dir = Path(root_dir)
        self.split = split
        self.class_names = class_names
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment
        self.augmentation_config = augmentation_config or {}
        self.return_path = return_path
        self.allowed_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (allowed_extensions or [".jpg", ".jpeg", ".png", ".bmp", ".webp"])
        }
        self.class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
        self.samples = self._build_samples()
        self.transform = self._build_transform()

    def _build_samples(self) -> list[SampleRecord]:
        """Scan the split directory and build a list of sample records."""
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples: list[SampleRecord] = []
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in self.allowed_extensions:
                    continue
                samples.append(
                    SampleRecord(
                        path=image_path,
                        label_name=class_name,
                        label_index=self.class_to_idx[class_name],
                    )
                )

        if not samples:
            raise ValueError(f"No image samples found under {split_dir}")
        return samples

    def _build_transform(self) -> transforms.Compose:
        """Create the image transform pipeline."""
        transform_steps: list[Any] = [transforms.Resize((self.image_size, self.image_size))]

        if self.augment:
            color_jitter_cfg = self.augmentation_config.get("color_jitter", {})
            transform_steps.extend(
                [
                    transforms.RandomHorizontalFlip(
                        p=float(self.augmentation_config.get("horizontal_flip", 0.5))
                    ),
                    transforms.ColorJitter(
                        brightness=float(color_jitter_cfg.get("brightness", 0.2)),
                        contrast=float(color_jitter_cfg.get("contrast", 0.2)),
                        saturation=float(color_jitter_cfg.get("saturation", 0.15)),
                        hue=float(color_jitter_cfg.get("hue", 0.02)),
                    ),
                    transforms.RandomRotation(
                        degrees=float(self.augmentation_config.get("rotation_degrees", 8))
                    ),
                ]
            )

        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return transforms.Compose(transform_steps)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one sample and return tensor plus metadata."""
        sample = self.samples[index]
        try:
            image = Image.open(sample.path).convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Failed to load image: {sample.path}") from error

        image_tensor: Tensor = self.transform(image)
        item: dict[str, Any] = {
            "image": image_tensor,
            "label": sample.label_index,
            "label_name": sample.label_name,
        }
        if self.return_path:
            item["path"] = str(sample.path)
        return item
