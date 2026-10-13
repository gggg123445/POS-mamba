"""Multiview file discovery, long-image preprocessing and crop-only batching."""
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .utils import parse_stenosis_value

_IMAGE_ROOT_FILELIST_CACHE = {}

def _list_image_root_files(image_root: str) -> list:
    cached = _IMAGE_ROOT_FILELIST_CACHE.get(image_root)
    if cached is not None:
        return cached

    files = []
    try:
        with os.scandir(image_root) as it:
            for e in it:
                if e.is_file():
                    files.append(e.name)
    except FileNotFoundError:
        files = []

    _IMAGE_ROOT_FILELIST_CACHE[image_root] = files
    return files


def find_multiview_images(image_root: str, record_id: str, scan_type: Optional[str] = None) -> list:
    rid = str(record_id).strip()
    st = str(scan_type).strip() if scan_type else ''
    pattern = f"{rid}_" + (f"{st}_" if st else '')
    pattern_l = pattern.lower()

    candidates = []
    for filename in _list_image_root_files(image_root):
        fl = filename.lower()
        if not fl.startswith(pattern_l):
            continue
        if not (fl.endswith('.png') or fl.endswith('.jpg') or fl.endswith('.jpeg')):
            continue
        view_match = re.search(r'view[_-]?(\d+)', fl)
        if view_match:
            view_num = int(view_match.group(1))
            candidates.append((filename, view_num))

    candidates.sort(key=lambda x: x[1])
    return [filename for filename, _ in candidates]


def sample_views(view_files: list, num_views: int, sampling_method: str = 'uniform') -> list:
    if num_views is None or num_views <= 0:
        return view_files
    if len(view_files) <= num_views:
        return view_files

    if sampling_method == 'uniform':
        indices = np.linspace(0, len(view_files) - 1, num_views, dtype=int)
        return [view_files[i] for i in indices]
    if sampling_method == 'random':
        indices = np.random.choice(len(view_files), num_views, replace=False)
        indices.sort()
        return [view_files[i] for i in indices]
    if sampling_method == 'first':
        return view_files[:num_views]
    if sampling_method == 'step':
        step = max(1, len(view_files) // num_views)
        return view_files[::step][:num_views]

    raise ValueError(f"Unknown sampling_method: {sampling_method}")


def center_crop_tensor_chw(x: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    _, h, w = x.shape
    out_h = min(int(out_h), int(h))
    out_w = min(int(out_w), int(w))
    top = max(0, (h - out_h) // 2)
    left = max(0, (w - out_w) // 2)
    return x[:, top:top + out_h, left:left + out_w]


class VesselLongImageDatasetNativeNoPad(Dataset):
    """
    将多视角血管图按顺序拼接成长图，直接送入 MambaVision。
    不做滑窗，不做序列 SSM。
    """
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: str,
        preprocess,
        num_views: int = 30,
        sampling_method: str = 'uniform',
        scan_type_filter: Optional[str] = None,
        center_crop_h_ratio: float = 1.0,
        center_crop_w_px: int = 40,
        seq_fixed_h: int = 0,
        seq_post_interp_h: int = 0,
        min_side_fallback: int = 32,
        stenosis_col: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.preprocess = preprocess
        self.num_views = int(num_views)
        self.sampling_method = sampling_method
        self.scan_type_filter = scan_type_filter
        self.center_crop_h_ratio = float(center_crop_h_ratio)
        self.center_crop_w_px = int(center_crop_w_px)
        self.seq_fixed_h = int(seq_fixed_h)
        self.seq_post_interp_h = int(seq_post_interp_h)
        self.min_side_fallback = int(min_side_fallback)
        self.stenosis_col = stenosis_col

    def __len__(self):
        return len(self.df)

    def _center_crop_vessel(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        crop_h = min(h, max(1, int(round(h * self.center_crop_h_ratio))))
        if self.center_crop_w_px > 0:
            crop_w = min(int(self.center_crop_w_px), w)
        else:
            crop_w = w
        left = max(0, (w - crop_w) // 2)
        top = max(0, (h - crop_h) // 2)
        return img.crop((left, top, left + crop_w, top + crop_h))

    def _load_ordered_views(self, row) -> Tuple[List[str], str]:
        record_id = str(row.get('record_id', row.get('uniq_ID', '')))
        view_files = find_multiview_images(self.image_root, record_id, self.scan_type_filter)
        selected_files = sample_views(view_files, self.num_views, self.sampling_method)
        if not selected_files:
            raise FileNotFoundError(f'No matching multiview images for record_id={record_id!r}')
        file_paths = [os.path.join(self.image_root, f) for f in selected_files]
        return file_paths, record_id

    def _build_long_image_tensor(self, file_paths: List[str]) -> torch.Tensor:
        if len(file_paths) == 0:
            raise ValueError('Cannot build a long image without views')

        vessel_parts = []
        for p in file_paths:
            with Image.open(p) as source:
                img = source.convert('L')
            center = self._center_crop_vessel(img)
            center = center.transpose(Image.Transpose.TRANSPOSE)
            part = self.preprocess(center) if self.preprocess is not None else T.ToTensor()(center)
            vessel_parts.append(part)

        parts_aligned = []
        if self.seq_fixed_h > 0:
            target_h = max(8, int(self.seq_fixed_h))
            for t in vessel_parts:
                h, w = int(t.shape[-2]), int(t.shape[-1])
                new_w = max(8, int(round(w * (target_h / max(h, 1)))))
                t_resized = F.interpolate(
                    t.unsqueeze(0),
                    size=(target_h, new_w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
                parts_aligned.append(t_resized)
        else:
            min_h = min(int(t.shape[-2]) for t in vessel_parts)
            parts_aligned = [center_crop_tensor_chw(t, min_h, int(t.shape[-1])) for t in vessel_parts]

        long_img = torch.cat(parts_aligned, dim=-1)  # [1, H, W_long]

        if self.seq_post_interp_h > 0:
            target_h_post = max(8, int(self.seq_post_interp_h))
            long_img = F.interpolate(
                long_img.unsqueeze(0),
                size=(target_h_post, int(long_img.shape[-1])),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

        return long_img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row.get('mapped_label', 0))
        file_paths, record_id = self._load_ordered_views(row)
        long_img = self._build_long_image_tensor(file_paths)

        if self.stenosis_col is not None:
            stenosis = parse_stenosis_value(row.get(self.stenosis_col, 0.0))
            stenosis = float(np.clip(stenosis, 0.0, 100.0))
            return long_img, label, stenosis, record_id

        return long_img, label, record_id


def collate_native_nopad(batch):
    images_list, labels, record_ids = zip(*batch)

    min_h = min(int(x.shape[-2]) for x in images_list)
    min_w = min(int(x.shape[-1]) for x in images_list)

    cropped = []
    for x in images_list:
        cropped.append(center_crop_tensor_chw(x, min_h, min_w))

    images = torch.stack(cropped, dim=0)  # [B, 1, H, W]
    labels = torch.tensor(labels, dtype=torch.long)
    record_ids = list(record_ids)
    return images, labels, record_ids


def collate_native_nopad_with_stenosis(batch):
    images_labels = [(x[0], x[1], x[3]) for x in batch]
    images, labels, record_ids = collate_native_nopad(images_labels)
    stenosis = torch.tensor([float(x[2]) for x in batch], dtype=torch.float32)
    return images, labels, stenosis, record_ids


def build_native_noresize_transforms(enable_aug: bool = True):
    train_tfms = []
    if enable_aug:
        train_tfms += [
            T.RandomApply([T.ColorJitter(brightness=0.12, contrast=0.12)], p=0.5),
            T.RandomApply([T.RandomAffine(degrees=4, translate=(0.02, 0.02), shear=2)], p=0.5),
        ]
    train_tfms += [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    val_tfms = [T.ToTensor(), T.Normalize(mean=[0.5], std=[0.5])]
    return T.Compose(train_tfms), T.Compose(val_tfms)
