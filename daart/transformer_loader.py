# daart/transformer_loader.py

import os
import cv2
import yaml
import warnings
import numpy as np
import torch

from functools import lru_cache
from torch.cuda.amp import autocast
from torchvision import transforms
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from daart.model.vit_mae.vit_mae import ImageEncoderViTMAE
import accelerate

MAX_FRAMES = 100000

# ─── 1) DALI pipeline (GPU decode + resize) ─────────────────────────
@pipeline_def
def video_pipeline(video_files, sequence_length):
    video = fn.readers.video(
        device="gpu",
        filenames=video_files,
        sequence_length=sequence_length,
        stride=1,
        shard_id=0,
        num_shards=1,
        random_shuffle=False,
        normalized=False,
        dtype=types.UINT8,
        initial_fill=16,
        prefetch_queue_depth=2,
    )
    # Resize to 224×224
    video = fn.resize(video, resize_x=224, resize_y=224)
    return video  # [sequence_length, H, W, C] on GPU


def create_single_video_loader(
    video_file: str,
    batch_size: int,
    sequence_length: int,
    num_threads: int = 32,
    device_id: int = 0
):
    """Returns (dali_iter, total_frames, seq_len), truncating at MAX_FRAMES."""
    if not os.path.isfile(video_file):
        raise FileNotFoundError(f"Video not found: {video_file}")

    cap = cv2.VideoCapture(video_file)
    actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if actual > MAX_FRAMES:
        warnings.warn(
            f"{video_file} has {actual} frames; truncating to {MAX_FRAMES}"
        )
        total_frames = MAX_FRAMES
    else:
        total_frames = actual

    seq_len = min(sequence_length, total_frames)

    pipe = video_pipeline(
        video_files=[video_file],
        sequence_length=seq_len,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        exec_async=True,
        exec_pipelined=True,
        prefetch_queue_depth=2,
    )
    pipe.build()

    n_batches = (total_frames + seq_len - 1) // seq_len
    loader = DALIGenericIterator(
        pipe, ["video"], size=n_batches,
        auto_reset=False, dynamic_shape=True
    )
    return loader, total_frames, seq_len


# ─── 2) Lazy‐loaded ViT‐MAE (HF + optional ckpt + accelerate) ───────
@lru_cache(maxsize=2)
def get_transformer_model(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda"
):
    cfg = yaml.safe_load(open(config_path))
    cfg["mask_ratio"] = 0.0
    model = ImageEncoderViTMAE(config=cfg)
    model.vit_mae.from_pretrained("facebook/vit-mae-base")

    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        filtered = {
            k.replace("vit_mae.", ""): v
            for k, v in ckpt.items()
            if k.startswith("vit_mae.")
            and k.replace("vit_mae.", "") in model.vit_mae.state_dict()
        }
        model.vit_mae.load_state_dict(filtered, strict=False)

    # wrap in accelerate for DDP-safety
    kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accel = accelerate.Accelerator(kwargs_handlers=[kwargs])
    model = accel.prepare(model)
    model.eval().to(device)
    return model


# ─── 3) Extract patch‐tokens only (no CLS) ──────────────────────────
def extract_patch_tokens(
    video_file: str,
    config_path: str,
    checkpoint_path: str,
    sequence_length: int = 1024,
    batch_size: int = 1,
    device: str = "cuda",
    num_threads: int = 32,
    device_id: int = 0
) -> np.ndarray:
    """
    Returns np.ndarray of shape (T, P*D), dropping the CLS token.
    Caps at MAX_FRAMES and caches the model load.
    """
    print('creating loader')
    loader, total_frames, seq_len = create_single_video_loader(
        video_file, batch_size, sequence_length, num_threads, device_id
    )
    print('getting model')
    model = get_transformer_model(config_path, checkpoint_path, device)

    # Use ImageNet stats
    norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    all_feats = []
    for batch in loader:
        # batch[0]['video']: [T, H, W, C]
        vid = batch[0]["video"][0]
        # to [T, C, H, W] float in [0,1]
        x = vid.permute(0, 3, 1, 2).float().div(255.0).to(device)
        x = norm(x)

        with torch.no_grad(), autocast():
            out = model(x)        # [T, P+1, D]

        # drop CLS token, flatten patches: [T, P*D]
        patches = out[:, 1:, :].reshape(x.shape[0], -1)
        all_feats.append(patches.cpu().numpy())

        # free GPU mem each batch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    feats = np.concatenate(all_feats, axis=0)[:total_frames]
    return feats
