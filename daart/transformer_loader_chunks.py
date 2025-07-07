#!/usr/bin/env python3
# transformer_loader_dali.py

import os
import yaml
import torch
import numpy as np
import time
import warnings
from functools import lru_cache
from contextlib import contextmanager
from torch.cuda.amp import autocast
from daart.model.vit_mae.vit_mae import ImageEncoderViTMAE

# DALI imports
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline

# Filter out the specific deprecation warning
warnings.filterwarnings("ignore", message="batch_size is deprecated")

@contextmanager
def timer(name: str):
    t0 = time.time()
    yield
    print(f"[TIMER] {name:30s}: {time.time() - t0:.3f}s")

@lru_cache(maxsize=2)
def load_model(config_path: str, checkpoint_path: str, device: str):
    """
    Load & prepare the ViT-MAE model. Cached so we only ever do this once.
    """
    with timer("load config & build model"):
        cfg = yaml.safe_load(open(config_path))
        cfg['mask_ratio'] = 0.0
        model = ImageEncoderViTMAE(config=cfg)
        model.vit_mae.from_pretrained("facebook/vit-mae-base")
    with timer("load checkpoint"):
        if checkpoint_path and os.path.isfile(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            filtered = {
                k.replace('vit_mae.', ''): v
                for k, v in ckpt.items()
                if k.startswith('vit_mae.')
                and k.replace('vit_mae.', '') in model.vit_mae.state_dict()
            }
            model.vit_mae.load_state_dict(filtered, strict=False)
    with timer("to device & eval"):
        model = model.to(device)
        model.eval()
        # use half precision on GPU
        if device.startswith('cuda'):
            #model.half()
            # Set optimal CUDA flags
            torch.backends.cudnn.benchmark = True
    return model


def extract_patch_tokens_chunk_gpu(frames_tensor, config_path, checkpoint_path, max_imgs_per_pass=1024, device="cuda"):
    vit_model = load_model(config_path, checkpoint_path, device)
    vit_model.eval()
    # Process in chunks efficiently
    with torch.no_grad():
        # For smaller batches, process in one go
        if frames_tensor.shape[0] <= max_imgs_per_pass:
            out = vit_model(frames_tensor)
            #patches = out[:, 1:, :].reshape(out.shape[0], -1)
            patches = out[:, 0, :].reshape(out.shape[0], -1)
        else:
            # For larger batches, process in chunks and concatenate
            outs = []
            for i in range(0, frames_tensor.shape[0], max_imgs_per_pass):
                sub = frames_tensor[i:i + max_imgs_per_pass]
                out = vit_model(sub)
                # Extract patch tokens directly
                #sub_patches = out[:, 1:, :].reshape(out.shape[0], -1)
                sub_patches = out[:, 0, :].reshape(out.shape[0], -1)
                outs.append(sub_patches)
            
            # Concatenate results efficiently
            patches = torch.cat(outs, dim=0)
    
    # Return in the same precision as the model (half precision)
    # Let the caller decide when to convert to full precision
    return patches