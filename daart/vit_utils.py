def integrate_vit_patches_with_daart(model, frames, return_cls_token=True):
    """
    Extract ViT patch embeddings and format them for use with DAART's DTCN segmenter
    
    Args:
        model: The ViT model with loaded weights
        frames: Video frames to process
        return_cls_token: Whether to include the CLS token (default: True)
        
    Returns:
        Patch embeddings ready for DTCN segmenter input
    """
    with torch.no_grad():
        # Extract all tokens (patch embeddings + CLS token)
        embeddings = model.vit_mae.embeddings(frames)  # [batch_size, 197, 768]
        
        if not return_cls_token:
            # Return only patch tokens (excluding CLS token)
            patch_embeddings = embeddings[:, 1:, :]  # [batch_size, 196, 768]
            return patch_embeddings
        
        # Extract CLS and patch tokens
        batch_size = embeddings.shape[0]
        cls_token = embeddings[:, 0:1, :]  # [batch_size, 1, 768]
        patch_tokens = embeddings[:, 1:, :]  # [batch_size, 196, 768]
        
        # Reshape patch tokens to 2D grid (14×14 for 224×224 images with 16×16 patches)
        patch_tokens_spatial = patch_tokens.reshape(batch_size, 14, 14, 768)
        
        # Transpose to format expected by CNN [batch_size, channels, height, width]
        patch_tokens_spatial = patch_tokens_spatial.permute(0, 3, 1, 2)  # [batch_size, 768, 14, 14]
        
        # Add CLS token as an extra "pixel" at the top-left
        cls_spatial = cls_token.reshape(batch_size, 768, 1, 1)
        # Pad one row at the top
        padded_tokens = torch.nn.functional.pad(patch_tokens_spatial, (0, 0, 1, 0))
        # Place CLS token at the top-left position
        padded_tokens[:, :, 0, 0] = cls_spatial.squeeze(-1).squeeze(-1)
        
        return padded_tokens  # [batch_size, 768, 15, 14]