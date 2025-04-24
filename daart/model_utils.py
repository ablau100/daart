def transfer_weights(target_model, source_model):
    """
    Transfer weights from a source model to a target model, skipping parameters with shape mismatches.

    Args:
        target_model (torch.nn.Module): The model to which weights will be transferred.
        source_model (torch.nn.Module): The model from which weights will be transferred.
    """
    for (name, param), (source_name, source_param) in zip(
        target_model.named_parameters(), source_model.named_parameters()
    ):
        if param.shape == source_param.shape:
            param.data.copy_(source_param.data)
        else:
            print(f"Skip loading parameter {name} due to shape mismatch: {param.shape} vs {source_param.shape}")

