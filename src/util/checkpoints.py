import os

def resolve_checkpoint_path(model_name, checkpoint_name):
    checkpoint_dir = os.path.join("outputs", "checkpoints", model_name)

    if checkpoint_name == "last":
        ckpt_path = os.path.join(checkpoint_dir, "last.ckpt")

    elif checkpoint_name == "best":
        candidates = sorted(
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("best-model") and f.endswith(".ckpt")
        )
        if not candidates:
            raise FileNotFoundError("No best checkpoint found.")
        ckpt_path = os.path.join(checkpoint_dir, candidates[0])

    elif checkpoint_name.startswith("epoch_"):
        try:
            epoch_num = int(checkpoint_name.split("_")[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid format for epoch checkpoint: {checkpoint_name}. Expected format: epoch_<int>")

        pattern = f"epoch_epoch-{epoch_num:02d}"
        candidates = sorted(
            f for f in os.listdir(checkpoint_dir)
            if f.startswith(pattern) and f.endswith(".ckpt")
        )
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found for epoch {epoch_num}.")
        ckpt_path = os.path.join(checkpoint_dir, candidates[0])

    else:
        raise ValueError(f"Unsupported checkpoint name: {checkpoint_name}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    return ckpt_path
