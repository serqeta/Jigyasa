import os

import torch


def load_aasist(path: str) -> torch.nn.Module:
    """Load AASIST-L checkpoint from path. Raises FileNotFoundError if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AASIST weights not found at {path}. Run scripts/download_model.py"
        )

    from voiceshield.classifier.aasist_model import AASIST

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # The checkpoint may be a raw state_dict or wrapped in a dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model = AASIST()
    model.load_state_dict(state_dict)
    model.eval()
    return model
