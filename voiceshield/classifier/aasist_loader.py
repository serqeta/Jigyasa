import os

import torch

# Exact config from clovaai/aasist config/AASIST-L.conf
_AASIST_L_ARGS = {
    "nb_samp": 64600,
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
    "gat_dims": [24, 32],
    "pool_ratios": [0.4, 0.5, 0.7, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}


def load_aasist(path: str) -> "torch.nn.Module":
    """Load AASIST-L checkpoint from path. Raises FileNotFoundError if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AASIST weights not found at {path}. Run scripts/download_model.py"
        )

    from voiceshield.classifier.aasist_model import AASIST

    model = AASIST(_AASIST_L_ARGS)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model
