import gc
from pathlib import Path
import pandas as pd
import torch
from torchcodec.decoders import AudioDecoder

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")



def save_timing(timing_records: list[dict], output_path: Path):
    """Save timing records to CSV, merging with any existing data."""
    new_df = pd.DataFrame(timing_records)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_df = existing_df.set_index("filename")
        new_df = new_df.set_index("filename")
        combined = existing_df.combine_first(new_df)
        combined = combined.reset_index().sort_values("filename")
    else:
        combined = new_df.sort_values("filename")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

def get_audio_duration(wav_path: Path) -> float:
    """Return audio duration in seconds."""
    decoder = AudioDecoder(wav_path)
    duration = decoder.metadata.duration_seconds
    return duration


def free_gpu():
    """Force garbage collection and free GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_device(device: str) -> str:
    """Normalize device string and check availability."""
    if device in ("gpu", "cuda"):
        if torch.cuda.is_available():
            return "cuda"
        logger.warning("CUDA requested but not available, falling back to CPU.")
        return "cpu"
    if device == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        logger.warning("MPS requested but not available, falling back to CPU.")
        return "cpu"
    return "cpu"
