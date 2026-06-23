import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Literal
from tqdm import tqdm
import lightning as pl
import pandas as pd
import numpy as np
import torch
# Add VTC submodule to path so we can import its scripts
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from VTC.scripts.infer import main as vtc_infer
from huggingface_hub import hf_hub_download
from addressee.dataloaders import AddresseeDataloader, id2label
from addressee.models.modeling_hubert import HubertFinetune
from addressee.utils.config import load_config
from addressee.utils.utils import save_timing, get_audio_duration, free_gpu, resolve_device
from glob import glob 

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y.%m.%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")

def process_inference_outputs(predictions, df):
    predictions = np.concatenate(predictions)
    predictions = np.argmax(predictions, axis=1)
    predictions = np.array([id2label[int(v)] for v in predictions])
    df.loc[(df["label"] == "FEM") | (df["label"] == "MAL"), "addressee"] = predictions
    df = df.drop(columns=["binary_classes", "segment_onset", "segment_offset", "duration(s)", "file_path", "recording_duration"])

    return df

# from https://github.com/MarvinLvn/BabAR/blob/main/src/pipeline.py
def run_vtc(uris,
            wavs,
            output,
            vtc_batch_size,
            high_precision,
            wav_files,
            wavs_needing_vtc,
            device,
            timing_path,
            recursive_search):
    logger.info(
        f"Step 1/2: Running VTC on {len(wavs_needing_vtc)}/{len(wav_files)} file(s) "
        f"({len(wav_files) - len(wavs_needing_vtc)} already have RTTM)..."
    )

    vtc_start = time.time()
    logger.info(f"Using {'high precision' if high_precision else 'F1'} thresholds.")
    vtc_infer(
        uris=uris,
        wavs=str(wavs),
        output=str(output),
        config=str(REPO_ROOT / "VTC" / "VTC-2" / "model" / "config.toml"),
        checkpoint=str(REPO_ROOT / "VTC" / "VTC-2" / "model" / "best.ckpt"),
        batch_size=vtc_batch_size,
        thresholds=REPO_ROOT / "VTC" / "thresholds" / ("hp.toml" if high_precision else "f1.toml"),
        device=device,
        recursive_search=recursive_search
    )
    vtc_total_sec = time.time() - vtc_start

    # VTC processes all files as a batch, so we distribute time
    # proportionally to each file's audio duration.
    vtc_durations = {
        w.stem: get_audio_duration(w) for w in wavs_needing_vtc
    }
    total_audio_dur = sum(vtc_durations.values())

    vtc_timing = []
    for w in wavs_needing_vtc:
        audio_dur = vtc_durations[w.stem]
        vtc_file_sec = (
            vtc_total_sec * audio_dur / total_audio_dur
            if total_audio_dur > 0
            else 0.0
        )
        vtc_timing.append({
            "filename": w.name,
            "audio_duration_sec": round(audio_dur, 2),
            "vtc_sec": round(vtc_file_sec, 2),
        })

    save_timing(vtc_timing, timing_path)
    logger.info(f"VTC total time: {vtc_total_sec:.1f}s (per-file estimates saved to {timing_path})")


    


def main(
    output: str,
    uris: Path | None = None,
    VTC2_output: Path | None = None,
    VTC2_batch_size: int= 128,
    max_utt_dur: float = 30.0,
    wavs: str = "data/debug/wav",
    batch_size: int = 16,
    n_workers: int = 4,
    VTC2_high_precision: bool = False,
    device: Literal["gpu", "cuda", "cpu", "mps"] = "gpu",
    model_dir: str | None = None,
    model_revision: str | None = None,
    recursive_search: bool = False,
    verbose=True,
):
    
    model_path = hf_hub_download(repo_id="coml/addressee", filename="best.ckpt", local_dir=model_dir, revision = model_revision)
    config_path = hf_hub_download(repo_id="coml/addressee", filename="config.yaml", local_dir=model_dir, revision = model_revision)


    device = resolve_device(device)
    output = Path(output)
    wavs = Path(wavs)
    output.mkdir(parents=True, exist_ok=True)
    rttm_dir = output / "rttm"
    timing_path = output / "timing.csv"


    if recursive_search == True:
        wav_files = list(wavs.rglob("*.wav"))
    else:
        wav_files = list(wavs.glob("*.wav"))
    wav_files = sorted(wav_files)
    if not wav_files:
        logger.error(f"No .wav files found in {wavs}")
        return
    
    
    logger.info(f"Found {len(wav_files)} audio file(s). Device: {device}")
    # -------------------
    # VTC2 inference
    # -------------------

    # -- Step 1: VTC on all files ------------------------------------------
    if uris is not None:
        uri_set = set(pd.read_csv(uris, low_memory=False, header=None, names=["uris"])["uris"])
        
        wavs_needing_vtc = [
            w for w in wav_files
            if (w.stem in uri_set and not (rttm_dir / f"{w.stem}.rttm").exists())
        ]
        logger.info(f"Found {len(wavs_needing_vtc)} wavs to run on from {len(uri_set)} uris.")
    else:
        wavs_needing_vtc = [
            w for w in wav_files
            if not (rttm_dir / f"{w.stem}.rttm").exists()
        ]
        logger.info(f"Found {len(wavs_needing_vtc)} wavs to run on.")
    
    if wavs_needing_vtc:
        run_vtc(uris,
                wavs,
                output,
                VTC2_batch_size,
                VTC2_high_precision,
                wav_files,
                wavs_needing_vtc,
                device,
                timing_path,
                recursive_search=recursive_search)
    else:
        if uris:
            logger.info(f"Step 1/2: All {len(wavs_needing_vtc)} wavs from {len(wav_files)} file(s) or {len(uri_set)} file(s) from uris subset already have RTTM, skipping VTC.")
        else:
            logger.info(f"Step 1/2: All {len(wavs_needing_vtc)} wavs from {len(wav_files)} file(s) already have RTTM, skipping VTC.")


    # Collect non-empty RTTMs
    rttm_files = sorted(
        f for f in rttm_dir.glob("*.rttm")
        if f.stat().st_size > 0
    )

    if not rttm_files:
        logger.warning("No RTTM files with speech found. Nothing to transcribe.")
        return

    logger.info(f"VTC done. {len(rttm_files)} file(s) with speech.")

    # Free VTC model memory before loading BabAR
    free_gpu()

    # -------------------
    # Addressee inference
    # -------------------
    
    config = load_config(train_config=config_path)
    config.train.batch_size = batch_size
    config.model_id = model_path
    
    model = HubertFinetune(config)
    model.eval()
    
    #filter abnormal audios ?
    if VTC2_output is not None: 
        df_VTC2 = pd.read_csv(VTC2_output, low_memory=False)
    else:
        df_VTC2 = pd.read_csv(output / "rttm.csv", low_memory=False)



    df_VTC2["recording_duration"] = -1.0
    df_VTC2["file_path"] = ""
    for uid in tqdm(df_VTC2["uid"].unique()):
        wav_file = wavs / (uid + ".wav")
        df_VTC2.loc[df_VTC2["uid"] == uid, "recording_duration"] = get_audio_duration(wav_file)
        df_VTC2.loc[df_VTC2["uid"] == uid, "file_path"] = str(wav_file)
    
    #segment_onset in milliseconds
    df_VTC2["segment_onset"] = df_VTC2["start_time_s"] * 1000
    df_VTC2["segment_offset"] = df_VTC2["segment_onset"] + (df_VTC2["duration_s"] * 1000)
    df_VTC2["duration(s)"] =df_VTC2["duration_s"]
    df_VTC2["addressee"] = pd.NA
    df_VTC2["binary_classes"] = "KCDS"

    df_VTC2.loc[(df_VTC2["duration(s)"] > max_utt_dur) & ((df_VTC2["label"] == "FEM") | (df_VTC2["label"] == "MAL")), "addressee"] = "Other"

    df_inference = df_VTC2.loc[(df_VTC2["duration(s)"] <= max_utt_dur) & ((df_VTC2["label"] == "FEM") | (df_VTC2["label"] == "MAL"))].copy()

    dm = AddresseeDataloader(dataset = "addressee",
                             dataset_path= config.data.dataset_path,
                             inference=True,
                             optional_df=df_inference,
                             config= config,
                             n_workers=n_workers)


    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        logger=False
    )

    predictions = trainer.predict(model, datamodule=dm, ckpt_path=model_path)

    df_VTC2.loc[(df_VTC2["duration(s)"] <= max_utt_dur) & ((df_VTC2["label"] == "FEM") | (df_VTC2["label"] == "MAL"))] = process_inference_outputs(predictions, df_inference)

    df_VTC2.to_csv(output / "rttm_addressee.csv", columns=["uid", "start_time_s", "duration_s", "label", "addressee"], index=False)
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uris", help="Path to a file containing the list of uris to use."
    )
    parser.add_argument(
        "--VTC2_output", help="Path to VTC2 output, rttm.csv",
        default=None,
    )
    parser.add_argument(
        "--wavs",
        default="data/debug/wav",
        help="Folder containing the audio files to run inference on.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Path to the folder that will contain the final predictions.",
    )
    parser.add_argument(
        "--max_utt_dur", help="Maximum utterance duration to classify, higher will be classified as Other",
        default=30,
    )

    parser.add_argument(
        "--model_dir",
        default=None,
        help="Model directory path, defaults to huggingface cache dir.",
    )
    parser.add_argument(
        "--model_revision",
        default=None,
        help="Model revision to select specific model checkpoint on huggingface.",
    )
    parser.add_argument(
        "--VTC2_batch_size",
        default=128,
        type=int,
        help="VTC2 batch size to use for the forward pass of the model.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Addressee batch size to use for the forward pass of the model.",
    )
    parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="Addressee number of workers to speedup the dataloading. Increases cpu memory consumption",
    )
    parser.add_argument(
        "--VTC2_high_precision",
        action="store_true",
        help="Use VTC2 high_precision thresholds",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["gpu", "cuda", "cpu", "mps"],
        help="Size of the batch used for the forward pass in the model.",
    )
    parser.add_argument(
        "--recursive_search",
        action="store_true",
        help="Recursively search for `.wav` files. Might be slow. Defaults to False.",
    )
    torch.set_float32_matmul_precision('high')
    args = parser.parse_args()
    main(**vars(args))