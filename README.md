# Addressee Classifier

The Addressee Classifier is a classification model that, given a human or automatically segmented adult speech audio, outputs a classification of child-directed speech.

The four classes that the model will output are:
- **KCDS** stands for key-child directed speech
- **ADS** stands for adult directed speech
- **Other** stands for other/overheard speech

The model has been specifically trained to work with child-centered long-form recordings. These are recordings that can span multiple hours and have been collected using a portable recorder attached to the vest of a child (usually 0 to 5 years of age).

## Table of contents

1. [Installation](#1-installation)
2. [Input data requirements](#2-input-data-requirements)
3. [Inference](#3-inference)
4. [CLI reference](#4-cli-reference)
5. [Output format](#5-output-format)
6. [Model architecture and design](#6-model-architecture-and-design)
7. [Model performance](#7-model-performance)
8. [Troubleshooting](#8-troubleshooting)
9. [Citation](#9-citation)
10. [Acknowledgement](#10-acknowledgement)

---

## 1. Installation

To use the model, you will need a unix-based machine (Linux or macOS) and Python 3.13 or higher. Windows is not currently supported.

As system dependencies, ensure that [uv](https://docs.astral.sh/uv/), [ffmpeg](https://ffmpeg.org/), and [git-lfs](https://git-lfs.com/) are installed. You can check this by running:

```bash
./check_sys_dependencies.sh
```

Clone the repository with:

```bash
git lfs install
git clone --recurse-submodules https://github.com/LAAC-LSCP/addressee.git
cd addressee
```

Install Python dependencies:

```bash
uv sync
```

---

## 2. Input data requirements

### Audio format

All audio files must be:
- Format: `.wav`
- Sample rate: 16 000 Hz (16 kHz)
- Channels: mono (single channel)

If your files do not meet these requirements, use the provided conversion script:

```bash
uv run scripts/convert.py \
    --wavs /path/to/raw_audio \
    --output /path/to/converted_audio
```

### Utterance duration

The pipeline classifies utterances of up to `max_utt_dur` seconds, set by default to 30s. Utterances from `FEM` or `MAL` speakers longer than `max_utt_dur` seconds are automatically assigned the label `Other` without being passed through the model.

---

## 3. Inference

The pipeline runs by default in two sequential steps:

**Step 1 — Voice Type Classifier (VTC):** The audio is processed by VTC 2, which segments the recording and assigns a broad speaker type to each segment: `KCHI` (key child), `OCH` (other child), `MAL` (male adult), `FEM` (female adult), or silence. Output is saved per audio as `.rttm` files and regrouped in the `rttm.csv` file.

**Step 2 — Addressee classification:** Segments labelled `FEM` or `MAL` are passed to the addressee model (HuBERT-based), which classifies each segment as `KCDS`, `ADS`, or `Other`. Non-adult segments are left unclassified.

```bash
uv run scripts/infer.py \
    --wavs audios \        # path to folder containing .wav files
    --output predictions \ # output folder
    --device cuda          # device: cpu, cuda/gpu, or mps
```

### Skipping VTC (using existing RTTM output)

If you have already run VTC separately (e.g., via the [VTC repository](https://github.com/LAAC-LSCP/VTC)), you can skip step 1 by providing the VTC CSV output directly:

```bash
uv run scripts/infer.py \
    --wavs audios \
    --output predictions \
    --VTC2_output /path/to/rttm.csv 
```

### Processing a subset of files

To run only on a specific list of recordings (e.g., a subset of a large corpus), provide a plain-text file listing the recording IDs (one per line, without the `.wav` extension) using `--uris`:

```bash
uv run scripts/infer.py \
    --wavs audios \
    --output predictions \
    --uris my_subset.txt 
```

### Helper script

An example bash script is available at `scripts/run.sh`. Set the variables at the top of the file and run:

```bash
sh scripts/run.sh
```

---

## 4. Output format

The pipeline writes the following files to `<output>/`:

```txt
<output>/
│
├── 📂 rttm/                  # One .rttm file per recording (VTC output, with segment merging)
├── 📄 rttm.csv               # CSV version of all VTC segments across all recordings
├── 📄 rttm_addressee.csv     # Final output: VTC segments + addressee classification column
└── 📄 timing.csv             # Per-file VTC processing time estimates
```

### `rttm_addressee.csv` column reference

This is the main output file. It extends VTC2 `rttm.csv` with the `addressee` column.

| Column | Description |
|---|---|
| `uid` | Recording identifier (filename without `.wav` extension). |
| `label` | Speaker type assigned by VTC: `KCHI`, `OCH`, `MAL`, `FEM`. |
| `start_time_s` | Utterance onset in seconds from the beginning of the recording. |
| `duration_s` | Utterance duration in seconds. |
| `addressee` | Addressee classification for adult (`MAL`, `FEM`) segments: `KCDS`, `ADS`, or `Other`. Empty (`NaN`) for non-adult segments. Segments longer than 30 s are assigned `Other` without model inference. |

---

## 5. CLI reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--wavs` | `str` | `data/debug/wav` | Path to the folder containing `.wav` audio files. |
| `--output` | `str` | *(required)* | Path to the output folder. Created if it does not exist. |
| `--device` | `str` | `cuda` | Device to run inference on. One of `cpu`, `cuda`, `gpu`, or `mps`. |
| `--VTC2_output` | `str` | `None` | Path to a pre-existing VTC output CSV (`rttm.csv`). If provided, the VTC step is skipped entirely. |
| `--uris` | `str` | `None` | Path to a plain-text file listing recording IDs (one per line) to process. Used to run on a subset of the files in `--wavs`. If omitted, all `.wav` files in `--wavs` are processed. |
| `--vtc_batch_size` | `int` | `128` | Batch size for the VTC model. Larger values are faster but require more GPU memory. |
| `--batch_size` | `int` | `128` | Batch size for the addressee model. |
| `--context_size` | `float` | `10.0` | Context window in seconds provided to the model around each utterance. |
| `--save_logits` | `flag` | `False` | If set, raw model logits are saved alongside predictions. Memory-intensive. |
| `--high_precision` | `flag` | `False` | Use high-precision VTC thresholds instead of F1-optimised thresholds. |
| `--write_empty` | `bool` | `True` | Whether to write output files for recordings that contain no detected speech. |
| `--write_csv` | `bool` | `True` | Whether to write the CSV version of the RTTM output. |
| `--model_dir` | `str` | `None` | Local directory to cache the downloaded model checkpoint and config. Uses the Hugging Face default cache if omitted. |
| `--model_revision` | `str` | `None` | Specific revision (branch, tag, or commit hash) of the `coml/addressee` HuggingFace model to use. Uses the latest version if omitted. |

---

## 6. Model architecture and design

This section describes the technical design of the pipeline for researchers and practitioners who want to understand or extend the system.

### Pipeline overview

```
.wav files
    │
    ▼
[Step 1] VTC 2.1 — Voice Type Classifier
    Segments audio into speaker-typed intervals (KCHI, OCH, MAL, FEM)
    Output: per-file .rttm + rttm.csv
    │
    ▼ (FEM and MAL segments ≤ 30 s only)
[Step 2] Addressee model — BabyHuBERT-based classifier
    Classifies each adult segment as KCDS, ADS, or Other
    Output: rttm_addressee.csv
```

### Addressee model

The addressee model is a fine-tuned [BabyHuBERT](https://arxiv.org/abs/2509.15001) model trained to classify adult speech segments into three classes. BabyHuBERT (Baby Hidden-Unit BERT) is a self-supervised speech representation model that learns discrete speech units from raw audio. The fine-tuned model adds a classification head on top of the BabyHuBERT encoder.

The model is loaded from the Hugging Face Hub at `coml/addressee` and requires both a checkpoint (`best.ckpt`) and a training config (`config.yaml`), which are downloaded automatically on first run.

### Context window

The `--context_size` argument (default 10 s) controls how much audio context is provided around each utterance to the model. A larger context window gives the model more surrounding speech to make its decision but increases memory usage. **The model has been trained with a context size of 10s**.

### Utterance length handling

The pipeline applies a hard 30-second cap before model inference: any adult utterance longer than 30 s is directly assigned `Other` without being passed to the model. This avoids GPU memory issues and reflects the training distribution as segment longer than 30s are abnormal. This can be bypassed by setting `--max_utt_dur`


## 7. Model performance

### 7.1 Runtime
### 7.2 Model Performance on the heldout set
### 7.3 Model Performance using VTC2 segments 

## 8. Troubleshooting

**`No .wav files found in <path>`**  
Check that `--wavs` points to a folder containing `.wav` files (not a single file). Ensure files use the `.wav` extension (lowercase).

**`No RTTM files with speech found`**  
VTC did not detect any speech in the provided recordings. This can happen with very short clips, very noisy recordings, or if the audio is not mono 16 kHz. Run `scripts/convert.py` to verify your audio format.

**Out-of-memory errors on GPU**  
Reduce `--batch_size` and/or `--vtc_batch_size`. If the problem persists, try `--device cpu`.

**VTC is re-running on files that already have RTTM output**  
The pipeline skips VTC for files whose `.rttm` already exists in `<output>/rttm/`. If the RTTM file is present but empty (0 bytes), it is treated as containing no speech and VTC is not re-run. Delete corrupted or zero-byte RTTM files and re-run if needed.

**Pinning to a specific model version for reproducibility**  
Use `--model_revision <git_hash_or_tag>` to fix the model to a known HuggingFace Hub revision. You can find available revisions on the [coml/addressee](https://huggingface.co/coml/addressee) model page.

**`addressee` column is empty for all rows**  
This column is only populated for `FEM` and `MAL` segments. Rows with `KCHI` or `OCH` labels are intentionally left unclassified. If all your `FEM`/`MAL` rows are also empty, check that VTC produced non-empty RTTM files.

**macOS / Apple Silicon (`mps`)**  
Use `--device mps`. Note that MPS support in PyTorch Lightning may require a recent version of macOS (13+) and PyTorch (2.0+). If you encounter errors, fall back to `--device cpu`.

---

## 9. Citation

The training code for the addressee model can be found at [labicquette/addressee](https://github.com/labicquette/BabyHuBERT).

To cite this work, please use the following BibTeX entry:

```bibtex
@misc{charlot2026addressee
}
```

To cite VTC2, please use the following BibTeX entry: 

```bibtex
@misc{charlot2025babyhubertmultilingualselfsupervisedlearning,
    title={BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings}, 
    author={Théo Charlot and Tarek Kunze and Maxime Poli and Alejandrina Cristia and Emmanuel Dupoux and Marvin Lavechin},
    year={2025},
    eprint={2509.15001},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url={https://arxiv.org/abs/2509.15001}, 
}
```

---

## 10. Acknowledgement

The Voice Type Classifier has benefited from numerous contributions over time. The following publications document its evolution, listed in reverse chronological order.

This README was adapted from the [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC) repository.

This work was performed using HPC resources from GENCI-IDRIS (2025-AD011016414) and was developed as part of the ExELang project funded by the European Union (ERC, ExELang, Grant No 101001095).