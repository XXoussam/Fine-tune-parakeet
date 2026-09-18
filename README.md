# Fine-tune-parakeet

Fine-tuning [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NeMo ASR, TDT architecture) on a custom speech dataset.

Training needs a CUDA GPU that this project doesn't assume you have locally, so the workflow is split: **data prep and evaluation run locally**, **training runs wherever there's a GPU** (a Colab notebook or a separate GPU VM/machine).

## Files

| File | Runs where | Purpose |
|---|---|---|
| `nemo.ipynb` | Colab (GPU runtime) | End-to-end notebook: install deps, verify the environment, run the AN4 smoke test, fine-tune parakeet. |
| `finetune_parakeet.py` | GPU machine (Colab or a VM) | Standalone fine-tuning entrypoint (Hydra/NeMo), adapted from NVIDIA's own `examples/asr/speech_to_text_finetune.py`, pinned to `parakeet-tdt-0.6b-v3`. After training it dumps `validation_predictions.jsonl` (raw reference/hypothesis text) for local evaluation. |
| `conf/parakeet_finetune.yaml` | GPU machine | Hyperparameters for `finetune_parakeet.py` (small LR, since this is fine-tuning a converged model, not training from scratch). |
| `prepare_manifest.py` | Local | Builds NeMo manifest(s) (`audio_filepath`/`duration`/`text` jsonl) from a CSV, a directory of `name.wav` + `name.txt` pairs, or this project's `dataset.json` (`clip_path`/`human_labeled`). `--val-split` splits into train/val manifests in one pass. |
| `an4_smoke_test_data.py` | GPU machine | Downloads NVIDIA's AN4 tutorial dataset and builds train/test manifests from it, headless (no Jupyter). Use this for a first pipeline smoke test on a VM before using real data. |
| `evaluate.py` | Local (no GPU/NeMo needed) | Reads `validation_predictions.jsonl`, computes WER itself, writes a CSV plus WER-distribution / WER-vs-duration plots. |
| `try_deberta-v3-base.py` | Local | Unrelated smoke test loading `microsoft/deberta-v3-base` - not part of the fine-tuning pipeline. |

## Workflow

1. **Prepare data locally.** Build train/val manifests with `prepare_manifest.py`, e.g.:
   ```
   python prepare_manifest.py --dataset-json dataset.json --val-split 0.1 \
       --train-out train_manifest.json --val-out val_manifest.json
   ```
2. **Train on a GPU.**
   - Easiest: open `nemo.ipynb` in Colab and run it top to bottom.
   - Or on a GPU VM (see the VM quickstart below).
3. **Copy back one small file.** Training writes `validation_predictions.jsonl` under `nemo_experiments/Parakeet_TDT_Finetuning/.../` — copy just that file to your local machine (e.g. `scp`). Nothing else needs to cross the wire.
4. **Evaluate locally.**
   ```
   python evaluate.py --predictions validation_predictions.jsonl --out-dir eval_report
   ```
   Prints corpus WER and the worst-scoring samples, and writes `per_sample_wer.csv` plus plots to `eval_report/`.

## Setup

Local (data prep + evaluation only): no ML dependencies needed beyond what's in `requirements.txt`'s comments (`soundfile`, `matplotlib` for `evaluate.py`'s plots).

GPU machine (training): see the VM quickstart below.

## VM quickstart (test run, using the AN4 example data)

On a fresh Linux GPU VM:

```bash
# 1. System packages (sph -> wav conversion needs ffmpeg)
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg

# 2. Get the repo
git clone https://github.com/XXoussam/Fine-tune-parakeet.git
cd Fine-tune-parakeet

# 3. Python env
python3 -m venv .venv
source .venv/bin/activate

# 4. PyTorch matching this VM's CUDA version - check first, then install
#    the matching command from https://pytorch.org/get-started/locally/
nvidia-smi   # look at the CUDA version in the top-right
pip install torch --index-url https://download.pytorch.org/whl/cu124   # example only - match your CUDA version

# 5. Everything else (nemo_toolkit, lightning, hydra-core, ...)
pip install -r requirements.txt

# 6. Example data for a first test run (AN4 - same data nemo.ipynb uses)
python an4_smoke_test_data.py --out-dir an4_data

# 7. Fine-tune parakeet-tdt-0.6b-v3 against it
python finetune_parakeet.py \
    model.train_ds.manifest_filepath=an4_data/an4_converted/train_manifest.json \
    model.validation_ds.manifest_filepath=an4_data/an4_converted/test_manifest.json \
    model.train_ds.batch_size=4 \
    model.validation_ds.batch_size=4 \
    trainer.max_epochs=1
```

Step 7 is a **pipeline smoke test**, not real training — `max_epochs=1` and a small batch size just prove everything runs end-to-end on this GPU without OOMing. Once it completes cleanly:

- For a real fine-tune, swap in your actual manifests (`prepare_manifest.py --dataset-json dataset.json --val-split 0.1 ...`) and raise `trainer.max_epochs` (typically 5-20 for a converged 0.6B model, not the ~200 appropriate for training from scratch) and `batch_size` as far as the GPU's VRAM allows.
- Copy `validation_predictions.jsonl` back to your local machine and run `evaluate.py` on it (see Workflow above).
