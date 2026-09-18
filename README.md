# Fine-tune-parakeet

Fine-tuning [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (NeMo ASR, TDT architecture) on a custom speech dataset.

Training needs a CUDA GPU that this project doesn't assume you have locally, so the workflow is split: **data prep and evaluation run locally**, **training runs wherever there's a GPU** (a Colab notebook or a separate GPU VM/machine).

## Files

| File | Runs where | Purpose |
|---|---|---|
| `nemo.ipynb` | Colab (GPU runtime) | End-to-end notebook: install deps, verify the environment, run the AN4 smoke test, fine-tune parakeet. |
| `finetune_parakeet.py` | GPU machine (Colab or a VM) | Standalone fine-tuning entrypoint (Hydra/NeMo), adapted from NVIDIA's own `examples/asr/speech_to_text_finetune.py`, pinned to `parakeet-tdt-0.6b-v3`. After training it dumps `validation_predictions.jsonl` (raw reference/hypothesis text) for local evaluation. |
| `conf/parakeet_finetune.yaml` | GPU machine | Hyperparameters for `finetune_parakeet.py` (small LR, since this is fine-tuning a converged model, not training from scratch). |
| `prepare_manifest.py` | Local | Builds a NeMo manifest (`audio_filepath`/`duration`/`text` jsonl) from a CSV or a directory of `name.wav` + `name.txt` pairs. |
| `build_manifest.py` | Local | Same idea, wired to this project's actual labeled dataset format (`dataset.json` with `clip_path`/`human_labeled`). |
| `evaluate.py` | Local (no GPU/NeMo needed) | Reads `validation_predictions.jsonl`, computes WER itself, writes a CSV plus WER-distribution / WER-vs-duration plots. |
| `try_parakeet.py` | Local | Quick smoke test transcribing a sample WAV via the Hugging Face `transformers` pipeline. |
| `try_deberta-v3-base.py` | Local | Unrelated smoke test loading `microsoft/deberta-v3-base`. |

## Workflow

1. **Prepare data locally.** Build a NeMo manifest with `prepare_manifest.py` (generic CSV/wav+txt input) or `build_manifest.py` (this project's `dataset.json`).
2. **Train on a GPU.**
   - Easiest: open `nemo.ipynb` in Colab and run it top to bottom.
   - Or on a GPU VM: `pip install -r requirements.txt`, then
     ```
     python finetune_parakeet.py \
         model.train_ds.manifest_filepath=/path/train_manifest.json \
         model.validation_ds.manifest_filepath=/path/val_manifest.json
     ```
     Override any other config value the same way, e.g. `trainer.max_epochs=10 model.train_ds.batch_size=8`.
3. **Copy back one small file.** Training writes `validation_predictions.jsonl` under `nemo_experiments/Parakeet_TDT_Finetuning/.../` — copy just that file to your local machine (e.g. `scp`). Nothing else needs to cross the wire.
4. **Evaluate locally.**
   ```
   python evaluate.py --predictions validation_predictions.jsonl --out-dir eval_report
   ```
   Prints corpus WER and the worst-scoring samples, and writes `per_sample_wer.csv` plus plots to `eval_report/`.

## Setup

Local (data prep + evaluation only): no ML dependencies needed beyond what's in `requirements.txt`'s comments (`soundfile`, `matplotlib` for `evaluate.py`'s plots).

GPU machine (training):
```
pip install -r requirements.txt
```
Install PyTorch matching the machine's CUDA version *first* — `nemo_toolkit` doesn't pin one for you. See [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
