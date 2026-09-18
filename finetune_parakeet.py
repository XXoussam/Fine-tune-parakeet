#!/usr/bin/env python3
"""Fine-tune NVIDIA Parakeet (parakeet-tdt-0.6b-v3) with NeMo.

This is meant to run on a Linux GPU VM (CUDA), not on this local machine -
NeMo's ASR training stack needs a CUDA-capable PyTorch install. See
requirements-finetune.txt for the packages to install on the VM first.

Adapted from NVIDIA's own examples/asr/speech_to_text_finetune.py in the
NVIDIA-NeMo/NeMo repo, pinned to the parakeet-tdt-0.6b-v3 checkpoint and
with defaults from conf/parakeet_finetune.yaml tuned for fine-tuning
(small LR, single GPU) rather than training from scratch.

1. Prepare NeMo manifests (jsonl with audio_filepath/duration/text) for your
   train/validation data. Use prepare_manifest.py if you don't have these yet.
2. Run:
   python finetune_parakeet.py \
       model.train_ds.manifest_filepath=/data/train_manifest.json \
       model.validation_ds.manifest_filepath=/data/val_manifest.json
3. Override any other config value the same way, e.g.:
       trainer.devices=2 model.train_ds.batch_size=8 trainer.max_epochs=20
4. The fine-tuned checkpoint (.nemo) is written under
   ./nemo_experiments/Parakeet_TDT_Finetuning/.../checkpoints/
"""

from __future__ import annotations

import time
from typing import Union

import lightning.pytorch as pl
from omegaconf import DictConfig, OmegaConf

from nemo.collections.asr.models import ASRModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.get_rank import is_global_rank_zero
from nemo.utils.trainer_utils import resolve_trainer_cfg


def get_base_model(trainer: pl.Trainer, cfg: DictConfig) -> ASRModel:
    """Load the checkpoint to fine-tune, from a pretrained name or a local .nemo file."""
    nemo_model_path = cfg.get("init_from_nemo_model", None)
    pretrained_name = cfg.get("init_from_pretrained_model", None)

    if nemo_model_path is not None and pretrained_name is not None:
        raise ValueError("Set only one of `init_from_nemo_model` or `init_from_pretrained_model`, not both")
    if nemo_model_path is None and pretrained_name is None:
        raise ValueError("Set one of `init_from_nemo_model` or `init_from_pretrained_model`")

    if nemo_model_path is not None:
        asr_model = ASRModel.restore_from(restore_path=nemo_model_path)
    else:
        # Only rank 0 downloads the checkpoint; other ranks wait so they don't race.
        num_ranks = trainer.num_devices * trainer.num_nodes
        if num_ranks > 1 and not is_global_rank_zero():
            wait_time = max(int(cfg.get("exp_manager", {}).get("seconds_to_sleep", 60)), 60)
            logging.info(f"Sleeping {wait_time}s so rank 0 can download the checkpoint first.")
            time.sleep(wait_time)
        asr_model = ASRModel.from_pretrained(model_name=pretrained_name)

    asr_model.set_trainer(trainer)
    return asr_model


def check_vocabulary(asr_model: ASRModel, cfg: DictConfig) -> ASRModel:
    """Reuse the pretrained tokenizer, or swap it if the config asks for that."""
    if cfg.model.tokenizer.update_tokenizer:
        if cfg.model.char_labels.update_labels:
            raise ValueError("Set only one of `model.tokenizer.update_tokenizer` or `model.char_labels.update_labels`")
        asr_model = update_tokenizer(asr_model, cfg.model.tokenizer.dir, cfg.model.tokenizer.type)
    elif cfg.model.char_labels.update_labels:
        asr_model.change_vocabulary(new_vocabulary=cfg.model.char_labels.labels)
        logging.warning("Vocabulary updated with the provided char labels.")
    else:
        logging.info("Reusing the pretrained tokenizer/vocabulary.")
    return asr_model


def update_tokenizer(asr_model: ASRModel, tokenizer_dir: Union[str, DictConfig], tokenizer_type: str) -> ASRModel:
    """Swap the tokenizer, reinitializing the decoder/joint only if vocab size changed."""
    if tokenizer_dir is None:
        raise ValueError("model.tokenizer.dir must be set if update_tokenizer is true")

    vocab_size = asr_model.tokenizer.vocab_size
    decoder_state = asr_model.decoder.state_dict()
    joint_state = asr_model.joint.state_dict() if hasattr(asr_model, "joint") else None

    asr_model.change_vocabulary(new_tokenizer_dir=tokenizer_dir, new_tokenizer_type=tokenizer_type)

    if asr_model.tokenizer.vocab_size != vocab_size:
        logging.warning("New tokenizer has a different vocab size; decoder/joint were reinitialized.")
    else:
        asr_model.decoder.load_state_dict(decoder_state)
        if joint_state is not None:
            asr_model.joint.load_state_dict(joint_state)
    return asr_model


def setup_dataloaders(asr_model: ASRModel, cfg: DictConfig) -> ASRModel:
    cfg = model_utils.convert_model_config_to_dict_config(cfg)
    asr_model.setup_training_data(cfg.model.train_ds)
    asr_model.setup_multiple_validation_data(cfg.model.validation_ds)
    if cfg.model.get("test_ds", {}).get("manifest_filepath") is not None:
        asr_model.setup_multiple_test_data(cfg.model.test_ds)
    return asr_model


@hydra_runner(config_path="conf", config_name="parakeet_finetune")
def main(cfg: DictConfig) -> None:
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_manager(trainer, cfg.get("exp_manager", None))

    asr_model = get_base_model(trainer, cfg)
    asr_model = check_vocabulary(asr_model, cfg)
    asr_model = setup_dataloaders(asr_model, cfg)

    asr_model.setup_optimization(cfg.model.optim)

    if cfg.model.get("spec_augment") is not None:
        asr_model.spec_augment = ASRModel.from_config_dict(cfg.model.spec_augment)

    trainer.fit(asr_model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
