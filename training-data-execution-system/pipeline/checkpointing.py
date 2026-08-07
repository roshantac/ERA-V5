"""
Checkpointing: save and restore complete pipeline state.
"""

import os
import copy
import json
from pipeline import config
from pipeline.utils import sha256_obj, write_json, read_json, log, now_iso

_CHECKPOINT_DIR = config.CHECKPOINTS_DIR


def _checkpoint_path(step: int) -> str:
    return os.path.join(_CHECKPOINT_DIR, f"checkpoint_step_{step:06d}.json")


def save_checkpoint(step: int,
                    batch: dict,
                    consumed_batch_ids: list,
                    scheduler_state: dict,
                    rng_state) -> dict:
    """
    Persist a checkpoint for *step*.

    Parameters
    ----------
    step               : global training step just completed
    batch              : the batch dict processed at this step
    consumed_batch_ids : all batch ids consumed so far (in order)
    scheduler_state    : MixtureScheduler.get_state()
    rng_state          : random.Random.getstate() for replay rng
    """
    os.makedirs(_CHECKPOINT_DIR, exist_ok=True)

    payload = {
        "global_step":       step,
        "batch_id":          batch["batch_id"],
        "consumed_batches":  list(consumed_batch_ids),
        "scheduler_state":   scheduler_state,
        "rng_state":         rng_state,
        "saved_at":          now_iso(),
    }
    # state hash covers everything except saved_at (which changes each run)
    state_hash = sha256_obj({
        "global_step":      payload["global_step"],
        "batch_id":         payload["batch_id"],
        "consumed_batches": payload["consumed_batches"],
    })
    payload["state_hash"] = state_hash

    path = _checkpoint_path(step)
    write_json(path, payload)
    log(f"[PASS] checkpoint_saved  step={step}  hash={state_hash[:16]}…")
    return payload


def load_latest_checkpoint() -> dict | None:
    """
    Return the checkpoint dict with the highest step number, or None.
    """
    if not os.path.isdir(_CHECKPOINT_DIR):
        return None
    files = sorted(
        f for f in os.listdir(_CHECKPOINT_DIR) if f.startswith("checkpoint_step_")
    )
    if not files:
        return None
    path = os.path.join(_CHECKPOINT_DIR, files[-1])
    return read_json(path)


def list_checkpoints() -> list:
    if not os.path.isdir(_CHECKPOINT_DIR):
        return []
    files = sorted(
        f for f in os.listdir(_CHECKPOINT_DIR) if f.startswith("checkpoint_step_")
    )
    result = []
    for f in files:
        result.append(read_json(os.path.join(_CHECKPOINT_DIR, f)))
    return result
