"""
Batch builder: assembles packed sequences into numbered batches with
deterministic batch hashes for replay/auditing.
"""

import os
import copy
from pipeline import config
from pipeline.utils import sha256_obj, write_json, now_iso


class BatchBuilder:
    """
    Groups packed sequences into fixed-size batches.
    Each batch receives a deterministic hash derived from its content.
    """

    def __init__(self, batch_size: int = config.BATCH_SIZE):
        self._batch_size = batch_size
        self._buffer: list = []          # pending sequences
        self._batches: list = []         # completed batch dicts
        self._global_batch_counter = 0

    # ── public ────────────────────────────────────────────────────────────────

    def add_sequence(self, seq_dict: dict):
        self._buffer.append(seq_dict)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def finalise(self):
        if self._buffer:
            self._flush()

    def get_batches(self) -> list:
        return list(self._batches)

    def reset(self):
        self._buffer  = []
        self._batches = []

    # ── private ───────────────────────────────────────────────────────────────

    def _flush(self):
        seqs       = self._buffer[:self._batch_size]
        self._buffer = self._buffer[self._batch_size:]

        batch_idx  = self._global_batch_counter
        self._global_batch_counter += 1

        sample_ids = self._extract_sample_ids(seqs)
        input_repr = {"sequences": seqs}
        input_hash = sha256_obj(input_repr)

        batch = {
            "batch_id":    f"batch_{batch_idx:06d}",
            "batch_index": batch_idx,
            "sample_ids":  sample_ids,
            "input_hash":  input_hash,
            "seq_count":   len(seqs),
            "token_count": len(seqs) * config.CONTEXT_LENGTH,
            "sequences":   seqs,
            "created_at":  now_iso(),
        }
        self._batches.append(batch)

    @staticmethod
    def _extract_sample_ids(seqs: list) -> list:
        """Collect unique doc_ids from all sequences in the batch."""
        ids = []
        seen = set()
        for seq in seqs:
            for src in seq.get("sources", []):
                did = src.get("doc_id", "")
                if did and did not in seen:
                    ids.append(did)
                    seen.add(did)
        return ids


def build_batches_from_packed(all_packed_results: list) -> list:
    """
    Given a list of pack_shard_documents results, build all batches.
    Returns list of batch dicts.
    """
    builder = BatchBuilder()
    for pack_result in all_packed_results:
        for seq in pack_result["packed_sequences"]:
            builder.add_sequence(seq)
    builder.finalise()
    return builder.get_batches()
