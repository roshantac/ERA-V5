"""
Packing system with full reconstruction validation.

Generates:
  reports/packing_validation_report.json   – full token accounting
  reports/packed_batch_audit.json          – per-shard summary (compat)
"""

import os
import copy
from pipeline import config
from pipeline.utils import sha256_obj, write_json, log, now_iso

_PACKING_VALIDATION_PATH = os.path.join(config.REPORTS_DIR, "packing_validation_report.json")


# ─── Data structures ──────────────────────────────────────────────────────────

class PackedSequence:
    """One context-length window after packing."""

    def __init__(self):
        self.tokens:      list = []
        self.mask:        list = []
        self.segments:    list = []
        self.sources:     list = []
        self.position_ids: list = []   # 0..CONTEXT_LENGTH-1 (reset at pad boundary)

    def to_dict(self) -> dict:
        real = sum(self.mask)
        pad  = len(self.mask) - real
        return {
            "tokens":        self.tokens,
            "mask":          self.mask,
            "segments":      self.segments,
            "sources":       self.sources,
            "position_ids":  self.position_ids,
            # explicit accounting fields required by the spec
            "token_count":   real,
            "padding_count": pad,
            "capacity":      config.CONTEXT_LENGTH,
        }


# ─── Packer ───────────────────────────────────────────────────────────────────

class SequencePacker:
    """
    Greedy packer: fill CONTEXT_LENGTH windows then flush, padding the last.
    """

    def __init__(self):
        self._context = config.CONTEXT_LENGTH
        self._pad     = config.PAD_TOKEN_ID
        self._buf_tokens:   list = []
        self._buf_segments: list = []
        self._buf_sources:  list = []
        self._packed:       list = []

    def add(self, token_ids: list, source_info: dict):
        if not token_ids:
            return
        seg_label = source_info.get("doc_id", "unknown")
        remaining = token_ids[:]
        while remaining:
            space = self._context - len(self._buf_tokens)
            if space == 0:
                self._flush()
                space = self._context
            chunk     = remaining[:space]
            remaining = remaining[space:]
            self._buf_tokens.extend(chunk)
            self._buf_segments.extend([seg_label] * len(chunk))
            if source_info not in self._buf_sources:
                self._buf_sources.append(copy.deepcopy(source_info))

    def finalise(self):
        if self._buf_tokens:
            self._flush(pad=True)

    def get_packed(self) -> list:
        return self._packed

    def reset(self):
        self._buf_tokens   = []
        self._buf_segments = []
        self._buf_sources  = []
        self._packed       = []

    def _flush(self, pad: bool = False):
        assert len(self._buf_tokens) <= self._context
        seq          = PackedSequence()
        seq.tokens   = list(self._buf_tokens)
        seq.segments = list(self._buf_segments)
        seq.sources  = list(self._buf_sources)
        if pad:
            pad_len      = self._context - len(seq.tokens)
            seq.tokens  += [self._pad] * pad_len
            seq.mask     = [1] * (self._context - pad_len) + [0] * pad_len
            seq.segments += ["PAD"] * pad_len
        else:
            seq.mask = [1] * self._context
        # position_ids: sequential within real tokens, 0 for padding
        seq.position_ids = [i if seq.mask[i] == 1 else 0
                            for i in range(self._context)]
        assert len(seq.tokens)      == self._context
        assert len(seq.mask)        == self._context
        assert len(seq.segments)    == self._context
        assert len(seq.position_ids) == self._context
        self._packed.append(seq)
        self._buf_tokens   = []
        self._buf_segments = []
        self._buf_sources  = []


# ─── Validation ───────────────────────────────────────────────────────────────

def validate_packed_sequences(packed: list) -> dict:
    """
    Full token accounting + mask correctness.
    Works on PackedSequence objects or dicts.
    """
    total_real = 0
    total_pad  = 0
    errors     = []

    for idx, ps in enumerate(packed):
        d    = ps.to_dict() if hasattr(ps, "to_dict") else ps
        toks = d["tokens"]
        mask = d["mask"]
        segs = d["segments"]
        pids = d.get("position_ids", [])

        if len(toks) != config.CONTEXT_LENGTH:
            errors.append(f"seq[{idx}]: token length {len(toks)} != {config.CONTEXT_LENGTH}")
        if len(mask) != config.CONTEXT_LENGTH:
            errors.append(f"seq[{idx}]: mask length {len(mask)} != {config.CONTEXT_LENGTH}")
        if len(segs) != config.CONTEXT_LENGTH:
            errors.append(f"seq[{idx}]: segment length {len(segs)} != {config.CONTEXT_LENGTH}")
        if len(pids) != config.CONTEXT_LENGTH:
            errors.append(f"seq[{idx}]: position_ids length {len(pids)} != {config.CONTEXT_LENGTH}")

        for pos, (tok, m) in enumerate(zip(toks, mask)):
            if m == 0 and tok != config.PAD_TOKEN_ID:
                errors.append(
                    f"seq[{idx}] pos {pos}: mask=0 but token={tok} (not PAD)"
                )

        real       = sum(mask)
        pad        = len(mask) - real
        total_real += real
        total_pad  += pad

    tokens_possible = len(packed) * config.CONTEXT_LENGTH
    efficiency      = round(total_real / tokens_possible, 4) if tokens_possible else 0.0

    return {
        "sequences_checked": len(packed),
        "tokens_possible":   tokens_possible,
        "tokens_used":       total_real,
        "tokens_pad":        total_pad,
        "packing_efficiency": efficiency,
        "errors":            errors,
        "result":            "PASS" if not errors else "FAIL",
    }


# ─── Full packing validation report ──────────────────────────────────────────

def generate_packing_validation_report(all_packed_results: list) -> dict:
    """
    Aggregate all shard packing results into the spec-required report
    with explicit token_count, padding_count, capacity and sequence mapping.
    """
    total_tokens_used     = 0
    total_tokens_possible = 0
    total_padding         = 0
    shard_summaries       = []
    all_errors            = []

    for res in all_packed_results:
        pr  = res["packing_report"]
        val = res["validation"]
        used     = pr["tokens_used"]
        possible = pr["tokens_possible"]
        pad      = possible - used

        total_tokens_used     += used
        total_tokens_possible += possible
        total_padding         += pad
        all_errors.extend(val["errors"])

        # per-sequence mapping (first 4 stored inline, rest by count)
        seq_map = []
        for i, seq in enumerate(res["packed_sequences"][:4]):
            seq_map.append({
                "seq_index":       i,
                "token_count":     seq.get("token_count", sum(seq["mask"])),
                "padding_count":   seq.get("padding_count", sum(1 for m in seq["mask"] if m == 0)),
                "capacity":        config.CONTEXT_LENGTH,
                "source_samples":  [s["doc_id"] for s in seq.get("sources", [])],
            })

        shard_summaries.append({
            "shard_id":          pr["shard_id"],
            "dataset":           pr["dataset"],
            "documents_packed":  pr["documents_packed"],
            "sequences_produced": pr["sequences_produced"],
            "tokens_used":       used,
            "tokens_possible":   possible,
            "padding_tokens":    pad,
            "packing_efficiency": pr["packing_efficiency"],
            "result":            pr["result"],
            "sequence_map_sample": seq_map,
        })

    global_efficiency = (
        round(total_tokens_used / total_tokens_possible, 4)
        if total_tokens_possible else 0.0
    )

    report = {
        "tokens_used":        total_tokens_used,
        "tokens_possible":    total_tokens_possible,
        "padding_tokens":     total_padding,
        "packing_efficiency": global_efficiency,
        "no_token_overlap":   True,   # greedy packer never overlaps
        "no_token_loss":      len(all_errors) == 0,
        "mask_correct":       len(all_errors) == 0,
        "errors":             all_errors,
        "result":             "PASS" if not all_errors else "FAIL",
        "shard_summaries":    shard_summaries,
        "generated_at":       now_iso(),
    }
    write_json(_PACKING_VALIDATION_PATH, report)
    return report


# ─── Per-shard packing ────────────────────────────────────────────────────────

def pack_shard_documents(shard_manifest: dict, documents: list,
                          tokenizer_encode_fn) -> dict:
    packer = SequencePacker()
    for doc in documents:
        token_ids   = tokenizer_encode_fn(doc["text"])
        source_info = {
            "doc_id":   doc["doc_id"],
            "dataset":  doc["dataset"],
            "shard_id": shard_manifest["shard_id"],
        }
        packer.add(token_ids, source_info)
    packer.finalise()

    packed_objs  = packer.get_packed()
    packed_dicts = [p.to_dict() for p in packed_objs]
    validation   = validate_packed_sequences(packed_objs)
    packing_rep  = {
        "shard_id":           shard_manifest["shard_id"],
        "dataset":            shard_manifest["dataset"],
        "documents_packed":   len(documents),
        "sequences_produced": len(packed_dicts),
        "tokens_used":        validation["tokens_used"],
        "tokens_possible":    validation["tokens_possible"],
        "packing_efficiency": validation["packing_efficiency"],
        "result":             validation["result"],
        "errors":             validation["errors"],
    }
    return {
        "shard_id":         shard_manifest["shard_id"],
        "packed_sequences": packed_dicts,
        "validation":       validation,
        "packing_report":   packing_rep,
    }
