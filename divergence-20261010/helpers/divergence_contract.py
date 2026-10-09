"""CPU contracts for the observed logical state and bounded prefix."""

from __future__ import annotations

import hashlib
import math
import json


def digest_tokens(tokens):
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":")).encode()
    ).hexdigest()


def prefix_contract(prompt, reference, history, step, input_token, position):
    """Verify the actual committed prefix before a selected decode update."""
    if not 1 <= step <= 19:
        raise ValueError("Observed decode is outside the finite common prefix")
    if len(reference) < 19:
        raise ValueError("Frozen reference does not cover 19 forced tokens")
    expected = list(prompt) + list(reference[:step])
    if list(history) != expected:
        raise ValueError("Committed GPU history differs from the common prefix")
    if input_token != reference[step - 1]:
        raise ValueError("Actual decode input differs from the committed prefix")
    if position != len(expected) - 1:
        raise ValueError("Actual position differs from the committed history")
    return {
        "step": step,
        "total_length": len(expected),
        "input_prefix_sha256": digest_tokens(reference[:step]),
        "prompt_token_ids_sha256": digest_tokens(prompt),
    }


def logical_slots(block_table, valid_tokens, block_size, num_blocks):
    """Map only initialized logical positions; ignore tail and unused pages."""
    if any(not isinstance(item, int) for item in (
        valid_tokens, block_size, num_blocks
    )):
        raise ValueError("KV dimensions must be integers")
    if valid_tokens <= 0 or block_size <= 0 or num_blocks <= 0:
        raise ValueError("KV dimensions must be positive")
    needed = math.ceil(valid_tokens / block_size)
    if len(block_table) < needed:
        raise ValueError("KV block table does not cover the valid prompt")
    pages = list(block_table[:needed])
    if any(not isinstance(page, int) or not 0 <= page < num_blocks
           for page in pages):
        raise ValueError("KV block table references an invalid physical page")
    if len(set(pages)) != len(pages):
        raise ValueError("Aliased prompt KV pages are unsupported")
    return [
        (pages[position // block_size], position % block_size)
        for position in range(valid_tokens)
    ]


def validate_kv_layout(backend, cache_shape, head_size, cache_dtype):
    """Accept only the pinned dense merged K/V layout used by these backends."""
    supported = {
        "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl",
        "vllm.v1.attention.backends.triton_attn.TritonAttentionImpl",
    }
    if backend not in supported:
        raise ValueError("Unreviewed attention KV accessor")
    if cache_dtype not in {"torch.bfloat16", "torch.float16", "torch.float32"}:
        raise ValueError("Quantized or unsupported KV dtype")
    if (len(cache_shape) != 4 or head_size <= 0
            or cache_shape[-1] != 2 * head_size
            or any(item <= 0 for item in cache_shape)):
        raise ValueError("KV cache is not [blocks, heads, block_size, 2*head_size]")
    return {
        "num_blocks": cache_shape[0],
        "num_kv_heads": cache_shape[1],
        "block_size": cache_shape[2],
        "head_size": head_size,
    }


def initial_gate(records, expected_indices, gdn_layers, attention_layers):
    """Missing observed initial state is a NOGO rather than assumed equality."""
    expected = {
        (kind, rank, layer, index)
        for kind, layers in (
            ("recurrent", gdn_layers),
            ("conv", gdn_layers),
            ("kv", attention_layers),
        )
        for rank in (0, 1)
        for layer in layers
        for index in expected_indices
    }
    actual = {
        (row["kind"], row["rank"], row["layer"], row["index"])
        for row in records
    }
    if len(actual) != len(records):
        raise ValueError("Duplicate initial state observation")
    return {
        "status": "INITIAL_COMPLETE" if actual == expected else "NOGO",
        "expected": len(expected),
        "observed": len(actual),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
    }
