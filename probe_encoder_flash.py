#!/usr/bin/env python3
"""Compare encoder flash (flat T) to dense (B, L) SDPA. Torch only — no spyre_inference.

Numeric refs always run on CPU. Spyre seq-dim slices at a non-zero offset are
corrupt (see examples/experimental/spyre_online_softmax_check.py); the per-seq
SDPA on device is not a valid ground truth.

Two flash modes:

``--blocked`` (the target design) never slices. It hands the kernel the whole
flat ``[T, H, D]`` list plus int32 row-index tables, and the kernel gathers
in-graph with ``index_select``, walks KV in 64-token blocks carrying a running
softmax max/sum, and stores through ``index_copy_``. Offsets are data, not
shapes, so the only compile axis is the block count. This mirrors the decoder's
``_create_compilable_page_attn``.

Default mode is the older host-slice + H2D per-sequence ``L×L`` GEMM, kept for
comparison. It exists because a compiled region reads its arguments from offset
0 (torch-spyre#3770), so slicing outside the graph reads the wrong rows.

    python probe_encoder_flash.py --blocked
    python probe_encoder_flash.py --device spyre --compile --blocked
    python probe_encoder_flash.py --device spyre --compile --blocked --seqs 4 --lens 62,62,62,62
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F

ALIGN = 64


def _align_up(n: int, align: int = ALIGN) -> int:
    return max(align, (n + align - 1) // align * align)


def _pad_seq(seq: torch.Tensor, aligned_len: int) -> torch.Tensor:
    n = seq.shape[0]
    if n == aligned_len:
        return seq
    return torch.cat([seq, seq.new_zeros((aligned_len - n, *seq.shape[1:]))], dim=0)


def _make_flash_kernel(aligned_len: int, num_heads: int, num_kv_heads: int, head_size: int):
    """Aligned ``L×L`` GEMM plus same-shape ``[H,L,L]`` 0/1 (no 1D broadcast)."""
    del aligned_len, head_size
    repeat = num_heads // num_kv_heads if num_heads != num_kv_heads else 1
    gqa = repeat > 1
    pad_logit = -1.0e4

    def _kernel(query, key, value, keep, scale):
        qh = query.permute(1, 0, 2) * scale
        kh = key.permute(1, 0, 2)
        vh = value.permute(1, 0, 2)
        if gqa:
            kh = kh.repeat_interleave(repeat, dim=0)
            vh = vh.repeat_interleave(repeat, dim=0)
        scores = torch.matmul(qh, kh.transpose(-1, -2))
        scores = scores + (1.0 - keep) * pad_logit
        scores_max = scores.amax(dim=-1)
        probs = torch.exp(scores - scores_max.unsqueeze(-1)) * keep
        tiny = torch.finfo(query.dtype).tiny
        den = probs.sum(dim=-1).clamp(min=tiny).unsqueeze(-1)
        out = torch.matmul(probs, vh) / den
        return out.permute(1, 0, 2).contiguous()

    return _kernel


def _host_seq(t: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """Slice on CPU. Spyre ``t[start:start+length]`` at start>0 is corrupt."""
    return t.detach().cpu()[start : start + length]


def _flash_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
    *,
    compile_enabled: bool,
    run_device: torch.device,
) -> torch.Tensor:
    cache: dict[tuple[int, int, int, int], object] = {}
    out = query.new_empty(query.shape).cpu()
    start = 0
    for length in query_lens:
        aligned = _align_up(length)
        q = _pad_seq(_host_seq(query, start, length), aligned)
        k = _pad_seq(_host_seq(key, start, length), aligned)
        v = _pad_seq(_host_seq(value, start, length), aligned)
        if run_device.type != "cpu":
            q, k, v = q.to(run_device), k.to(run_device), v.to(run_device)
        key_id = (aligned, q.shape[1], k.shape[1], q.shape[2])
        fn = cache.get(key_id)
        if fn is None:
            print(f"    compile/load kernel L={aligned}", flush=True)
            raw = _make_flash_kernel(*key_id)
            fn = torch.compile(raw, dynamic=False) if compile_enabled else raw
            cache[key_id] = fn
        keep = torch.zeros(q.shape[1], aligned, aligned, dtype=torch.float32)
        keep[:, :, :length] = 1
        keep = keep.to(device=q.device, dtype=q.dtype)
        got = fn(q, k, v, keep, scale).cpu()
        out[start : start + length] = got[:length]
        start += length
    return out


def _make_block_kernel(
    n_blocks: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    *,
    needs_q_gather: bool,
    needs_kv_gather: bool,
):
    """Blocked online softmax over the flat list. ``aligned_q == n_blocks * ALIGN``."""
    per_kv = num_heads // num_kv_heads
    aligned_q = n_blocks * ALIGN

    def _kernel(query, key, value, out, q_rows, kv_rows, mask_tiles, scale):
        # A gather that selects its whole source faults the card
        # (torch-spyre#4033), so the caller decides whether it is needed.
        q_flat = query.index_select(0, q_rows) if needs_q_gather else query
        q = q_flat.unsqueeze(0).transpose(1, 2).reshape(num_kv_heads, per_kv, aligned_q, head_size)
        if needs_kv_gather:
            k_flat = key.index_select(0, kv_rows)
            v_flat = value.index_select(0, kv_rows)
        else:
            k_flat, v_flat = key, value
        k_blocks = k_flat.reshape(n_blocks, ALIGN, num_kv_heads, head_size)
        v_blocks = v_flat.reshape(n_blocks, ALIGN, num_kv_heads, head_size)

        tile_max = tile_sum = tile_out = None
        for i in range(n_blocks):
            k_blk = k_blocks[i].permute(1, 0, 2).unsqueeze(1)
            v_blk = v_blocks[i].permute(1, 0, 2).unsqueeze(1)
            scores = torch.matmul(q, k_blk.transpose(-2, -1)) * scale + mask_tiles[i]
            block_max = torch.amax(scores, dim=-1, keepdim=True)
            if i == 0:
                tile_max = block_max
                probs = torch.exp(scores - tile_max)
                tile_out = torch.matmul(probs, v_blk)
                tile_sum = probs.sum(dim=-1, keepdim=True)
                continue
            new_max = torch.maximum(tile_max, block_max)
            rescale = torch.exp(tile_max - new_max)
            probs = torch.exp(scores - new_max)
            tile_out = tile_out * rescale + torch.matmul(probs, v_blk)
            tile_sum = tile_sum * rescale + probs.sum(dim=-1, keepdim=True)
            tile_max = new_max

        attn = (tile_out / tile_sum).reshape(1, num_heads, aligned_q, head_size)
        attn = attn.transpose(1, 2).reshape(aligned_q, num_heads, head_size)
        # Pad rows repeat the sequence's last row, so the duplicate-index
        # writes here store the same value the real row already got.
        out.index_copy_(0, q_rows, attn)
        return out

    return _kernel


def _row_table(start: int, length: int, extent: int, dtype: torch.dtype) -> torch.Tensor:
    """Absolute rows, pad lanes clamped to the last real row.

    int32 on Spyre, which has no int64 and whose compiled ``index_copy_`` takes
    it; int64 everywhere else, where eager ``index_copy_`` rejects int32.
    """
    pos = torch.arange(extent, dtype=dtype)
    return start + pos.clamp(max=max(length - 1, 0))


def _mask_tiles(n_blocks: int, kv_len: int, num_kv_heads: int, per_kv: int, dtype):
    """One additive ``[KV, per_kv, 1, ALIGN]`` tile per block; interiors share a zero."""
    zero = torch.zeros(num_kv_heads, per_kv, 1, ALIGN, dtype=dtype)
    neg = torch.finfo(dtype).min
    tiles = []
    for i in range(n_blocks):
        lo = i * ALIGN
        if lo + ALIGN <= kv_len:
            tiles.append(zero)
            continue
        tile = torch.full((num_kv_heads, per_kv, 1, ALIGN), neg, dtype=dtype)
        keep = max(0, min(ALIGN, kv_len - lo))
        if keep:
            tile[..., :keep] = 0
        tiles.append(tile)
    return tiles


def _blocked_varlen(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_lens: list[int],
    scale: float,
    *,
    compile_enabled: bool,
    run_device: torch.device,
) -> torch.Tensor:
    """Flat-list blocked flash: no slicing, offsets ride in int32 index tables."""
    num_heads, head_size = query.shape[1], query.shape[2]
    num_kv_heads = key.shape[1]
    per_kv = num_heads // num_kv_heads
    total = query.shape[0]

    q_dev = query.to(run_device)
    k_dev = key.to(run_device)
    v_dev = value.to(run_device)
    out = torch.zeros_like(q_dev)

    index_dtype = torch.int32 if run_device.type == "spyre" else torch.int64
    cache: dict[tuple, object] = {}
    start = 0
    for length in query_lens:
        n_blocks = max(1, (length + ALIGN - 1) // ALIGN)
        extent = n_blocks * ALIGN
        rows = _row_table(start, length, extent, index_dtype)
        needs_q_gather = not (start == 0 and length == extent and total == extent)
        key_id = (n_blocks, num_heads, num_kv_heads, head_size, needs_q_gather)
        fn = cache.get(key_id)
        if fn is None:
            print(f"    compile/load blocked kernel n_blocks={n_blocks} gather={needs_q_gather}")
            raw = _make_block_kernel(
                n_blocks,
                num_heads,
                num_kv_heads,
                head_size,
                needs_q_gather=needs_q_gather,
                needs_kv_gather=needs_q_gather,
            )
            fn = torch.compile(raw, dynamic=False) if compile_enabled else raw
            cache[key_id] = fn
        rows_dev = rows.to(run_device)
        tiles = [
            t.to(run_device)
            for t in _mask_tiles(n_blocks, length, num_kv_heads, per_kv, query.dtype)
        ]
        fn(q_dev, k_dev, v_dev, out, rows_dev, rows_dev, tiles, scale)
        start += length
    return out.cpu()


def _dense_grid_sdpa(query, key, value, query_lens, scale) -> torch.Tensor:
    """Shipped path: pack into (B, L), one SDPA, unpack. CPU tensors only."""
    query, key, value = query.cpu(), key.cpu(), value.cpu()
    batch = len(query_lens)
    aligned_len = _align_up(max(query_lens, default=1))
    heads, dim = query.shape[1], query.shape[2]
    kv_heads = key.shape[1]
    q_b = torch.zeros(batch, heads, aligned_len, dim, dtype=query.dtype)
    k_b = torch.zeros(batch, kv_heads, aligned_len, dim, dtype=query.dtype)
    v_b = torch.zeros_like(k_b)
    mask = torch.full(
        (batch, 1, aligned_len, aligned_len),
        torch.finfo(query.dtype).min,
        dtype=query.dtype,
    )
    start = 0
    slots: list[tuple[int, int, int]] = []
    for s, length in enumerate(query_lens):
        q_b[s, :, :length] = query[start : start + length].transpose(0, 1)
        k_b[s, :, :length] = key[start : start + length].transpose(0, 1)
        v_b[s, :, :length] = value[start : start + length].transpose(0, 1)
        mask[s, 0, :length, :length] = 0
        slots.append((s, start, length))
        start += length
    kwargs: dict = {"is_causal": False, "scale": scale, "attn_mask": mask}
    if query.shape[1] != key.shape[1]:
        kwargs["enable_gqa"] = True
    attn = F.scaled_dot_product_attention(q_b, k_b, v_b, **kwargs)
    flat = query.new_empty(query.shape)
    for s, start, length in slots:
        flat[start : start + length] = attn[s, :, :length].transpose(0, 1)
    return flat


def _per_seq_sdpa(query, key, value, query_lens, scale) -> torch.Tensor:
    query, key, value = query.cpu(), key.cpu(), value.cpu()
    outs: list[torch.Tensor] = []
    start = 0
    for length in query_lens:
        q = query[start : start + length].unsqueeze(0).transpose(1, 2)
        k = key[start : start + length].unsqueeze(0).transpose(1, 2)
        v = value[start : start + length].unsqueeze(0).transpose(1, 2)
        kwargs: dict = {"is_causal": False, "scale": scale}
        if query.shape[1] != key.shape[1]:
            kwargs["enable_gqa"] = True
        out = F.scaled_dot_product_attention(q, k, v, **kwargs)
        outs.append(out.transpose(1, 2).squeeze(0))
        start += length
    return torch.cat(outs, dim=0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.reshape(-1).float(), b.reshape(-1).float()
    return float(F.cosine_similarity(x, y, dim=0).item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--blocked", action="store_true", help="flat-list blocked kernel")
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--kv-heads", type=int, default=12)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--seqs", type=int, default=3)
    parser.add_argument("--lens", default="30,12,8")
    parser.add_argument("--cosine-min", type=float, default=0.99)
    parser.add_argument("--max-abs", type=float, default=1e-2)
    args = parser.parse_args()

    device = torch.device(args.device)
    lens = [int(x) for x in args.lens.split(",") if x]
    if len(lens) == 1 and args.seqs > 1:
        lens = lens * args.seqs
    t = sum(lens)
    mode = "blocked" if args.blocked else "keep_hl"
    print(
        f"probe_encoder_flash torch-only {mode} seqs={lens} T={t} "
        f"compile={args.compile} device={device}",
        flush=True,
    )

    torch.manual_seed(0)
    dtype = torch.float16 if device.type == "spyre" else torch.float32
    q = torch.randn(t, args.heads, args.dim, dtype=dtype)
    k = torch.randn(t, args.kv_heads, args.dim, dtype=dtype)
    v = torch.randn(t, args.kv_heads, args.dim, dtype=dtype)
    scale = args.dim**-0.5

    print("  cpu refs...", flush=True)
    dense = _dense_grid_sdpa(q, k, v, lens, scale)
    per_seq = _per_seq_sdpa(q, k, v, lens, scale)
    runner = _blocked_varlen if args.blocked else _flash_varlen
    axis = "block count" if args.blocked else "aligned L"
    print(f"  flash on {device} (one compile per {axis})...", flush=True)
    try:
        flash = runner(q, k, v, lens, scale, compile_enabled=args.compile, run_device=device)
    except Exception:
        import traceback

        traceback.print_exc()
        print("  FAIL (exception during flash)", flush=True)
        return 1
    print("  flash done", flush=True)

    flash_c, dense_c, ref_c = flash.float(), dense.float(), per_seq.float()
    cos_dense = _cosine(flash_c, dense_c)
    cos_ref = _cosine(flash_c, ref_c)
    max_abs_dense = (flash_c - dense_c).abs().max().item()
    max_abs_ref = (flash_c - ref_c).abs().max().item()
    print(
        f"  flash vs dense (B,L) SDPA (cpu): cosine={cos_dense:.6f} max|diff|={max_abs_dense:.4e}"
    )
    print(f"  flash vs per-seq SDPA (cpu):     cosine={cos_ref:.6f} max|diff|={max_abs_ref:.4e}")
    off = 0
    for i, length in enumerate(lens):
        c = _cosine(flash_c[off : off + length], ref_c[off : off + length])
        d = (flash_c[off : off + length] - ref_c[off : off + length]).abs().max().item()
        print(f"    seq{i} len={length}: cosine={c:.6f} max|diff|={d:.4e}")
        off += length
    abs_lim = max(args.max_abs, 5e-2 if dtype == torch.float16 else args.max_abs)
    ok = (
        cos_dense >= args.cosine_min
        and cos_ref >= args.cosine_min
        and max_abs_dense <= abs_lim
        and max_abs_ref <= abs_lim
        and math.isfinite(cos_dense)
    )
    print("  PASS" if ok else "  FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
