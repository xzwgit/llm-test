#!/usr/bin/env python3
"""b12x compact-N64 TP8 padding patch: 把 per-rank 中间维 288 零填充到 320 (2*128+64),
让 TP8 落进 FlashInfer 的 compact N64 微内核路径 (n % 128 == 64)。

原理:
- FlashInfer _impl.py 按 n%128 三分路径: ==0 走 N128; ==64 走 compact N64 (最快); 其它掉出快路径
- TP4 的 576=4*128+64 命中; TP8 的 288=2*128+32 两边不沾 -> 零填充到 320
- 数学无损: 填充神经元权重全 0 -> silu(0)*0=0, FC2 的 K 维零贡献 (0*任何有限 scale=0)
- 布局: W13Layout 块拼接 [gate(n); up(n)] (或 W31 互换) -> gate 块后插 32 零行 + 尾部 32 零行
- n 的两个消费点同步: 权重张量 + self.intermediate_size_per_partition (惰性建 B12xMoEWrapper 用)

用法: python3 patch_b12x_pad.py [--file PATH] [--check]   (幂等)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PATH = Path(
    "/root/b12xenv/lib/python3.12/site-packages/vllm/model_executor/layers/"
    "fused_moe/experts/flashinfer_b12x_moe.py"
)
MARKER = "[sm120opt] b12x tp8 pad 288->320"

# 注入到 process_weights_after_loading 的 scale 归一化之前
ANCHOR = "    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:\n"

PAD_BLOCK = f'''        # {MARKER}: zero-pad intermediate dim onto the compact-N64 lattice (n % 128 == 64)
        _n = self.intermediate_size_per_partition
        _target = _n if _n % 128 == 64 else (_n // 128) * 128 + 64
        if _target != _n and getattr(layer, "w13_weight", None) is not None:
            import torch as _t
            _pad = _target - _n
            _w13 = layer.w13_weight.data
            _E, _rows = _w13.shape[0], _w13.shape[1]
            assert _rows == 2 * _n, f"b12x pad: w13 rows {{_rows}} != 2*n {{2 * _n}}"
            _zero13 = _t.zeros((_E, _pad, *_w13.shape[2:]), dtype=_w13.dtype, device=_w13.device)
            layer.w13_weight.data = _t.cat(
                [_w13[:, :_n], _zero13, _w13[:, _n:], _zero13], dim=1
            ).contiguous()
            _s13 = layer.w13_weight_scale.data
            if _s13.shape[1] == 2 * _n:
                _zeros13 = _t.ones((_E, _pad, *_s13.shape[2:]), dtype=_s13.dtype, device=_s13.device)
                layer.w13_weight_scale.data = _t.cat(
                    [_s13[:, :_n], _zeros13, _s13[:, _n:], _zeros13], dim=1
                ).contiguous()
            _w2 = layer.w2_weight.data
            _zero2 = _t.zeros((*_w2.shape[:-1], _pad), dtype=_w2.dtype, device=_w2.device)
            layer.w2_weight.data = _t.cat([_w2, _zero2], dim=-1).contiguous()
            _s2 = layer.w2_weight_scale.data
            if _s2.shape[-1] == _n // 16:
                _zeros2 = _t.ones((*_s2.shape[:-1], _pad // 16), dtype=_s2.dtype, device=_s2.device)
                layer.w2_weight_scale.data = _t.cat([_s2, _zeros2], dim=-1).contiguous()
            self.intermediate_size_per_partition = _target

'''


def patch_text(src: str) -> str:
    if MARKER in src:
        return src
    n = src.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"anchor found {n} times (expected 1)")
    return src.replace(ANCHOR, ANCHOR + PAD_BLOCK, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    src = args.file.read_text()
    if MARKER in src:
        print(f"{args.file}: already patched")
        return 0
    patched = patch_text(src)
    compile(patched, str(args.file), "exec")
    if args.check:
        print(f"{args.file}: patch applies cleanly (not written)")
        return 0
    args.file.write_text(patched)
    print(f"{args.file}: patched (pad n -> n%128==64 lattice, compact N64 path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
