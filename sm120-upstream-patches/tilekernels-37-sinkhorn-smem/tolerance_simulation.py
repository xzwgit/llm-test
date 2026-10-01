# -*- coding: utf-8 -*-
"""Reproduce the accumulation-order rounding that the default test tolerance flags.

A pure-torch row-serial accumulation (the kernel's order) vs torch.sum
(pairwise tree) reaches ~1.4e-6 relative error on some columns at
num_partials=564 -- right at torch.testing.assert_close's float32 default
rtol of 1.3e-6. No kernel is involved: this shows the residual mismatch
is algorithmic rounding, not a kernel bug.

Run on the target device (we verified on RTX PRO 6000, SM120).
"""
import torch

torch.manual_seed(42)
for n, h in [(564, 5120), (564, 256), (507, 5120)]:
    for t in range(3):
        x = torch.rand(n, h, device='cuda')
        ref = x.sum(0)
        acc = torch.zeros(h, device='cuda')
        for r in range(n):
            acc += x[r]  # the kernel's accumulation order
        diff = (acc - ref).abs()
        allowed = 1e-5 + 1.3e-6 * ref.abs()  # assert_close float32 defaults
        viol = (diff > allowed).sum().item()
        print(f"pure-serial n={n} h={h} t{t}: "
              f"max_rel={(diff / ref.abs()).max().item():.3e} violations={viol}/{h}")
