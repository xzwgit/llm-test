#!/usr/bin/env python3
"""Apply/revert Fix B on .155 v3env breakable_cudagraph.py (idempotent, backup kept).

Fix B: env-gated VLLM_BREAKABLE_CAPTURE_ATTN=1 — skip the eager break for the
unified-attention op during breakable capture, letting the attention (incl. its
cuBLAS absorption GEMM) be captured into the segment like FULL mode already
does. Avoids executing nvJet TMA GEMM kernels between capture segments, which
faults on SM120 (Device Unknown Exception, coredump-named kernel).
Note: NOT safe with PD-disaggregation KV transfer (side effects must stay eager);
env-gated opt-in keeps that explicit.
"""
import os
import shutil
import sys

F = "/root/v3env/lib/python3.12/site-packages/vllm/compilation/breakable_cudagraph.py"
BAK = F + ".stock_bak_fixb"

ORIG = """        if not capture._capturing:
            return fn(*args, **kwargs)
        if is_forward_context_available():"""
FIX = """        if not capture._capturing:
            return fn(*args, **kwargs)
        # FIXB-SM120: capture attention-family ops into the segment instead of
        # eager-break (opt-in; NOT for PD-disagg KV transfer; KDA must keep breaking)
        import os as _os
        if _os.environ.get("VLLM_BREAKABLE_CAPTURE_ATTN") == "1" and "mla" in getattr(fn, "__name__", ""):
            return fn(*args, **kwargs)
        if is_forward_context_available():"""


def apply():
    src = open(F).read()
    if "FIXB-SM120" in src:
        print("already patched")
        return
    assert src.count(ORIG) == 1, f"anchor count={src.count(ORIG)}"
    if not os.path.exists(BAK):
        shutil.copy2(F, BAK)
    open(F, "w").write(src.replace(ORIG, FIX))
    print("FIXB applied, backup:", BAK)


def revert():
    if os.path.exists(BAK):
        shutil.copy2(BAK, F)
        print("reverted from backup")
    else:
        src = open(F).read()
        if "FIXB-SM120" in src:
            open(F, "w").write(src.replace(FIX, ORIG))
            print("reverted by reverse replace")
        else:
            print("nothing to revert")


def check():
    src = open(F).read()
    print("FIXB present" if "FIXB-SM120" in src else "stock")
    print("os import:", "import os" in src or src.count("os.environ") > 0)


if __name__ == "__main__":
    {"apply": apply, "revert": revert, "check": check}[sys.argv[1]]()
