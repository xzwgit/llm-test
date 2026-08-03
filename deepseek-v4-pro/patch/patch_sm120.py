import os, sys, vllm
OLD1 = """    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales"""
NEW1 = """    if cap.major == 12:
        # SM12x: DeepGEMM einsum consumes raw row-major f32 block scales;
        # the SM100 packed layout produces NaNs here.
        return (1, 128, 128), False
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales"""
OLD2 = """        ws = ws.view(g, r // quant_block_shape[0], d // quant_block_shape[1])
        dg_ws = deepgemm_post_process_weight_scale_block("""
NEW2 = """        ws = ws.view(g, r // quant_block_shape[0], d // quant_block_shape[1])
        cap = current_platform.get_device_capability()
        if cap is not None and cap.major == 12:
            return wq, ws.contiguous()
        dg_ws = deepgemm_post_process_weight_scale_block("""
root = os.path.dirname(vllm.__file__)
print("vllm package:", root)
targets = [
    ("models/deepseek_v4/nvidia/ops/o_proj.py", OLD1, NEW1),
    ("model_executor/layers/quantization/utils/fp8_utils.py", OLD2, NEW2),
]
for rel, old, new in targets:
    path = os.path.join(root, rel)
    src = open(path, encoding="utf-8").read()
    if new in src:
        print("SKIP (already patched):", path)
        continue
    n = src.count(old)
    if n != 1:
        print(f"FAIL: found {n} matches (expected 1) in {path}")
        sys.exit(1)
    open(path + ".bak", "w", encoding="utf-8").write(src)
    open(path, "w", encoding="utf-8").write(src.replace(old, new))
    print("PATCHED:", path)
print("done")
