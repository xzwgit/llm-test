# ④ NVFP4 KV（nvfp4_ds_mla）SM120 落地结论（.157，2026-10-28）

## 一句话

**NVFP4 KV 在 SM120 上全链路打通（kacper 双补丁：vLLM 3 文件解门禁 + FlashInfer 19 文件 AOT interleaved 读取），KV 池 +55.8%（13.29M vs 8.53M tokens，12.67x@1M），性能持平，确定性保留（patch 机械正确）；但 KV 压缩固有有损——一条简单算术题确定性答错（FP8 腿全对）→ 定位为容量型选项，上生产前须 GSM8K 级质量门。**

## 补丁链（全部幂等留档）

1. `patch.py`（vLLM 3 文件 11 处）：SM120 走 V4.1 双缓存记录 + 能力探针 + dtype 门
2. `dsv41-interleaved.patch` + `patch_fi.py --build`（FlashInfer 19 文件 + nvcc AOT `sparse_mla_sm120.so` 12.0f 3MiB）：SM120 读取器支持 interleaved scale placement（上游 main 至今无此能力，2026-10-28 核实零命中）
3. 顺序坑：先 FI 补丁后探针才转绿（vLLM 探针查 `kv_scale_placement` 参数存在性）

## 数据（k7prob TP8 生产配方，同日双腿）

| 项 | FP8 | NVFP4 | Δ |
|---|---|---|---|
| **KV 池** | 8,529,675 tok（8.13x@1M） | **13,290,418 tok（12.67x@1M）** | **+55.8%** |
| boot | 110s | 110s | 持平 |
| out tok/s c1/c8/c32/64K-1K-c8 | 128.9 / 533.9 / 948.6 / 165.1 | 131.7 / 439.7 / 983.1 / 162.4 | 持平（c8 -17.6% 单样本存疑） |
| probe3 A 层 | distinct=1 全对 | **distinct=1（确定性保留）但 `12*12=` 确定性答错**（答成 10^12 阶乘系） | 有损实锤 |
| probe3 B/C 层 | 3/3 过；C 层 6/8·12/16·20/32 | 3/3 过；**C 层 6/8·12/16·20/32（与 FP8 逐档相同）** | 并发带不变 |

FP8 回归腿（打补丁后）：4 档在历史带内 + C 层与基线带一致 → **补丁对 FP8 生产路径无回归**（footer 逐位一致的宣称实测成立）。

## 判定

- **机械正确**：确定性保留=interleaved 读取无错位（"mismatch decodes garbage"未发生）
- **固有有损**：4-bit 压缩缓存的代价在简单算术上可观测——与 sybil 的 0.974 top-1 一致性量级相称
- **定位**：**容量型选项**——1M 上下文/满并发 KV 受限场景的 opt-in（配方表新行：`--kv-cache-dtype nvfp4_ds_mla`，代价标注"有损 KV，质量门未过"）；默认仍 fp8
- **上生产前置**：GSM8K+logprob 正式质量门量化降幅（预期 GSM8K 降 1-3 分级别）

## 文件
- 本地：D:\llmtest\data\nvfp4kv-157\（双腿+探针+补丁三件套+构建日志全量）
- 远端：/root/nvfp4_kv_ops/；GPU 清零；v3env 已带双补丁（幂等可回退）
