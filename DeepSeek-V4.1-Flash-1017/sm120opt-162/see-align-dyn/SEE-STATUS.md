# SEE 项目状态快照 · 2026-10-25 上午

> 本文件 = SEE fork 线的全部已完成/进行中/待做工作快照。后续会话读本文件 + `xzwgit/vllm see 分支的 SEE.md` + `llm-test DeepSeek-V4.1-Flash-1017/sm120opt-162/` 即可接手。

## 分支与机器

- **分支**：`xzwgit/vllm` → **`see`**（head `cc17a5b3e`，自 main 分出，章程 SEE.md）
- **.162**（SEE 线）：venv `/root/sm120opt`（dev451），源码 `/root/sm120opt_src`，产物 `/root/sm120opt_ops/`
- **.158**（k7 线 / 上游）：venv `/root/v60762`（dev440 stock），产物 `/root/test_ops/`
- **权重**：两台均在 `/data/models/DeepSeek-V4.1-Flash-1017`（.162 是 /root/ 软链→/data/）

## 已完成（按时间序）

### 1. b12x MoE 首轮实验（已封存）
- **结论**：快但不可信。TPMoe 路径 c1 +18.8% 但精确任务静默损坏；functional API 122/122 全过但只支持 W4A4（无 W4A8-MX）
- **产出**：llm-test `DeepSeek-V4.1-Flash-1017/sm120opt-162/` 根目录（b12x_*.log 系列）
- **封存原因**：正确修法=自研 fused kernel（kacper 样板）或等 FI 补 W4A8 路线

### 2. DeepGEMM 动态对齐 patch（★ 最大成果）
- **发现**：vendored DeepGEMM 在 SM120 返回 alignment=128，导致 MoE grouped GEMM 每个专家 pad 到 128 行、BLOCK_M=128 tile 跑 1-6 实行
- **修法**：per-shape 动态——expected per-expert M ≤ 64 时对齐降 64；三点同步（apply×2 / permute / warmup）
- **commit**：see@077fa3316（3 文件：deep_gemm_utils / deep_gemm_moe / deep_gemm_warmup）
- **验证**：
  - .162 dev451: c1 252→284 (+12.5%) / c32 973→1094 (+12.4%) / 8K c8 546→826 (+51.3%) / 64K c4 804（新档）
  - .158 dev440: 8K c8 三连 663/666/642 vs stock 605.8 (+8.5% 稳定)；c32 方差带内无回归
  - 加固矩阵（.162, 7 档×2 轮）：1K c64 1518/1576, 4K c8 558/671, 32K c8 666/623 等
  - 质量三连（323/9.9大/北京）全过
- **归档**：llm-test `DeepSeek-V4.1-Flash-1017/sm120opt-162/see-align-dyn/`（A/B 日志 + patch 文件 + validation matrix）
- **注意**：.158 的 v60762 已还原 stock（k7 线机器保持干净）；.162 的 sm120opt 仍带 patch

### 3. #60762 新环境 A/B 验证（已交付）
- .158 全新 venv（dev440 stock）→ 裸 main 180s 死 `attention.hpp:166` → +PR 两文件 READY 255s + 323 ✓
- 帖 6381833361 已发

### 4. k7 完整矩阵（.158，已交付）
- 29 档+VL E2E+Agent 3×12 全零失败，c32 1006 tok/s，接受率 Σ 全程 56.7%
- 报告：`压测报告-HTML/DSV4.1-Flash-1017-PRO6000D-FlashInfer072-DSpark-k7prob-完整矩阵.html`
- 原始：llm-test `DeepSeek-V4.1-Flash-1017/dsv41-flash-1017-6000d-sm120/logs/k7full-prob/` @ 141bd7aae

### 5. A1 KV 分层（已定案）
- 容量实证：512K×c32=16.4M token > GPU 池 12.66M 全过（TTFT 均 602s，出速 92.7 tok/s）
- 热路径代价：1K c32 837 vs 1006（-17%）→ **opt-in 专用，不作默认**

### 6. A2 indexer mxfp4（已关闭）
- 上游硬拒 SM120（"requires Blackwell datacenter GPUs sm_10x"）→ 不可用

### 7. #63473 防呆 PR（已两轮收窄）
- hclsys 第一轮（NVFP4 误伤）→ 收窄到 activation_key==kMxfp8Dynamic（e194fd8c1）
- hclsys 第二轮（SM121 干净）→ 收窄到精确 cc(12,0)（aeea41630）
- 当前 head aeea41630，等 review/CI

### 8. b12x 数值隔离（已封存）
- b12x_numeric.py harness：TPMoePlan 无 scratch_specs 属性 → 无法直接调 kernel 做 bit-identity
- w4a8/weights.py 的 repack 函数已读（[E,N,K/2]+[E,N,K/32]→N256/K128 tile 重排），但 execute 层 API 不可达

## 进行中 / 未完成

### THP (use_thp:true) 探测（nothp 基线已跑，thp 腿需补跑）
- nothp 基线：c1 229 / c32 1079 / 8K c8 567 / 64K c4（中断）
- thp 腿：**未跑**（脚本中断，boot_thp.log 不存在）
- 数据：本地 `data/thp-probe/`（不完整）

### PR#318 kernel 摘入（已评估，不可行）
- vendored DG 的 csrc/ 已编译进 _C_extension.abi3.so → 不能盖文件
- 新 kernel 的 host 侧调度在预编译 .so 里，只有 JIT .cuh 可加但没有 entry point
- **需要重建 _C_extension 才能用 PR318**——工作量大，暂缓

## 待做（优先级序）

1. **对齐 patch 上游化**：干净的 3 文件改动，直接对 vllm main 提 PR
2. **THP 补跑**：.158 上一晚出数
3. **对齐 patch 正式矩阵化**：在 .162 上跑 k7full 同款 29 档矩阵（patched vs stock），出规范级报告
4. **b12x fused kernel**：vendor kacper 的 .cu（Apache-2.0）或等 FI 补 W4A8 路线
5. **use_thp 配方**：如 THP 有效，写入部署方案

## GitHub 归档索引

| 位置 | 内容 | commit |
|---|---|---|
| `xzwgit/vllm see` | 分支本体（章程+对齐 patch） | cc17a5b3e |
| `xzwgit/llm-test DeepSeek-V4.1-Flash-1017/sm120opt-162/` | b12x 首轮+b12x 封存+对齐 patch A/B+加固矩阵 | b06703517/0ea4a3232 |
| `xzwgit/llm-test DeepSeek-V4.1-Flash-1017/dsv41-flash-1017-6000d-sm120/logs/k7full-prob/` | k7 完整矩阵原始数据 | 141bd7aae |
| `xzwgit/llm-test DeepSeek-V4.1-Flash-1017/dsv41-flash-1017-6000d-sm120/logs/see-align-dyn/` | 对齐 patch .158 跨机验证 | b06703517 |
| 本地 `data/see-align/` | 对齐 patch 文件+.162 日志 | — |
| 本地 `data/see-align-158/` | .158 D3+repeat 日志 | — |
| 本地 `data/k7full/` | k7 完整矩阵原始数据 | — |
| 本地 `data/k7-test/` | k7 快测 | — |
| 本地 `data/thp-probe/` | THP 探测（不完整） | — |
| 本地 `data/sm120opt-162/` | b12x 首轮全部产物 | — |
