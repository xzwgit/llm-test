# ① b12x TP8 padding 验证结论（.155，2026-10-27，Phase C）

## 一句话

**padding 让 b12x TP8 成功 boot 并跑完全部档位（10 月的 288 硬阻塞解除），但质量门 FAIL：Layer A 温度 0 确定性破防（8 重复 7 种输出+退化重复），且 boot 日志无 compact N64 路径选中证据、性能无优势——b12x TP8 路线暂停，不建议继续投入。**

## 证据

1. PAD-v2 boot 成功（200s，323 探针过，`Using 'B12X_MXFP4_MXFP8' backend`，6/6 档 0 失败）——padding 本身工程上成立
2. **质量门 FAIL**：Layer A distinct=4~8（BASE 腿同探针 distinct=1 全对）——与 .162 v4 十月未解的 TP8 质量疑点（32319 续写异常）同族
3. **无 compact 路径生效证据**：boot 日志无 compact/grouped_contiguous 行——可能走了回退路径（若回退，损坏在回退路径而非 padding）
4. 性能无优势：c1=130/95 tok/s（与 DeepGEMM BASE 同带），未见 .162 v4 时代的 +18.8% 单流优势
5. 首轮 PAD 失败原因=b12x oracle 拒 EP（十月已立案），v2 去 EP 后 boot 通过

## 判定与去向

- **判定：b12x TP8 暂停**。两层未解：padding 布局假设（W13 块拼接中插）可能错 / FlashInfer b12x 该 shape 内核数值 bug（与十月疑点同源）
- 若要重启需：①TP4 对照腿（同 env 同 patch，验十月干净结论在 dev466 是否仍成立）②padding 离线单测（参考 matmul 逐位比对）——深度 kernel 调试，成本高收益不确定
- patch 与全部证据留档（patch_b12x_pad.py 幂等可复用；b12xenv 已写入，回退=删 MARKER 段）

## 文件
data/b12x-pad-155/（BASE/PAD/PAD2 三腿全量）+ 远端 /root/b12x_pad_ops/；GPU 清零
