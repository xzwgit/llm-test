# SM120 上游补丁集

> 在 8×RTX PRO 6000（SM120，消费级 Blackwell，每 block 100 KB 共享内存）上运行 DeepSeek-V4.1-Flash 服务后，向上游提交的已验证补丁与证据。每个子目录包含所提交的补丁及来自我们机器的验证证据。
>
> English version: [README.md](README.md)

| 补丁 | 上游 | 类型 | 归档时状态 |
|---|---|---|---|
| NIXL PD 握手回环诊断 | [vllm-project/vllm#59583](https://github.com/vllm-project/vllm/pull/59583) | 代码 | open |
| NIXL P/D 排障文档（鉴权分层、side-channel 回环） | [vllm-project/vllm#59584](https://github.com/vllm-project/vllm/pull/59584) | 文档 | open |
| `vllm bench serve --metrics-url`（PD router 后取投机解码统计） | [vllm-project/vllm#59587](https://github.com/vllm-project/vllm/pull/59587) | 代码 | open |
| Engram sinkhorn `step_reduce` 适配 100 KB SMEM + 测试容差修正 | [deepseek-ai/TileKernels#37](https://github.com/deepseek-ai/TileKernels/pull/37) | 代码+测试 | open（注：TileKernels 建库以来 0 个外部 PR 被合并——我们同时以 vendored 补丁形式自用） |

带证据的其它上游产物（无补丁）：[TileKernels#36](https://github.com/deepseek-ai/TileKernels/issues/36)（sinkhorn/SM120 失败清单）、[FlashMLA#124](https://github.com/deepseek-ai/FlashMLA/issues/124) 验证评论（SM120 可编译、卡在 227 KB SMEM 计划）。

## 目录结构

```
vllm-59583-nixl-loopback-diagnostic/      patch.diff + verification.txt
vllm-59584-nixl-pd-troubleshooting-docs/  patch.diff
vllm-59587-bench-metrics-url/             patch.diff + verification.txt
tilekernels-37-sinkhorn-smem/             kernel.patch + test.patch + verification.txt + tolerance_simulation.py
```

## 这些补丁为什么存在（共同主线）

SM120 上反复出现的墙是**每 block 100 KB 共享内存**（数据中心 sm_100/sm_103 为 228 KB）——按数据中心规格设计的内核在启动时失败或静默出错。此外还有只在真正部署双节点时才会暴露的 PD 分离运维缺口。四个补丁全部先在我们自己的双节点 SM120 集群上端到端验证过才提交，没有任何一个是推测性的。

TileKernels 补丁保持原累加顺序（数据中心部件上结果逐位一致），仅在设备预算需要时降低拷贝流水深度；配套测试修正替换了默认容差比较——纯 torch 串行行累加在 num_partials=564 时就已越线（该形状只有消费级 Blackwell 的 188 SM 数会生成）。
