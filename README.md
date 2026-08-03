# llm-test

LLM 部署与性能测试归档。每个子目录是一个独立测试项目。

## 目录结构

```
llm-test/
├── deepseek-v4-pro/    # DeepSeek-V4-Pro 部署与压测 (2026-08)
│   ├── docs/           # 压测报告 + 调优建议 + 配置说明
│   ├── scripts/        # 启动脚本 (start_server01/02.sh, 含 max-num-batched-tokens 优化)
│   ├── bench_scripts/  # 可复用的压测脚本
│   ├── patch/          # SM120 必需补丁
│   └── logs/           # 历史运行日志
├── _template/          # 新测试模板 (后续加测试参考)
└── README.md
```

## 测试项目

| 项目 | 状态 | 日期 | 说明 |
|---|---|---|---|
| [deepseek-v4-pro](deepseek-v4-pro/) | ✅ 完成 | 2026-08 | DP2+MTP 部署, 多上下文压测 |

## 如何新增测试

1. 复制 `_template/` 为新目录 (如 `qwen3-test/`)
2. 子目录内放 `docs/`, `bench_scripts/`, `logs/`
3. 在本 README 表格登记一行
4. git commit + push

## 环境要求

- 目标机: 2× 8-GPU 服务器 (RTX PRO 6000 Blackwell, SM120)
- 互联: IB (ConnectX-8 400G) + 以太网
- 详见各子项目的 docs/
