# 部署配置

## 架构
- 2 节点，每节点 TP=8；vLLM DP=2（data-parallel rpc 13345）
- 节点1 = DP rank 0，对外 OpenAI API :8123；节点2 = DP rank 1，headless

## 启动命令（节点1；节点2 加 --headless 与 --data-parallel-start-rank 1）

```bash
source ~/vllm/bin/activate
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
export http_proxy="http://<PROXY>:7897/" https_proxy="http://<PROXY>:7897/"
export no_proxy=localhost,127.0.0.1,<SERVER1_IP>,<SERVER2_IP>,<INTERNAL_NET>/24
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export NCCL_SOCKET_IFNAME=ens1f3 GLOO_SOCKET_IFNAME=ens1f3
export NCCL_IB_DISABLE=0 NCCL_IB_HCA=mlx5_2,mlx5_3
export NCCL_NET_GDR_LEVEL=SYS NCCL_P2P_LEVEL=SYS

vllm serve Qwen/Qwen3.8-27B \
  --served-model-name Qwen3.8-27B \
  --trust-remote-code --port 8123 --host 0.0.0.0 \
  --tensor-parallel-size 8 \
  --data-parallel-size 2 --data-parallel-size-local 1 \
  --data-parallel-address <SERVER1_IP> \
  --data-parallel-rpc-port 13345 \
  --dtype auto --max-model-len 262144 --max-num-seqs 512 \
  --gpu-memory-utilization 0.95 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --mm-encoder-tp-mode data \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --api-key <API-KEY>
```

启动顺序：先节点1（rank 0 起 rpc 协调器），约 10 秒后节点2。
两节点权重需同为本地 HF cache（52GB bf16）。

## 压测方法
`vllm bench serve` 官方工具，random 数据集（range-ratio=0），burst 全并发
（num-prompts=并发数、request-rate=inf），`--ignore-eos` 精确输出长度，
`--header "Authorization=Bearer <API-KEY>"` 鉴权，`--tokenizer` 指向本地快照避免外网拉取。
见 `bench_scripts/run_bench.sh`。
