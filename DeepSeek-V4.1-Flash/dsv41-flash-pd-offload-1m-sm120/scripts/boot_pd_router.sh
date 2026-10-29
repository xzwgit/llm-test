#!/bin/bash
exec /root/vllm/bin/vllm-router --host 0.0.0.0 --port 8130 --api-key <API_KEY> --vllm-pd-disaggregation --prometheus-host 0.0.0.0 --prefill http://192.168.200.3:8123 14579 --decode http://192.168.200.4:8123
