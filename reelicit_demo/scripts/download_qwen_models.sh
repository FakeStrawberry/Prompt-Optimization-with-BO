#!/usr/bin/env bash
set -euo pipefail

mkdir -p /nas1/zyj/models /nas1/zyj/hf-cache

# On this machine, httpx may pick the SOCKS all_proxy and fail with
# "Network is unreachable". The HTTP proxy works, so unset all_proxy.
unset all_proxy
unset ALL_PROXY

export HF_HOME=/nas1/zyj/hf-cache
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HUB_ETAG_TIMEOUT=60

hf download Qwen/Qwen3-8B \
  --local-dir /nas1/zyj/models/Qwen3-8B \
  --max-workers 2

hf download Qwen/Qwen3-14B \
  --local-dir /nas1/zyj/models/Qwen3-14B \
  --max-workers 2
