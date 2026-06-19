#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/experiment/000_smoke_toy_opd.yaml}"
echo "Launch GPU job for ${CONFIG} using your scheduler."

