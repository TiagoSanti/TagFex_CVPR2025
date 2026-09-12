#!/usr/bin/env bash
# Compatibility entry point; canonical implementation: experiments/imagenet100/run_imagenet100_50-10_ant_queue.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$ROOT/experiments/imagenet100/run_imagenet100_50-10_ant_queue.sh" "$@"
