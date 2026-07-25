#!/bin/bash
set -x
curl -sL "https://huggingface.co/datasets/quentinll/lewm-cube/resolve/main/cube_single_expert.tar.zst" \
  | zstd -dc | tar -x -C /workspace/datasets/lewm_cube_full \
  && echo OK > /workspace/datasets/lewm_cube_full/extract.done
