#!/bin/bash

OPTION_WITH_OVERHEAD="--cudabacktrace=true \
                      --cuda-memory-usage=true"
OPTION_GPU_METRICS="--gpu-metrics-device=1 \
                    --gpu-metrics-frequency=10000"
CUDA_VISIBLE_DEVICES=0 nsys profile -o animatediff_infer  --stats=true --force-overwrite=true \
                       --trace=cuda,cudnn,cublas,cusparse,osrt,nvtx,nvvideo \
                       $OPTION_WITH_OVERHEAD \
                       --capture-range=cudaProfilerApi --capture-range-end stop \
                       python -m scripts.animate --config configs/prompts/5-RealisticVision.yaml
