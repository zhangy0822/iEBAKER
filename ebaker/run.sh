#!/usr/bin/env bash
set -euo pipefail

# Conference EBAKER training entry.
#
# Usage:
#   cd ebaker
#   DATA_ROOT=/path/to/RS/Datasets bash run.sh
#
# The bundled CSV files still contain the original absolute paths from the
# authors' machine. For a new machine, either regenerate the CSV paths or set
# TRAIN_CSV/RSICD_TEST_CSV/RSITMD_TEST_CSV/NWPU_TEST_CSV to your own files.

cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-/path/to/RS/Datasets}"
TRAIN_CSV="${TRAIN_CSV:-ret3.csv}"
RSICD_TEST_CSV="${RSICD_TEST_CSV:-rsicd_test.csv}"
RSITMD_TEST_CSV="${RSITMD_TEST_CSV:-rsitmd_test.csv}"
NWPU_TEST_CSV="${NWPU_TEST_CSV:-nwpu_test.csv}"
IMAGES_DIR="${IMAGES_DIR:-}"
RETRIEVAL_IMAGES_DIR="${RETRIEVAL_IMAGES_DIR:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EBA_STRATEGY="${EBA_STRATEGY:-joint}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

torchrun --nproc_per_node "${NPROC_PER_NODE}" -m training.main \
    --train-data "${TRAIN_CSV}" \
    --retrieval-data1 "${RSICD_TEST_CSV}" \
    --retrieval-data2 "${RSITMD_TEST_CSV}" \
    --retrieval-data3 "${NWPU_TEST_CSV}" \
    --retrieval-frequency 1 \
    --w_simcse 1.0 \
    --dropratio 0.01 \
    --dropepoch 4 \
    --eba-strategy "${EBA_STRATEGY}" \
    --csv-img-key filename \
    --csv-caption-key title \
    --retrieval-csv-img-key filename \
    --retrieval-csv-caption-key title \
    --images-dir "${IMAGES_DIR}" \
    --retrieval-images-dir "${RETRIEVAL_IMAGES_DIR}" \
    --datasets-dir "${DATA_ROOT}" \
    --epochs 7 \
    --save-frequency 0 \
    --batch-size 100 \
    --workers 8 \
    --lr 1.5e-05 \
    --warmup 200 \
    --weight_decay 0.7 \
    --max-grad-norm 50.0 \
    --image-model ViT-B-32 \
    --image-model-builder openclip \
    --text-model ViT-B-32 \
    --text-model-builder openclip \
    --pretrained-image-model \
    --pretrained-text-model \
    --loss InfoNCE \
    --report-to tensorboard \
    --logs logs/RS5M \
