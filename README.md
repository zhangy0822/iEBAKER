# EBAKER

Code for **iEBAKER**: *Improved Remote Sensing Image-Text Retrieval Framework via Eliminate Before Align and Keyword Explicit Reasoning*

## Repository Structure

```text
ebaker/       Training, model, loss, and retrieval evaluation code
keyword/      Keyword vocabulary files for keyword explicit reasoning
requirements.txt
README.md
```

## Installation

```bash
conda create -n ebaker python=3.10 -y
conda activate ebaker
pip install -r requirements.txt
```

Install a CUDA-compatible PyTorch version according to your local CUDA environment if needed.

## Data

Training and evaluation CSV files use two columns:

```csv
filename,title
/path/to/image.jpg,A caption sentence .
```

The provided CSV files are examples from the original experiment environment. If the image paths do not exist on your machine, regenerate the CSV files or set the image directory variables in `run.sh`.

Common CSV files:

```text
ebaker/ret3.csv
ebaker/rsicd_test.csv
ebaker/rsitmd_test.csv
ebaker/nwpu_test.csv
```

## Training

```bash
cd ebaker
DATA_ROOT=/path/to/RS/Datasets bash run.sh
```

Useful environment variables:

```bash
CUDA_VISIBLE_DEVICES=0
DATA_ROOT=/path/to/RS/Datasets
TRAIN_CSV=/path/to/train.csv
RSICD_TEST_CSV=/path/to/rsicd_test.csv
RSITMD_TEST_CSV=/path/to/rsitmd_test.csv
NWPU_TEST_CSV=/path/to/nwpu_test.csv
IMAGES_DIR=/path/to/train/images
RETRIEVAL_IMAGES_DIR=/path/to/test/images
EBA_STRATEGY=joint
```

`EBA_STRATEGY` supports:

```text
joint
split
```

Logs are saved under:

```text
ebaker/logs/
```

## Keywords

The keyword reasoning module uses:

```text
keyword/merged_words.txt
```

To use a custom keyword file:

```bash
--keyword-file /path/to/merged_words.txt
```

If needed, regenerate the merged keyword file with:

```bash
python keyword/merge.py
```

## Acknowledgements

This project is built with reference to several excellent open-source projects and resources:

- [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP)
- [GaLR](https://github.com/xiaoyuan1996/GaLR)
- [ITRA](https://itra.readthedocs.io/en/latest/)

## Notes

- The default backbone is OpenCLIP `ViT-B-32`.
- If OpenCLIP weights cannot be downloaded automatically, prepare the cache manually.
- If the BPE vocabulary is not found, install `open_clip_torch` or pass `--bpe-path`.

## Citation

If this repository is helpful for your research, please consider citing:

```bibtex
@article{zhang2026iebaker,
  title={iEBAKER: Improved remote sensing image-text retrieval framework via eliminate before align and keyword explicit reasoning},
  author={Zhang, Yan and Ji, Zhong and Meng, Changxu and Pang, Yanwei},
  journal={Expert Systems with Applications},
  volume={296},
  pages={128968},
  year={2026},
  doi={10.1016/j.eswa.2025.128968}
}
```
