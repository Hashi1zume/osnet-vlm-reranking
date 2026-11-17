# YOLO26 + OSNet + Qwen2.5-VL for Video Person Re-Identification

[日本語版 README](README.ja.md)

This repository evaluates whether a vision-language model can reduce identity changes when a person disappears and later re-enters a camera view. YOLO26n detects people, OSNet retrieves visually similar past observations, and Qwen2.5-VL checks candidate image pairs before an existing identity is reused.

## Project overview

For each detected person crop, OSNet produces a 512-dimensional feature. The temporal gallery ranks up to 10 past candidates by cosine similarity. The VLM compares the query crop with each candidate. A prior global ID is reused only when the VLM returns a valid same-person decision with confidence at least 0.8 and raw OSNet cosine similarity at least 0.75. If no candidate passes both conditions, a new ID is assigned.

The public validation result on the first 214 frames of MOT20-01 is:

| Metric | Temporal OSNet | OSNet + Qwen2.5-VL gate |
|---|---:|---:|
| HOTA | 14.889 | 16.576 |
| AssA | 52.804 | 65.507 |
| IDF1 | 7.700 | 9.356 |
| ID switches | 94 | 0 |

This is one short sequence, and YOLO recall on this run was about 4.9%. The result supports an improvement in identity association on this sequence; it does not establish general performance across cameras or datasets. The MSMT17-trained OSNet checkpoint is a supervised OSNet baseline, not a VLM-trained OSNet. Qwen2.5-VL is used for inference-time image comparison and is not fine-tuned here.

## Repository contents

- `reid_pipeline/`: detection-to-identity assignment, OSNet retrieval, VLM reranking, and evaluation modules.
- `scripts/`: video inference, TrackEval export, weight download, OSNet training, and MOT20 evaluation commands.
- `configs/`: portable MSMT17 OSNet training configuration.
- `tests/`: unit tests for the project modules.
- `patches/`: NumPy 2 compatibility patch for the pinned TrackEval revision.

Drive clips, datasets, model checkpoints, generated videos, slides, and research reports are intentionally excluded. The `.gitignore` keeps those local artifacts out of commits. MSMT17 and MOT20 must be obtained separately from their providers.

## Environment setup

Tested in Python 3.12 with PyTorch 2.15.0 + CUDA 13.2, TorchVision 0.29.2 + CUDA 13.2, and Ultralytics 8.4.213. PyTorch/CUDA wheels depend on the host operating system and GPU driver, so install a matching PyTorch and TorchVision pair using the [official PyTorch selector](https://pytorch.org/get-started/locally/).

```bash
conda env create -f environment.yml
conda activate yolo26-osnet-vlm

# Install the CUDA or CPU build selected for this machine from pytorch.org.
python -m pip install torch torchvision

python -m pip install -r requirements.txt
./scripts/setup_dependencies.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`setup_dependencies.sh` checks out fixed revisions of [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) and [TrackEval](https://github.com/JonathonLuiten/TrackEval). It applies the included compatibility patch to TrackEval's removed NumPy aliases.

Install and start [Ollama](https://ollama.com/download), then fetch the VLM:

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

In another terminal, download the detector and MSMT17-trained OSNet weights:

```bash
./weights/download_weights.sh
```

The script saves `yolo26n.pt` and `osnet_x0_25_msmt17.pth` under the ignored `weights/` directory. YOLO26n is downloaded from the [official Ultralytics assets release](https://github.com/ultralytics/assets/releases/tag/v8.4.0); the OSNet checkpoint is the MSMT17 model-zoo weight published by [Torchreid](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md).

## OSNet training

Request the MSMT17 dataset from its [official project page](https://www.pkuvmc.com/publications/msmt17.html) and follow its access terms. Extract it so these paths exist:

```text
datasets/msmt17/MSMT17_V1/list_train.txt
datasets/msmt17/MSMT17_V1/train/
datasets/msmt17/MSMT17_V1/test/
```

Start a new OSNet x0.25 run initialized from ImageNet weights. The configuration uses 256x128 person images, random flip and random erase, Adam, batch size 32, and triplet plus identity-classification losses for 60 epochs.

```bash
./scripts/train_osnet_msmt17.sh
```

The checkpoint is written under `logs/osnet_msmt17_supervised/model/`. Training data and generated checkpoints are ignored by Git. To evaluate the locally trained checkpoint instead of the downloaded model-zoo checkpoint, set `OSNET_WEIGHTS` to its path as shown below.

## Video inference and validation

For an arbitrary video, run the pipeline directly:

```bash
python scripts/run_video_reid.py \
  --video /path/to/video.mp4 \
  --output runs/video_reid.jsonl \
  --detector weights/yolo26n.pt \
  --mode rerank \
  --frame-skip 1 \
  --vlm-model qwen2.5vl:7b \
  --osnet-weights weights/osnet_x0_25_msmt17.pth
```

For the standard MOT20 comparison, download MOT20 and provide the TrackEval ground-truth root. It must contain `MOT20-train/MOT20-01/seqinfo.ini` and `MOT20-train/MOT20-01/gt/gt.txt`.

```bash
./scripts/validate_mot20.sh /path/to/MOT20-01.mp4 /path/to/trackeval/gt MOT20-01
```

The script runs both temporal OSNet and the VLM gate, exports each JSONL result to MOT format, sanitizes duplicate same-frame IDs without using ground truth, and runs TrackEval HOTA, CLEAR, and Identity metrics. To use a newly trained OSNet checkpoint:

```bash
OSNET_WEIGHTS=logs/osnet_msmt17_supervised/model/model.pth.tar-60 \
  ./scripts/validate_mot20.sh /path/to/MOT20-01.mp4 /path/to/trackeval/gt MOT20-01
```

Run the unit tests with:

```bash
python -m pytest -q tests
```

## References

- [OSNet paper](https://arxiv.org/abs/1905.00953)
- [MSMT17 paper](https://arxiv.org/abs/1711.08565)
- [MOT20 benchmark paper](https://arxiv.org/abs/2003.09003)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/)
