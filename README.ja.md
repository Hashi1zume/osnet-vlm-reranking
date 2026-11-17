# 動画人物再同定：YOLO26 + OSNet + Qwen2.5-VL

[English README](README.md)

人物が画面から消えた後に再登場したとき、IDの切り替わりを減らせるか評価する研究リポジトリです。YOLO26nが人物を検出し、OSNetが過去の人物画像から候補を探し、Qwen2.5-VLが候補との画像照合を行います。

## プロジェクト概要

人物cropをOSNetへ入力して512次元の特徴を取り出し、コサイン類似度で過去の候補を最大10件順位付けします。Qwen2.5-VLは現在画像と各候補画像を比較します。VLMが同一人物と判断し、信頼度0.8以上、かつOSNetの生コサイン類似度0.75以上の候補だけ、過去の人物IDを引き継ぎます。条件を満たす候補がなければ、新しいIDを割り当てます。

MOT20-01先頭214フレームでの評価結果:

| 指標 | 時系列OSNet | OSNet + Qwen2.5-VLゲート |
|---|---:|---:|
| HOTA | 14.889 | 16.576 |
| AssA | 52.804 | 65.507 |
| IDF1 | 7.700 | 9.356 |
| ID switches | 94 | 0 |

これは短い1シーケンスでの結果で、検出再現率も約4.9%でした。この結果から言えるのは、このシーケンスで人物IDの対応が改善したことまでです。MSMT17学習済みOSNetは教師あり学習済みの比較モデルで、VLMを使ってOSNet自体を学習したものではありません。Qwen2.5-VLは推論時の画像比較に使い、追加学習していません。

## リポジトリ構成

- `reid_pipeline/`: OSNet候補検索、VLM比較、時系列ID割当、評価処理。
- `scripts/`: 動画推論、重み取得、OSNet学習、TrackEval実行。
- `configs/`: MSMT17用OSNet学習設定。
- `tests/`: プロジェクトコードの単体テスト。
- `patches/`: 固定したTrackEval版に対するNumPy 2互換修正。

Drive動画、データセット、モデル重み、生成動画、スライド、研究報告書は含めていません。`.gitignore` でもこれらのローカル成果物を除外しています。MSMT17とMOT20は各配布元から別途取得してください。

## 環境構築

確認済み環境はPython 3.12、PyTorch 2.15.0 + CUDA 13.2、TorchVision 0.29.2 + CUDA 13.2、Ultralytics 8.4.213です。PyTorch/CUDAはOSやGPUドライバーに合わせる必要があるため、[PyTorch公式インストール選択ページ](https://pytorch.org/get-started/locally/)から対応する組み合わせを選んでください。

```bash
conda env create -f environment.yml
conda activate yolo26-osnet-vlm

# pytorch.org でこのPCに合うCUDA版またはCPU版を選んでインストール
python -m pip install torch torchvision

python -m pip install -r requirements.txt
./scripts/setup_dependencies.sh
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`setup_dependencies.sh` は[Torchreid](https://github.com/KaiyangZhou/deep-person-reid)と[TrackEval](https://github.com/JonathonLuiten/TrackEval)の固定リビジョンを取得します。NumPy 2で削除された型別名に対する互換パッチもTrackEvalへ適用します。

[Ollama](https://ollama.com/download)をインストールして起動し、VLMを取得します。

```bash
ollama pull qwen2.5vl:7b
ollama serve
```

別ターミナルで検出器とMSMT17学習済みOSNetの重みを取得します。

```bash
./weights/download_weights.sh
```

重みはGit管理対象外の`weights/`に保存します。YOLO26nは[Ultralytics公式配布](https://github.com/ultralytics/assets/releases/tag/v8.4.0)、OSNetは[TorchreidのMSMT17モデル](https://github.com/KaiyangZhou/deep-person-reid/blob/master/docs/MODEL_ZOO.md)を取得します。

## OSNetの学習

[MSMT17公式ページ](https://www.pkuvmc.com/publications/msmt17.html)からデータセットの利用手順に従って取得・展開し、以下の配置にします。

```text
datasets/msmt17/MSMT17_V1/list_train.txt
datasets/msmt17/MSMT17_V1/train/
datasets/msmt17/MSMT17_V1/test/
```

ImageNet重みから初期化したOSNet x0.25を学習します。入力は256×128画素、横反転とランダム消去を使用し、Adam、batch size 32、triplet lossと人物ID分類lossで60 epoch学習します。

```bash
./scripts/train_osnet_msmt17.sh
```

学習checkpointは`logs/osnet_msmt17_supervised/model/`に保存され、Gitには含まれません。

## 動画推論と検証

任意の動画に推論する例:

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

標準評価ではMOT20を取得し、TrackEvalの正解データrootを指定します。rootには`MOT20-train/MOT20-01/seqinfo.ini`と`MOT20-train/MOT20-01/gt/gt.txt`が必要です。

```bash
./scripts/validate_mot20.sh /path/to/MOT20-01.mp4 /path/to/trackeval/gt MOT20-01
```

OSNetのみとVLMゲートの両方を実行し、JSONLからMOT形式へ変換後、同一フレーム内の重複IDを正解ラベルを使わず分割して、TrackEvalのHOTA、CLEAR、Identity指標を計算します。

自分で学習した重みを使う場合:

```bash
OSNET_WEIGHTS=logs/osnet_msmt17_supervised/model/model.pth.tar-60 \
  ./scripts/validate_mot20.sh /path/to/MOT20-01.mp4 /path/to/trackeval/gt MOT20-01
```

単体テスト:

```bash
python -m pytest -q tests
```

## 参考資料

- [OSNet論文](https://arxiv.org/abs/1905.00953)
- [MSMT17論文](https://arxiv.org/abs/1711.08565)
- [MOT20論文](https://arxiv.org/abs/2003.09003)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [Ultralytics YOLO26ドキュメント](https://docs.ultralytics.com/models/yolo26/)
