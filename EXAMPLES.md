# MarsBench Usage Examples

Focused, comprehensive examples for running MarsBench with Hugging Face–hosted datasets and (optionally) locally downloaded data (from Zenodo).

Sections:
1. Base Example
2. Command Builder
3. Classification Tutorial
4. Segmentation Tutorial
5. Detection Tutorial
6. Callbacks (and Logging / Monitoring)
7. Advanced Patterns
8. Batch Runs

---

### Reference Table: Datasets (internal name vs Hugging Face repo slug) and Available Models

NOTE:
- data_name is the value you pass via data_name=...
- HF Repo Slug column shows the suffix part used in repo_id="ORG/<slug>" (Organization/user prefix e.g. Mirali33 may differ)
- Model Name column lists all registered models for that task 
- If any slug differs in your local version, open marsbench/data/__init__.py for the authoritative mapping

| Task Type | Dataset Name (data_name param) | HF Repo Slug (example) | Model Names (model_name) |
|-----------|--------------------------------|------------------------|---------------------------|
| classification | domars16k | mb-domars16k | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| classification | atmospheric_dust_classification_edr | mb-atmospheric_dust_cls_edr | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| classification | atmospheric_dust_classification_rdr | mb-atmospheric_dust_cls_rdr | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| classification | change_classification_ctx | mb-change_cls_ctx | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| classification | change_classification_hirise | mb-change_cls_hirise | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| classification | frost_classification | mb-frost_cls | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| classification | landmark_classification | mb-landmark_cls | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| classification | surface_classification | mb-surface_cls | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| classification | multi_label_mer | mb-surface_multi_label_cls | resnet101, vit, swin_transformer, inceptionv3, squeezenet |
| segmentation | boulder_segmentation | mb-boulder_seg | unet, deeplab, dpt, mask_rcnn, mask2former, segformer |
| segmentation | conequest_segmentation | mb-conequest_seg | unet, deeplab, dpt, mask_rcnn, mask2former, segformer |
| segmentation | crater_binary_segmentation | mb-crater_binary_seg | unet, deeplab, dpt, mask_rcnn, mask2former, segformer |
| segmentation | crater_multi_segmentation | mb-crater_multi_seg | unet, deeplab, dpt, mask_rcnn, mask2former, segformer |
| segmentation | mmls | mb-mmls | unet, deeplab, dpt, mask_rcnn, mask2former, segformer |
| segmentation | mars_seg_mer | mb-mars_seg_mer | unet, deeplab, dpt, mask_rcnn, mask2former, segformer |
| segmentation | mars_seg_msl | mb-mars_seg_msl | unet, deeplab, dpt, mask_rcnn, mask2former, segformer |
| segmentation | s5mars | mb-s5mars | unet, deeplab, dpt, mask_rcnn, mask2former, segformer |
| detection | boulder_detection | mb-boulder_det | fasterrcnn, retinanet, ssd |
| detection | conequest_detection | mb-conequest_det* | fasterrcnn, retinanet, ssd |
| detection | dust_devil_detection | mb-dust_devil_det | fasterrcnn, retinanet, ssd |

---

## 1. Base Example

A minimal training run on a Hugging Face dataset:

```bash
python -m marsbench.main \
  mode=train \
  task=classification \
  model_name=resnet101 \
  data_name=domars16k \
  load_from_hf=true \
  repo_id="Mirali33/mb-domars16k" \
  training.trainer.max_epochs=1
```

Key points:
- mode=train (default if omitted)
- load_from_hf=true + repo_id="..." tells MarsBench to fetch the dataset via Hugging Face Hub.
- training.trainer.max_epochs=1 keeps it fast for a smoke test.

(If the dataset is already cached by datasets library, it will reuse it.)

---

## 2. Command Builder

MarsBench uses Hydra. Every CLI override is key=value. Nested config segments (e.g., training.trainer.max_epochs) descend into YAML structure.

General template:

```bash
python -m marsbench.main \
  mode=<train|test|predict> \
  task=<classification|segmentation|detection> \
  model_name=<registered_model_key> \
  data_name=<registered_dataset_key> \
  [load_from_hf=true repo_id="HF_ORG/REPO"] \
  [dataset_path=/absolute/or/relative/path] \
  [checkpoint_path=path/to/checkpoint.ckpt] \
  [output_path=custom/output/dir] \
  [prediction_output_path=preds/out] \
  training.trainer.max_epochs=E \
  training.batch_size=B \
  training.optimizer.lr=LR \
  transforms=<transforms_config_name> \
  callbacks.early_stopping.patience=10 \
  logger.wandb.enabled=true
```

Parameter sources:
- model_name: must exist in configs/model/<task>/
- data_name: must exist in configs/data/<task>/
- transforms: defined in configs/transforms/
- logger.*, callbacks.*, training.* are in respective config trees.
- Use +key=value to introduce keys not predefined (Hydra strict mode safeguard).

Quoting:
- Quote strings containing special characters or uppercase letters when in doubt: repo_id="Mirali33/mb-boulder_det"

Hydra multirun (launch several sweeps):
```bash
python -m marsbench.main -m \
  task=classification model_name=resnet101 data_name=domars16k load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  training.optimizer.lr=0.001,0.0005 training.batch_size=32,64
```
Outputs go into multirun/ timestamped folders.

---

## 3. Classification Tutorial

Step-by-step from zero to evaluation.

### 3.1 Choose a Dataset (Hugging Face)

Check mapping in marsbench/data/__init__.py (datasets like domars16k, atmospheric_dust_classification_edr, surface_classification, etc.). Example using atmospheric dust:

```bash
python -m marsbench.main \
  mode=train \
  task=classification \
  model_name=resnet101 \
  data_name=atmospheric_dust_classification_edr \
  load_from_hf=true \
  repo_id="Mirali33/mb-atmospheric_dust_cls_edr" \
  training.trainer.max_epochs=3 \
  training.batch_size=32
```

### 3.2 Using Local Data (Zenodo Download)

1. Download the archive (e.g., domars16k.zip) from Zenodo
2. Unzip: `unzip domars16k.zip -d /data/mars/`
3. Ensure folder structure matches the expected pipeline
4. Run without load_from_hf:
```bash
python -m marsbench.main \
  task=classification \
  model_name=resnet101 \
  data_name=domars16k \
  dataset_path=/data/mars/domars16k \
  training.trainer.max_epochs=2
```

NOTE: Do not keep .zip files compressed; the dataloader expects extracted directories.

### 3.3 Adjust Training Hyperparameters

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  training.optimizer.name=AdamW \
  training.optimizer.lr=0.0007 \
  training.scheduler.name=cosine \
  training.trainer.max_epochs=10 \
  training.trainer.accumulate_grad_batches=2 \
  training.trainer.precision=16
```

### 3.4 Validation Limits / Quick Debug

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  training.trainer.fast_dev_run=1
```

OR subset:
```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  +data.subset=1000
```

### 3.5 Testing After Training

Either add:
```bash
... test_after_training=true
```
Or run separately:
```bash
python -m marsbench.main \
  mode=test \
  task=classification \
  model_name=resnet101 \
  data_name=domars16k \
  checkpoint_path=outputs/classification/domars16k/resnet101/<RUN_ID>/checkpoints/best.ckpt
```

### 3.6 Generating Predictions

```bash
python -m marsbench.main \
  mode=predict \
  task=classification \
  model_name=resnet101 \
  data_name=domars16k \
  checkpoint_path=outputs/classification/domars16k/resnet101/<RUN_ID>/checkpoints/best.ckpt \
  prediction_output_path=predictions/domars16k_resnet101
```

### 3.7 Multi-GPU (if available)

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  training.trainer.accelerator=gpu \
  training.trainer.devices=2 \
  training.trainer.strategy=ddp
```

---

## 4. Segmentation

Example dataset names: boulder_segmentation, conequest_segmentation, crater_binary_segmentation, crater_multi_segmentation, mars_seg_mer, mars_seg_msl, s5mars, mmls 

### 4.1 Basic U-Net on ConeQuest (Hugging Face)

```bash
python -m marsbench.main \
  task=segmentation \
  model_name=unet \
  data_name=conequest_segmentation \
  load_from_hf=true \
  repo_id="Mirali33/mb-conequest_seg" \
  training.trainer.max_epochs=5
```

### 4.2 Partial Dataset Partition (Fast Experiment)

If config exposes partition (e.g., partition=0.1 used earlier):

```bash
python -m marsbench.main \
  task=segmentation model_name=unet data_name=boulder_segmentation \
  load_from_hf=true repo_id="Mirali33/mb-boulder_seg" \
  partition=0.1 \
  training.trainer.max_epochs=2
```

### 4.3 Changing Transforms

Check configs/transforms/ for segmentation-specific transforms (e.g., seg_default, heavy_aug). Example:

```bash
python -m marsbench.main \
  task=segmentation model_name=unet data_name=cone_quest_segmentation \
  load_from_hf=true repo_id="Mirali33/mb-conequest_seg" \
  transforms=seg_default \
  training.trainer.max_epochs=10
```

### 4.4 Switch Model (DeepLab)

```bash
python -m marsbench.main \
  task=segmentation \
  model_name=deeplab \
  data_name=cone_quest_segmentation \
  load_from_hf=true repo_id="Mirali33/mb-conequest_seg" \
  training.optimizer.lr=0.0001 \
  training.trainer.max_epochs=15
```

### 4.5 Mixed Precision + Gradient Accumulation

```bash
python -m marsbench.main \
  task=segmentation model_name=unet data_name=boulder_segmentation \
  load_from_hf=true repo_id="Mirali33/mb-boulder_seg" \
  training.trainer.precision=16 \
  training.trainer.accumulate_grad_batches=4 \
  training.trainer.max_epochs=20
```

---

## 5. Detection

Dataset keys may include boulder_detection, conequest_detection, dust_devil_detection.

### 5.1 SSD on Boulder Detection (Hugging Face)

```bash
python -m marsbench.main \
  task=detection \
  model_name=ssd \
  data_name=boulder_detection \
  load_from_hf=true \
  repo_id="Mirali33/mb-boulder_det" \
  training.trainer.max_epochs=5
```

### 5.2 Faster R-CNN with Learning Rate Override

```bash
python -m marsbench.main \
  task=detection \
  model_name=faster_rcnn \
  data_name=boulder_detection \
  load_from_hf=true repo_id="Mirali33/mb-boulder_det" \
  training.optimizer.lr=0.0002 \
  training.trainer.max_epochs=12
```

### 5.3 Evaluation Only

```bash
python -m marsbench.main \
  mode=test \
  task=detection \
  model_name=ssd \
  data_name=boulder_detection \
  checkpoint_path=outputs/detection/boulder_detection/ssd/<RUN_ID>/checkpoints/best.ckpt
```

---

## 6. Callbacks (and Logging)

MarsBench integrates PyTorch Lightning callbacks & loggers via configs/callbacks and configs/logger.

Common callbacks keys (exact keys depend on YAML):
- callbacks.early_stopping.*
- callbacks.best_checkpoint.* (model checkpoint)
- callbacks.lr_monitor.enabled
- callbacks.progress_bar.refresh_rate

### 6.1 Early Stopping

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  callbacks.early_stopping.monitor=val/accuracy \
  callbacks.early_stopping.mode=max \
  callbacks.early_stopping.patience=8
```

### 6.2 Multiple Best Checkpoints

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  callbacks.best_checkpoint.save_top_k=3 \
  callbacks.best_checkpoint.monitor=val/accuracy \
  callbacks.best_checkpoint.mode=max
```

### 6.3 Learning Rate Monitor

```bash
python -m marsbench.main \
  task=segmentation model_name=unet data_name=cone_quest_segmentation \
  load_from_hf=true repo_id="Mirali33/mb-conequest_seg" \
  callbacks.lr_monitor.enabled=true
```

### 6.4 WandB Logging

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  logger.wandb.enabled=true \
  logger.wandb.project=MarsBench \
  logger.wandb.name=domars16k_resnet101_test
```

### 6.5 TensorBoard + CSV

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  logger.tensorboard.enabled=true \
  logger.csv.enabled=true
```

### 6.6 Custom Output Path

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  output_path=experiments/domars16k_resnet101_custom
```

---

## 7. Advanced Patterns

### 7.1 Hydra Sweep (Models x LRs)

```bash
python -m marsbench.main -m \
  task=classification data_name=domars16k load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  model_name=resnet101,vit \
  training.optimizer.lr=0.001,0.0005
```

### 7.2 Reproducibility

```bash
python -m marsbench.main \
  task=classification model_name=resnet101 data_name=domars16k \
  load_from_hf=true repo_id="Mirali33/mb-domars16k" \
  seed=42
```

---

## Local vs Hugging Face Summary

| Scenario | Required Args |
|----------|---------------|
| Hugging Face dataset | load_from_hf=true repo_id="ORG/REPO" |
| Local extracted folder | dataset_path=/path/to/folder (omit load_from_hf) |
| Zenodo zip just downloaded | MUST unzip first; then dataset_path=<unzipped_root> |

If both load_from_hf and dataset_path are given, precedence depends on dataset_path

---

## Troubleshooting Notes

- ModuleNotFound: run `pip install -e .` from repo root.
- CUDA OOM: reduce training.batch_size or use training.trainer.precision=16.
- Slow startup: first HF dataset download; subsequent runs cached.
- Permission issues on output: ensure outputs/ is writable or set output_path.

---

## 8. Batch Runs (SLURM Array on HPC)

Purpose:
Run many independent MarsBench configurations in parallel on an HPC using a SLURM job array. Each array index runs exactly one configuration (no Hydra -m multirun inside). This keeps logging isolated and makes failed reruns trivial.

Key ideas:
- Encode the Cartesian product of parameter lists into a COMBOS array.
- Use SLURM_ARRAY_TASK_ID to select one combo.
- Write one log (stdout+stderr) per array element via #SBATCH --output / --error.
- Trap errors to record failing indices.

### 8.1 Basic Classification Sweep Script

Save as scripts/sweep_classification.sbatch (make executable: chmod +x scripts/sweep_classification.sbatch).

```bash
#!/bin/bash
#SBATCH --job-name=marsbench_hf
#SBATCH --array=0-10             # TEMP placeholder; will be reset after combo count is known
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p general
#SBATCH -q public
#SBATCH -A grp_hkerner
#SBATCH --mem=16G
#SBATCH --output=./outputs/hpc/slurm-%A_%a.out
#SBATCH --error=./outputs/hpc/slurm-%A_%a.out

set -euo pipefail

# --- User environment (EDIT) ---
module load mamba
source activate kerner_lab

mkdir -p outputs/hpc

# Parameter lists (EDIT freely)
DATASETS=(domars16k atmospheric_dust_classification_edr atmospheric_dust_classification_rdr change_classification_ctx change_classification_hirise frost_classification landmark_classification surface_classification)
MODELS=(resnet101 vit swin_transformer inceptionv3 squeezenet)
HF_FLAG_VAL=(True False)
BATCH_SIZES=(32)
SEEDS=(42)

# Build combos (Cartesian product)
COMBOS=()
for d in "${DATASETS[@]}"; do
  for m in "${MODELS[@]}"; do
    for hf in "${HF_FLAG_VAL[@]}"; do
      for bs in "${BATCH_SIZES[@]}"; do
        for seed in "${SEEDS[@]}"; do
          COMBOS+=("${m},${d},${hf},${bs},${seed}")
        done
      done
    done
  done
done

TOTAL=${#COMBOS[@]}

# Optional: Add print-total helper
if [[ "${1:-}" == "--print-total" ]]; then
  echo "Total combinations: ${#COMBOS[@]}"
  exit 0
fi

# Extract combo
IFS=',' read MODEL DATASET HF_FLAG BS SEED <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

RUN_TAG="cls_${DATASET}_${MODEL}_bs${BS}_seed${SEED}_hf${HF_FLAG}"
FAILED_LOG="outputs/hpc/failed_runs.txt"

# On failure, append a single line with index and params
trap 'echo "idx=${SLURM_ARRAY_TASK_ID} model=${MODEL} dataset=${DATASET} bs=${BS} seed=${SEED} hf=${HF_FLAG} " >> "${FAILED_LOG}"' ERR

echo "=== Starting ${RUN_TAG} (array index ${SLURM_ARRAY_TASK_ID}/${TOTAL}) ==="

python -m marsbench.main \
  task=classification \
  model_name="${MODEL}" \
  data_name="${DATASET}" \
  load_from_hf=${HF_FLAG} \
  training.batch_size="${BS}" \
  seed="${SEED}" \
  training.trainer.max_epochs=1 \
  logger.csv.enabled=true \
  logger.tensorboard.enabled=false

echo "=== Finished ${RUN_TAG} ==="

```
(Or temporarily add: echo "TOTAL=$TOTAL"; exit 0 just after computing TOTAL.) Then set --array=0-(TOTAL-1).


### 8.2 Rerunning Failed Indices

After job finishes:
```bash
sort -u status/failed_indices.txt > status/failed_indices.unique.txt
FAILED=$(paste -sd, status/failed_indices.unique.txt)
if [[ -n "$FAILED" ]]; then
  sbatch --array=${FAILED} scripts/sweep_classification.sbatch
fi
```

### 8.3 Segmentation Sweep Variant (Example Changes Only)

Key differences:
- task=segmentation
- data_name=cone_quest_segmentation
- models subset (unet deeplab)
- maybe partition=0.2 for quick tuning

Command section replacement:
```bash
python -m marsbench.main \
  task=segmentation \
  model_name="${MODEL}" \
  data_name=cone_quest_segmentation \
  load_from_hf=true \
  repo_id="Mirali33/mb-conequest_seg" \
  partition=0.2 \
  training.optimizer.lr="${LR}" \
  training.batch_size="${BS}" \
  seed="${SEED}" \
  training.trainer.max_epochs=20 \
  output_path="${OUT_DIR}" \
  callbacks.best_checkpoint.save_top_k=1
```

---
## Minimal Cheatsheet

Classification (HF):
```bash
python -m marsbench.main task=classification model_name=resnet101 data_name=domars16k load_from_hf=true repo_id="Mirali33/mb-domars16k"
```

Segmentation (HF):
```bash
python -m marsbench.main task=segmentation model_name=unet data_name=conequest_segmentation load_from_hf=true repo_id="Mirali33/mb-conequest_seg"
```

Detection (HF):
```bash
python -m marsbench.main task=detection model_name=ssd data_name=boulder_detection load_from_hf=true repo_id="Mirali33/mb-boulder_det"
```

Local (after unzip):
```bash
python -m marsbench.main task=classification model_name=resnet101 data_name=domars16k dataset_path=/data/mars/domars16k
```

Add early stopping:
```bash
... callbacks.early_stopping.patience=5 callbacks.early_stopping.monitor=val/accuracy callbacks.early_stopping.mode=max
```

Enable WandB:
```bash
... logger.wandb.enabled=true logger.wandb.project=MarsBench
```
