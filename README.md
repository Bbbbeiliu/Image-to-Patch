# Image to Patch

**Description:** Source code for scar segmentation, model training and evaluation, statistical analysis, marker calibration, contour processing, nesting, and DXF export.

Core research code accompanying **From Image to Patch: An Automated Framework for Precise and Efficient Radionuclide Scar Therapy Preparation**.

```text
Supplementary_Code/
|-- segmentation/       Proposed model and training
|-- benchmark/          Baselines, evaluation and statistics
|-- experiments/        Experiment runners and split assignments
|-- analysis/           Diagnostics and measurement
|-- fabrication/        Calibration, contours, nesting and DXF
|-- tools/              Input and dependency preparation
|-- notebooks/          Optional launch examples
`-- README.md
```

## Download code and best checkpoint

Download the source using **Code > Download ZIP**, extract it, and open a terminal
in the extracted repository root (the directory containing this README).
Alternatively:

```bash
git clone https://github.com/Bbbbeiliu/Image-to-Patch.git
cd Image-to-Patch
```

The best checkpoint is available from the [v1.0.0 release](https://github.com/Bbbbeiliu/Image-to-Patch/releases/tag/v1.0.0).
Download `best.pth`, create a `weights` directory in the repository root, and place
the file at `weights/best.pth`. For example, in PowerShell:

```powershell
New-Item -ItemType Directory -Force weights | Out-Null
Invoke-WebRequest -Uri https://github.com/Bbbbeiliu/Image-to-Patch/releases/download/v1.0.0/best.pth -OutFile weights/best.pth
Get-FileHash -Algorithm SHA256 weights/best.pth
```

Expected checkpoint SHA-256:
`f3f4700c695cd408937461961f173980f9c8b4dbae689fb30410cfd600cb1ba8`.
The checkpoint is a raw FP32 model state dictionary. No support images or
inference-time prototype are required. Clinical-data-dependent commands still
require authorized images and annotations, as described below.

## Contents

| Directory | Purpose |
|---|---|
| `segmentation/` | ResNet34 U-Net, four attention blocks, external prototype construction, BCE + Dice loss, preprocessing and original training loop |
| `benchmark/` | Seven baseline adapters, resumable training, per-image evaluation, boundary metrics, grouped bootstrap, paired Wilcoxon tests and plots |
| `experiments/` | Baseline and ablation entry points, inference efficiency and repeated nesting |
| `analysis/` | Otsu plus morphology, error overlays, ROC/PR, occlusion sensitivity, cutting measurements and geometric coverage sensitivity |
| `fabrication/` | Contour extraction, fitting, marker calibration, nesting, label placement and offline DXF export |
| `experiments/splits/` | Pseudonymous patient-level assignments for the two evaluated partitions |
| `notebooks/` | Training, evaluation and ablation launch examples |
| `tools/` | Data preparation, pinned upstream-source retrieval, initialization weights and package checks |

Clinical photographs, annotations, baseline checkpoints, software environments and hardware drivers are not included in the source repository. The proposed-model best checkpoint is distributed separately as the `best.pth` release asset. The clinical data remain subject to institutional access arrangements. Public support data are available from the [ISIC 2018 challenge](https://challenge.isic-archive.com/data/).

## Environment

The segmentation experiments used Python 3.12, PyTorch 2.9.0 with CUDA 12.8, torchvision 0.24.0, segmentation-models-pytorch 0.5.0 and Albumentations 2.0.8. Install a PyTorch build appropriate for your hardware, followed by:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_fabrication.txt
python tools/check_package.py
```

Use a fresh environment and one OpenCV wheel. The package requirements select the headless OpenCV wheel. The Qt geometry classes used in fabrication are included through PyQt5.

## Quick start

Run all commands from this directory. Detailed commands are provided below.

```bash
# Show available options. These commands do not start training.
python segmentation/train.py --help
python experiments/run_baseline.py --help
python experiments/train_ablation.py --help

# A geometry-only example with no clinical inputs or trained weights.
python tools/make_synthetic_examples.py --out examples/generated
python analysis/coverage_sensitivity.py --input examples/generated/contours --out runs/example_coverage
python fabrication/run_pipeline.py --mask examples/generated/mask.png --mm-per-pixel 0.1 --out runs/example_fabrication
```

## Core protocol

Clinical inputs are RGB, divided by 255 and proportionally resized with centered zero padding to 256 × 256. Validation and test transforms are deterministic. The proposed model uses four attention blocks and a public lesion-foreground prototype refreshed every five epochs. The auxiliary weight is 0.5. The main loss is BCE plus soft Dice loss with coefficient 1.0. Adam uses learning rate 0.0001 and batch size 4. Training is limited to 100 epochs, with early stopping after five consecutive validations without a new minimum loss.

The primary partition contains 1,189/255/255 images and 417/90/94 patient groups. The additional partition is supplied separately in `experiments/splits/seed43.csv`. All images of each patient remain in one subset. The assignments contain pseudonyms only.

Per-image evaluation saves confusion counts, foreground/background scores, probabilities, predicted masks, reference masks, geometry and boundary metrics. Dice and IoU use a probability threshold greater than 0.5, with padding excluded. HD95 and ASSD are measured in network-input pixels. Statistical scripts use 10,000 grouped bootstrap resamples and optional two-sided Wilcoxon tests with Holm correction.

## External model implementations and weights

Adapters are included. Official TransUNet, PANet and SAMed implementations are retrieved separately at the exact commits listed in `external_sources.json`:

```bash
python tools/fetch_external_sources.py --models TransUNet PANet SAMed
python tools/prepare_weights.py --models resnet34 vgg16 transunet sam
```

These commands explicitly download external source code or initialization weights. No download or training occurs merely by opening the archive. Third-party code remains subject to its upstream license. Checkpoint-dependent commands require separately supplied weights. Place the proposed-model raw FP32 state dictionary at `weights/best.pth` before running those commands, or replace that path with your own trained checkpoint. The study checkpoint has 45,723,234 parameters and SHA-256 `f3f4700c695cd408937461961f173980f9c8b4dbae689fb30410cfd600cb1ba8`. No inference-time public prototype or support images are required. Use the `retained` evaluation option after preparing the clinical data and dataset lock.

## Source and packaging

Packaging changes include portable paths, explicit entry points, English runtime messages and omission of comments, unused example code and unused splitting helpers. Original algorithm expressions are retained in the copied core modules. The primary training loop and the ablation training implementation are supplied separately. Dataset locks must be generated for this packaged source, because source checksums are part of the lock.

The offline DXF exporter retains the application's display-coordinate factor `1/2.3`. Its coordinate settings are exposed in the example output summary. Laser-controller SDKs, cloud credentials and machine-execution code are outside this offline research package.

Please cite the accompanying article and the upstream methods used in your experiments.

# Running the supplementary code

All paths below are relative to the repository root. Replace example paths with your authorized local inputs. Output folders must be new unless an entry point explicitly supports `--resume`.

## 1. Prepare the clinical and public data

Use matching image and Labelme JSON filenames such as `0001.jpg` and `0001.json`. The split CSV contains `sample_id,patient_id,split` and does not contain names or hospital identifiers.

```bash
python tools/prepare_data.py --images /path/to/images --masks /path/to/annotations --split-csv experiments/splits/seed42.csv --out data/clinical
python benchmark/benchmark.py lock --data data/clinical --out benchmark/dataset_lock.json
```

Expected layout:

```text
data/
  clinical/
    train/images/    train/masks/
    val/images/      val/masks/
    test/images/     test/masks/
    _audit/manifest.json
    _audit/summary.json
    _audit/train.txt  val.txt  test.txt
  public/
    images/<support_id>.jpg
    masks/<support_id>_segmentation.png
```

Use all 3,694 paired ISIC 2018 Task 1 support images and masks for the prototype configurations. Public images are not part of the clinical test set. Preserve the fixed clinical partition when comparing models. Generate a new lock after preparing a distinct partition or changing packaged source files.

For the additional partition:

```bash
python tools/prepare_data.py --images /path/to/images --masks /path/to/annotations --split-csv experiments/splits/seed43.csv --out data/clinical_split43
python benchmark/benchmark.py lock --data data/clinical_split43 --out benchmark/dataset_lock43.json
```

## 2. Fetch optional upstream implementations and initialization

```bash
python tools/fetch_external_sources.py --models TransUNet PANet SAMed
python tools/prepare_weights.py --models resnet34 vgg16 transunet sam
```

Git and network access are required for the first command. Weights are downloaded separately and checked against recorded SHA256 values. SMP uses the same local `benchmark/torch_cache` directory for its ResNet34 initialization.

## 3. Train the proposed model

```bash
python segmentation/train.py --data data/clinical --public-data data/public --out runs/proposed --device cuda
```

This exposes the original training loop and its model/prototype/loss modules. The output contains `best.pth`, `last.pth`, `prototype.pt`, configuration and training logs. An optional `--seed` specifies a seed for a new run. The canonical loop saves weights, while the separate baseline and ablation engines retain full resume state.

After supplying `weights/best.pth` or replacing its path with a trained checkpoint, evaluate the proposed model through the common exporter. This wrapper always saves per-image probabilities for subsequent diagnostics:

```bash
python experiments/run_baseline.py evaluate --model retained --checkpoint weights/best.pth --split val --out runs/proposed_val
python experiments/run_baseline.py evaluate --model retained --checkpoint weights/best.pth --split test --out runs/proposed_test
```

## 4. Train and evaluate a baseline

```bash
python experiments/run_baseline.py train --model unet --out runs/unet
python experiments/run_baseline.py train --model unetpp --out runs/unetpp
python experiments/run_baseline.py train --model deeplabv3plus --out runs/deeplabv3plus
python experiments/run_baseline.py train --model transunet --out runs/transunet
python experiments/run_baseline.py train --model manet --out runs/manet
python experiments/run_baseline.py train --model panet --out runs/panet
python experiments/run_baseline.py train --model samed --out runs/samed
python experiments/run_baseline.py evaluate --model unet --checkpoint runs/unet/best.pt --split test --out runs/unet/test
```

U-Net uses random initialization. U-Net++, DeepLabV3+ and MAnet use ImageNet ResNet34 initialization by default during training. TransUNet uses the official R50+ViT-B/16 checkpoint. PANet uses ImageNet VGG16 features and a fixed train-only support during validation/testing. SAMed uses the official SAM ViT-B initialization with rank-4 LoRA and a binary output head.

Resume a baseline by repeating its training command with `--resume`. Optional `--show-plot` displays training progress. To evaluate a new partition, pass its `--data` and `--lock` to every training and evaluation command.

## 5. Ablations

```bash
python experiments/train_ablation.py train --variant backbone --out runs/ablation_backbone
python experiments/train_ablation.py train --variant attention_only --out runs/ablation_attention
python experiments/train_ablation.py train --variant prototype_only --out runs/ablation_prototype
python experiments/train_ablation.py train --variant full_no_augmentation --out runs/ablation_no_augmentation
python experiments/train_ablation.py evaluate --variant backbone --checkpoint runs/ablation_backbone/best.pt --out runs/ablation_backbone/test --probabilities
```

The `full` variant is also available for a newly trained full-model control. A completed full-model checkpoint can instead be evaluated directly, without retraining it. The ablation entry point uses training seed 42, batch size 4, Adam at 0.0001, a 100-epoch maximum and patience 5. Use `--resume` to resume the same ablation run.

## 6. Statistics and diagnostic plots

```bash
python benchmark/compare_extended.py --reference runs/proposed_test --baselines runs/unet/test runs/samed/test --out runs/paired_comparison --resamples 10000 --seed 20261105 --wilcoxon
python benchmark/analyze_saved.py --evaluation runs/proposed_test --out runs/proposed_analysis --resamples 10000
python analysis/diagnose_predictions.py --evaluation runs/proposed_test --out runs/proposed_diagnostics
python analysis/occlusion_sensitivity.py --images data/clinical/test/images --masks data/clinical/test/masks --sample-id 0538 --checkpoint weights/best.pth --out runs/occlusion
```

Pass all seven baseline evaluation folders in one paired-comparison command to apply Holm correction to the complete 14-comparison family. The figures include paired Dice/IoU scatter and difference plots. The diagnostic entry point uses 256 equal-width probability bins by default and selects the three lowest-Dice images and one median-Dice image for overlays.

Otsu selection requires the same model's saved validation and test image exports under a common parent:

```bash
python analysis/run_otsu.py --source runs/reference_exports --out runs/otsu
```

## 7. Inference efficiency

Create a JSON file mapping model names to checkpoint paths. Use `proposed` for the proposed model, for example:

```json
{"proposed":"weights/best.pth","unet":"runs/unet/best.pt","samed":"runs/samed/best.pt"}
```

```bash
python experiments/measure_efficiency.py --checkpoints checkpoints.json --out runs/efficiency
```

Defaults are FP32, 256 × 256, batch size 1, three randomized-order rounds, 50 warm-up calls and 100 timed calls per round. CUDA calls are synchronized. Processing rate is calculated from the mean time, while median latency is reported separately. PANet's fixed evaluation support is preloaded on the device.

## 8. Offline contour processing

```bash
python fabrication/calibrate_marker.py --image /path/to/image.png --diameter-mm 10 --out runs/calibration
python fabrication/run_pipeline.py --mask /path/to/mask.png --mm-per-pixel 0.1 --out runs/fabrication
python experiments/run_nesting.py --datasets /path/to/contour_sets --out runs/nesting
```

The nesting input contains `Dataset_40`, `Dataset_80`, `Dataset_100` and `Dataset_130`. Each folder contains a `contour_info.csv` with `contour_id` and matching `contour_<id>.json` files with `nurbs_points`, `actual_width_cm` and `actual_height_cm`. Algorithm modules preserve the original geometry and method-specific spacing settings. The repetitions use one warm-up run followed by 20 timed runs per method and set.

The DXF implementation preserves the application's `1/2.3` display-coordinate factor. The synthetic example records this factor in `summary.json`. This package stops at file export and does not send commands to a laser machine.

## 9. Measurement and boundary-offset analysis

```bash
python analysis/measure_cut_precision.py --plan /path/to/contour.json --scans /path/to/scans --dpi 200 --ddof 0 --out runs/cut_measurements
python analysis/coverage_sensitivity.py --input /path/to/contours --out runs/coverage
```

Cutting measurements use farthest-axis alignment, reflected-orientation selection and upper/lower vertical profile differences at 100 sampling positions. The original metric implementation uses SD denominator `n` (`--ddof 0`), while `--ddof 1` requests sample SD. Coverage inputs may provide `points_mm`, or `nurbs_points` together with `pixel_scale_mm_per_px`. Offsets are −3, −2, −1, +1, +2 and +3 mm. These are prescribed geometric perturbations of the input designs.

## Code notice

The author-written modules in this archive are supplied as supplementary research code for the accompanying manuscript. Portable entry points and synthetic examples are provided alongside those components.

TransUNet, PANet and SAMed are external dependencies. Their repositories and exact commits are listed in `external_sources.json`. They are retrieved only when explicitly requested through `tools/fetch_external_sources.py`. Keep the license, copyright and attribution files distributed by those repositories. Other dependencies retain their respective upstream licenses.

This archive does not include commercial laser-controller SDKs, device drivers, login credentials or cloud-service configuration.

## Package checks

The source passed syntax checks for 58 Python modules and three notebooks, and help checks for 18 command-line entrypoints. Synthetic smoke tests covered data export, statistics, calibration, coverage analysis, nesting and DXF generation. These checks did not retrain the research models. `SHA256SUMS.txt` records file integrity. Run `python tools/check_package.py` to verify it.
