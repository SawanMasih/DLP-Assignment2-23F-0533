# Deep Learning for Perception – Assignment 2 (Section B)

Robust CIFAR-10 classification from a very small training set (500 images selected, 400 used for training),
using an Inception-style CNN trained from scratch, data augmentation and regularisation.
Framework: **TensorFlow / Keras 2.21**. Roll No: 23F-0533.

## Repository contents

| File | Description |
|---|---|
| `DL_ASS02_23F_0533.ipynb` | Completed notebook: data split, model, all experiments, ablation, error analysis, TTA bonus, written discussion (Parts A–F) |
| `Assignment2_SectionB_Summary.docx` | One-page summary (final score, configuration, most useful change) |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Final result (selected configuration)

Selection was done on the 100-image **validation** split only. Mean ± std over 3 seeds (42, 123, 2026).

| Model | Validation | Clean test | Shifted test |
|---|---|---|---|
| **Geometric augmentation (selected)** | 43.33 ± 2.08% | 35.35 ± 0.92% | 24.37 ± 2.32% |
| Baseline (no augmentation) | 40.67 ± 3.21% | 35.86 ± 4.31% | 21.18 ± 0.99% |

Selected configuration: Inception-style CNN (44,122 parameters) + random crop (4 px padding), horizontal flip and
rotation up to ±25°; Adam (lr 1e-3), batch size 32, 60 epochs, best-validation checkpoint restored.
The change that contributed most to the selected model is **adding geometric augmentation** (validation +2.66 pp, shifted test +3.19 pp).
Gains are modest relative to the 2–3 pp seed-to-seed standard deviation.

The full 7-configuration ablation table is printed in the notebook (section 12) and discussed in Part D.

**Important:** the shifted test set is *simulated locally* (rotation, brightness/contrast, blur, noise, 25 % occlusion and an
*assumed* grayscale corruption, fixed seed 314159). It is not the official hidden test set.

## How to reproduce the results

### Option A – Google Colab (the environment the notebook was run in)
1. Open <https://colab.research.google.com> → **File → Upload notebook** → select `DL_ASS02_23F_0533.ipynb`.
2. *(Optional but faster)* **Runtime → Change runtime type → GPU**.
3. **Runtime → Run all.** CIFAR-10 is downloaded automatically by `keras.datasets.cifar10.load_data()`.
4. Outputs are written to `/content/DLP_Assignment2_Results` (CSV tables, plots, 21 checkpoints). Download them before the session ends.

### Option B – Local machine / Kaggle
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook DL_ASS02_23F_0533.ipynb
```
The notebook saves files under `/content/...`. Outside Colab, change these two paths in the notebook before running
(for example to `./checkpoints` and `./results`):
* `CHECKPOINT_DIR = Path("/content/checkpoints")` (section 4)
* `backup_dir = Path("/content/DLP_Assignment2_Results")` (section 10)

On Kaggle, enable **Internet** in the notebook settings so the CIFAR-10 download works.

### What the notebook does (run cells top to bottom, do not skip any)
1. Seeds everything (`SEEDS = [42, 123, 2026]`, deterministic TF ops enabled).
2. Draws 50 images per class (fixed seed) and splits them 40 train / 10 validation per class (400 / 100).
3. Trains 7 configurations × 3 seeds = 21 runs (baseline, geometric, photometric, combined, + Cutout, + Mixup/label smoothing, final regularised model), each for at most 60 epochs, keeping the best-validation checkpoint.
4. Builds the simulated shifted test set and evaluates every trained model on the clean and shifted test sets **once, after training**.
5. Produces the ablation table, per-class accuracy, confusion matrices, 8 misclassified examples, the grayscale-only test, learning curves, and the TTA bonus.

Approximate run time: a few minutes on a GPU, longer on CPU.

### Reproducibility notes
* Rules followed: trained from scratch, ≤ 1M parameters (44,122), Inception-style module, ≤ 60 epochs, 3 seeds, augmentation applied on the fly, no tuning on test data.
* Results can differ slightly on different hardware or library versions even with fixed seeds. The numbers above come from the original Colab run (TensorFlow 2.21.0).
