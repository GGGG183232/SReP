# SReP

Core search implementation for **SReP: Efficient Retrosynthetic Planning via Structural Rerooting**. SReP adapts local rerooting to multi-step retrosynthetic planning, combining global–local search, depth regularization, and duplicate-state pruning to improve route discovery under limited search budgets.

## 1. Installation

Use a Linux environment. GPU evaluation requires a CUDA-compatible PyTorch installation and an available NVIDIA GPU.

```bash
pip install torch numpy pandas tqdm rdkit rdchiral
pip install -r requirements-structure.txt
```

## 2. Required Files

Prepare the pretrained models, reaction templates, building-block stock, and target files separately, and place them as follows:

```text
RetrosynthesisPlanning/
├── saved_model/
│   ├── retro_star_value_ours.ckpt   # Single-step policy model
│   ├── template_rules.dat          # Reaction templates
│   └── best_epoch.pt               # Value model
├── prepare_data/
│   └── origin_dict.csv             # Building-block stock
└── test_dataset/
    └── USPTO.pkl                   # Target SMILES list
```

The stock CSV must contain a `mol` column. Each target file must contain a pickled list or tuple of target SMILES strings. The supported dataset names are `USPTO`, `Ames`, `BBBP`, `ClinTox`, `Toxicity_LD50`, `logP`, and `logS`; name each target file `<dataset>.pkl`.

To reproduce the paper's experiments, use the pretrained models, fixed building-block stock, and target subsets described in Section 4.1. The policy and value models remain frozen during search.

## 3. Running SReP

Run the following commands from the repository root.

Check the environment and required files using one GPU:

```bash
KEEA_DATASETS=USPTO KEEA_GPUS=0 KEEA_PREFLIGHT_ONLY=1 bash run_evaluation.sh
```

Run the first five USPTO targets with one GPU and one worker:

```bash
KEEA_DATASETS=USPTO KEEA_GPUS=0 KEEA_NUM_WORKERS=1 \
KEEA_TARGET_LIMIT=5 bash run_evaluation.sh
```

After preparing all seven target files, run the default evaluation:

```bash
bash run_evaluation.sh
```

The launcher starts evaluation in the background and prints the output directory and log-monitoring command. Adjust `CUDA_VISIBLE_DEVICES`, `KEEA_GPUS`, `KEEA_NUM_WORKERS`, and `KEEA_DATASETS` for the available hardware and selected datasets.

## 4. Default Configuration

| Setting | Default |
|---|---:|
| Model-call budget per target | 500 |
| Time limit per target | 600 seconds |
| Parallel workers | 20 |
| GPU IDs | 0, 1, 2, 3 |
| Base random seed | 20260920 |
| Retained reaction templates per prediction | 50 |
| Starting-point weight parameter α | 10 |
| Leiden resolution / iterations | 1 / 2 |
| Repartitioning at cumulative state expansions | 32, 64, 128, … |
| Depth regularization thresholds | 20 / 40 |

## 5. Outputs and Metrics

Each run creates a separate directory under `runs/`, containing the run configuration, evaluation log, per-target results, predicted routes, search traces, and dataset summaries.

- **Model calls:** actual invocations of the single-step model, including calls that return no candidate reactions. Cache hits do not consume the model-call budget.
- **State expansions:** the number of task states selected for expansion, regardless of whether candidate reactions are obtained.
- **Effective expansions:** expansions that yield at least one candidate reaction, including those using nonempty cached predictions. The exported `expanded` field reports this count.

## 6. Code Structure

- `RetrosynthesisPlanning/structure_search.py`: coordinated global–local search.
- `RetrosynthesisPlanning/structural_rerooter.py`: region partitioning and starting-point weights.
- `RetrosynthesisPlanning/state_graph.py` and `reroot_priority.py`: shared states and local label maintenance.
- `RetrosynthesisPlanning/depth_penalty.py`: depth regularization.
- `RetrosynthesisPlanning/evaluate_structure.py`: model loading, evaluation, and result export.
