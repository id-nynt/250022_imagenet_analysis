# XGBoost Domain Adaptation Pipeline

A comprehensive machine learning pipeline for domain adaptation using XGBoost on EVA02 features. This project implements persistent, artifact-driven workflows that enable analysis and refinements without re-running entire pipelines.

## Project Structure

```
A3/
├── code/                          # Jupyter notebooks and scripts
│   ├── 01_data_analysis.ipynb     # Data exploration and analysis
│   ├── 02_experiments_pipeline.ipynb  # Main ML pipeline
│   └── 03_hypothesis_testing.py   # Statistical hypothesis testing
├── data/                          # Input data (to be downloaded)
│   ├── features/                  # Feature CSV files
│   └── images/                    # ImageNetV2 images (optional)
├── results/                       # Generated artifacts and outputs
├── plots/                         # Generated visualizations
├── report/                        # Final report and documentation
└── requirements.txt               # Python dependencies
```

## Setup Instructions

### 1. Download Dataset

Choose one of the following options to download the required datasets:

**Option 1: From Assignment Guide**

- Follow the data download instructions provided in assignment guide
- Ensure you download the EVA02 feature CSV files

**Option 2: Alternative Download Link**

- Download from: [MyGGDrive-imagenetdata](https://drive.google.com/drive/folders/1yi9xb9ppeN5us_yof8Dxb4SsfKbU1VW4?usp=sharing)
- Extract all files to maintain the expected structure

**Required Files:**
After downloading, ensure these files are moved to the `data/features` directory:

- `train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv`
- `val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv`
- `v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv`

### 2. Environment Setup

**Basic Installation:**

```bash
pip install -r requirements.txt
```

**For CUDA GPU support:**

```bash
# For CUDA GPU support (optional but recommended)
pip install xgboost[gpu]

# Or using conda
conda install -c conda-forge xgboost-gpu
```

**Verify Installation:**

```python
import xgboost as xgb
print("XGBoost version:", xgb.__version__)
print("GPU support:", xgb.build_info()['USE_CUDA'])
```

### 3. Running the Pipeline

Follow these steps in order:

**Step 1: Data Analysis**

1. Open `code/01_data_analysis.ipynb`
2. Run all cells sequentially
3. Review generated plots and analysis

**Step 2: Main Experiments Pipeline**

1. Open `code/02_experiments_pipeline.ipynb`
2. Run Steps 0-9 (setup through performance gap analysis)
3. Generated artifacts will be saved to `results/` directory

**Step 3: Statistical Hypothesis Testing**

```bash
# Run from the A3 directory
python code/03_hypothesis_testing.py
```

This script analyzes the results from Step 2 and generates statistical summaries for hypotheses H1-H3.

**Step 4: Model Refinement**

1. Return to `code/02_experiments_pipeline.ipynb`
2. Continue with Step 10 (hypothesis testing & model refinement)
3. Run remaining cells to complete the pipeline

## Pipeline Features

### GPU/CPU Auto-Detection

The pipeline automatically detects CUDA GPU availability. If found, it uses `device='cuda'`; otherwise falls back to CPU. Detection includes NVML, `nvidia-smi`, and PyTorch checks.

### Main Pipeline Steps

1. **Setup** - Path configuration, device detection, directory creation
2. **Data Inspection** - Validate headers, infer feature columns
3. **Data Loading** - Chunked CSV loading with memory management
4. **Hyperparameter Tuning** - Optuna optimization (optional)
5. **Model Training** - XGBoost with checkpointing and auto-resume
6. **Evaluation** - Performance metrics and accuracy analysis
7. **Feature Analysis** - Importance scores and visualization
8. **Save Results** - Export performance metrics and model artifacts
9. **Performance Gap Analysis** - Domain shift analysis, t-SNE, clustering
10. **Hypothesis and Model Refinement**

- **H1**: Domain shift hypothesis testing using KS statistics
- **H2**: Per-class difficulty analysis with paired t-tests
- **H3**: Feature importance-shift correlation analysis
- **Model Refinement** - Domain-weighted and class-weighted training
- **Comparative Analysis** - Model performance comparison

## Output Artifacts

### Results Directory (`results/`)

**Core Results:**

- `final_results.json` - Performance metrics and evaluation results
- `xgb_checkpoint.json` / `xgb_meta.json` - Main model checkpoints and metadata
- `feature_importances.csv` - Feature importance scores and rankings
- `class_accuracy_delta.csv` - Per-class performance analysis
- `feature_shift_analysis.csv` - Feature distribution shift statistics
- `gap_analysis_summary.json` - Performance gap analysis summary

**Hypothesis Testing Results:**

- `H1_shift_summary.json` - Domain shift analysis summary
- `H1_shift_top10.csv` - Top 10 features with highest KS statistics
- `H2_class_delta_summary.json` - Class difficulty analysis
- `H2_top10_val_minus_test.csv` - Top 10 classes with largest val>test gaps
- `H2_top10_test_minus_val.csv` - Top 10 classes with largest test>val gaps
- `H3_importance_shift_correlation.json` - Feature importance-shift correlations

**Model Refinement Results:**

- `model_refined_comparison.csv` - Comparison of refined models
- `importance_weights.npy` - Domain adaptation weights

### Plots Directory (`plots/`)

**Data Analysis Plots:**

- `class_distribution_analysis.png` - Class balance analysis across datasets
- `feature_correlation_heatmap.png` - Feature correlation heatmap
- `feature_distribution_comparison.png` - Feature distribution comparison

**Performance Analysis Plots:**

- `feature_importances.png` - Feature importance visualization
- `tsne_val_vs_test.png` - t-SNE visualization of domain shift
- `cluster_distribution.png` - Cluster analysis visualization
- `feature_distribution_shifts.png` - Feature distribution shift analysis
- `cm_val_problem.png` / `cm_test_problem.png` - Confusion matrices for problematic classes
- `model_comparison.png` - Model performance comparison

## Data Requirements

The pipeline expects these CSV files in `data/features/`:

**Required Files:**

- `train_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` (Training data)
- `val_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` (Validation/Test Set 1)
- `v2_eva02_large_patch14_448.mim_m38m_ft_in22k_in1k.csv` (Test Set 2/ImageNetV2)

**Expected Format:**

- CSV files with 1024 feature columns (labeled 0-1023)
- One 'label' column with class indices (0-999)
- Optional 'path' column (ignored during processing)

## Troubleshooting

### Memory Issues

If you encounter memory errors:

1. Reduce `sample_size` in hyperparameter tuning
2. Use smaller `chunk_size` in data loading
3. Consider using subset training data (10k samples)
4. Close other applications to free up RAM

### Missing Dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn optuna torch scipy
```

### CUDA Issues

- Ensure CUDA toolkit is installed
- Verify XGBoost GPU support: `xgb.build_info()['USE_CUDA']`
- Fall back to CPU if GPU issues persist

### Data File Issues

- Ensure all CSV files are in `data/features/` directory
- Check file permissions and disk space
- Verify CSV files are not corrupted
- Make sure file paths match exactly (case-sensitive on Linux/Mac)

### Pipeline Execution Issues

- Run cells in order as specified
- Check that Step 9 completes before running `03_hypothesis_testing.py`
- Ensure `results/` directory contains required CSV files before hypothesis testing
- If interrupted, the pipeline can resume from checkpoints

## Key Notes

- **Validation vs Test Sets**: Validation set is Test Set 1; v2 CSV is Test Set 2 (ImageNetV2)
- **Domain Adaptation**: Uses logistic regression derived importance weights
- **Class Weighting**: Uses per-class performance deltas for rebalancing
- **Auto-Resume**: Pipeline automatically resumes from checkpoints if interrupted
- **Statistical Testing**: Run hypothesis testing script between Steps 9 and 10
- **Reproducibility**: All random operations use `random_state=42` for consistency

## Pipeline Persistence

The pipeline is designed for persistence and re-runnability:

- All artifacts saved to `results/` and `plots/`
- Models include checkpointing for auto-resume
- Each step prints key outputs and saves intermediate results
- Hypothesis testing validates statistical claims with proper effect sizes

## Dependencies

Core requirements (see `requirements.txt`):

- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- xgboost>=2.0.0
- scipy>=1.10.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- optuna>=3.4.0
- torch>=2.0.0

For CUDA support, install XGBoost with GPU support as described in the setup section.
