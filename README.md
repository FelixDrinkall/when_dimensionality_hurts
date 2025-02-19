# when_dimensionality_hurts


This repository contains the code for the paper [When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks](https://arxiv.org/abs/2502.02199). The dataset can be found [here](https://github.com/FelixDrinkall/financial-news-dataset/) and [here](https://huggingface.co/datasets/luckycat37/financial-news-dataset).

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Arguments and Configuration](#arguments-and-configuration)
- [Directory Structure](#directory-structure)
- [License](#license)
- [Citation](#citation)

---

## Installation

1. **Clone this repository**:
```bash
   git clone https://github.com/FelixDrinkall/when_dimensionality_hurts.git
```

2. **Install required Python packages**:
```bash
    pip install -r requirements.txt
```

3. **Set up Weights & Biases** (optional):
If you want to log experiments to Weights & Biases, sign up [here](https://wandb.ai/site) and run:

```bash
    wandb login
```

## Usage

After installing dependencies, you can run the main script. For example:

```bash
python main.py \
    --task regression \
    --train_years 2015 2016 2017 2018 \
    --test_years 2019 \
    --standardize_targets \
    --dataset_path <<dataset_dir>> \
    --text_column "text" \
    --label_column "rating"
```

Replace the arguments as needed to suit your use case. See [Arguments and Configuration](#arguments-and-configuration) below for an explanation of the available flags.

---

## Arguments and Configuration

Below are some commonly used arguments (the script will have more in `config.py` or as documented by `parse_args()`):

| Argument                  | Description                                                                                                          | Example                |
|---------------------------|----------------------------------------------------------------------------------------------------------------------|------------------------|
| `--task`                  | Task to perform: `classification` or `regression`.                                                                  | `classification`       |
| `--pretrained_model_name` | Name of the HuggingFace model to load.                                                                              | `bert-base-uncased`    |
| `--train_years`           | For **financial** datasets, the training year(s).                                                                   | `2015 2016`            |
| `--test_years`            | For **financial** datasets, the testing year(s).                                                                    | `2017`                 |
| `--outlets`               | For **financial** datasets, which news outlet(s).                                                                   | `Bloomberg`            |
| `--batch_size`            | Batch size for embedding extraction.                                                                                | `16`                   |
| `--random_seed`           | Random seed for reproducibility.                                                                                    | `42`                   |
| `--temporal_split`        | Use a temporal split (train < certain date, validation >= that date) for train/val sets.                            | *(flag)*               |
| `--single_company`        | Filter dataset for samples that only include one mentioned company.                                                 | *(flag)*               |
| `--epochs_ae`            | Number of epochs for autoencoder training.                                                                           | `25`                   |
| `--epochs_reg`           | Number of epochs for regression/classification model training.                                                       | `25`                   |
| `--learning_rate`         | Learning rate for the optimizer.                                                                                    | `1e-4`                 |
| `--ae_hidden_dim`         | Latent dimensionality (bottleneck) of the autoencoder.                                                              | `256`                  |
| `--mlp_hidden_dim`        | Hidden dimensionality for the MLP regressor/classifier.                                                             | `128`                  |
| `--weight_decay`          | Weight decay (L2 regularization) for the optimizer.                                                                 | `0.0`                  |
| `--grad_acc_steps`        | Gradient accumulation steps.                                                                                        | `4`                    |
| `--dropout_prob`          | Dropout probability in the MLP.                                                                                     | `0.1`                  |
| `--max_length`            | Maximum token length for the transformer model.                                                                     | `512`                  |
| `--ae_patience`           | Patience for early stopping in autoencoder training.                                                                | `5`                    |
| `--mlp_patience`          | Patience for early stopping in MLP training.                                                                        | `5`                    |
| `--huggingface_dataset`   | Whether to use a generic HuggingFace dataset instead of stock returns.                                              | *(flag)*               |
| `--dataset_path`          | Path to the HuggingFace dataset on disk.                                                                            | `data/my_dataset`      |
| `--text_column`           | Name of the text column in HuggingFace dataset.                                                                     | `review_text`          |
| `--label_column`          | Name of the label column in HuggingFace dataset.                                                                    | `rating`               |
| `--label_shift`           | Integer shift to apply to the label column (e.g., if labels are in `[1..5]`, shift by -1).                          | `0`                    |
| `--standardize_targets`   | Apply per-ticker standardization to the targets (mean=0, std=1) in regression.                                      | *(flag)*               |
| `--normalize_targets`     | Apply per-ticker min-max normalization to the targets in regression.                                               | *(flag)*               |
| `--random_forest`         | Use Random Forest instead of MLP for regression/classification.                                                     | *(flag)*               |
| `--use_raw_embeddings`    | Use embeddings directly (no autoencoder) for the final prediction.                                                  | *(flag)*               |
| `--emotion`               | Use emotion features in prediction (if available).                                                                  | *(flag)*               |
| `--sentiment`             | Use sentiment features in prediction (if available).                                                                | *(flag)*               |

To see all available arguments, either check the `config.py` file or run:

```bash
python main.py --help
```

---

## Directory Structure

This should be the directory structure of the repo:

```
project_root/
├─ processed_data_final/
│  ├─ finance.yahoo.com/
│  │  ├─ 2017_processed.json
│  │  ├─ 2018_processed.json
│  │  ├─ 2019_processed.json
│  │  ├─ 2020_processed.json
│  │  ├─ 2021_processed.json
│  │  ├─ 2022_processed.json
│  │  ├─ 2023_processed.json
│  ├─ [Other possible data sources]
│
├─ output/
│  ├─ datasets/
│  │  └─ <dataset_type>/
│  │     └─ <dataset_hash>/
│  │        ├─ dataset_config.json
│  │        ├─ [dataset files...]
│  │        └─ [embeddings...]
│  ├─ results/
│  │  └─ <dataset_type>/
│  │     └─ <task>/
│  │        └─ <training_hash>/
│  │           ├─ results/
│  │           │  ├─ [CSV logs]
│  │           │  └─ [TXT logs]
│  │           ├─ combined_results/
│  │           └─ training_config.json
│
├─ models/                                  
│  ├─ autoencoder.py/
│  ├─ embeddings.py/
│  ├─ regressor.py/
│
├─ config.py
├─ data_loaders.py
├─ utils/
├─ main.py
├─ requirements.txt
├─ README.md
```

- **`<dataset_hash>`** and **`<training_hash>`** are automatically generated to ensure reproducible experiments and avoid overwriting previous runs.

---

## License

[MIT License](https://opensource.org/license/mit).

---

Feel free to open an Issue or Pull Request if you encounter any bugs or have suggestions for new features.


## Citation

If you use this repository or find it helpful, please cite the following paper:

```
@misc{drinkall2025dimensionalityhurtsrolellm, 
    title={When Dimensionality Hurts: The Role of LLM Embedding Compression for Noisy Regression Tasks}, 
    author={Felix Drinkall and Janet B. Pierrehumbert and Stefan Zohren}, 
    year={2025}, 
    eprint={2502.02199}, 
    archivePrefix={arXiv}, 
    primaryClass={cs.CL}, 
    url={https://arxiv.org/abs/2502.02199} 
}
```

You can also access the paper [here](https://arxiv.org/abs/2502.02199).
