import json
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datasets import load_from_disk
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import os
import wandb
import pickle
from tqdm import tqdm

from config import parse_args
from data_loaders import (
    TextRegressionDataset,
    YelpDataset,
    merge_datasets,
    inverse_transform,
    extract_sentiment_emotion,
    equal_chunks_collate,
    apply_transform,
    get_transform_params,
    balance_classes,
    balance_binary_train_set,
    GenericHFDataset,
    create_train_val_test_representation
)
from models.autoencoder import train_autoencoder_with_metrics
from models.regressor import train_and_evaluate_with_early_stopping, compute_regression_metrics
from utils import generate_latent_representations, build_dataset_config, build_training_config, get_hash


def prepare_directories_and_configs(args):
    """
    Prepare dataset and training configurations, create necessary
    output directories for storing results, and return relevant paths.

    :param args: The command-line arguments parsed by parse_args().
    :return: A tuple containing:
             - dataset_config (dict)
             - training_config (dict)
             - dataset_dir (str): Path to the directory where dataset files are stored
             - res_dir (str): Path to the top-level results directory for this run
             - results_dir (str): Subdirectory under res_dir to store experiment results
             - dataset_type (str): Identifier for the dataset type (e.g., 'fin_returns')
             - dataset_hash (str): Hash for dataset configuration
             - training_hash (str): Hash for training configuration
    """
    # Build dataset & training configs
    dataset_config = build_dataset_config(args)
    training_config = build_training_config(args, dataset_config)
    dataset_hash = get_hash(dataset_config)
    training_hash = get_hash(training_config)

    if args.huggingface_dataset:
        dataset_type = args.dataset_path.split("/")[-1]
    else:
        dataset_type = "fin_returns"

    # Prepare directories and files
    dataset_dir = os.path.join("output", "datasets", dataset_type, dataset_hash)
    os.makedirs(dataset_dir, exist_ok=True)

    res_dir = os.path.join("output", "results", dataset_type, args.task, training_hash)
    os.makedirs(res_dir, exist_ok=True)

    results_dir = os.path.join(res_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    if dataset_type == "fin_returns":
        combined_results_dir = os.path.join(res_dir, "all_tickers_combined_results")
        os.makedirs(combined_results_dir, exist_ok=True)

    dataset_config_json = os.path.join(dataset_dir, "dataset_config.json")
    with open(dataset_config_json, "w") as f:
        json.dump(dataset_config, f, indent=4)

    train_config_json = os.path.join(res_dir, "training_config.json")
    with open(train_config_json, "w") as f:
        json.dump(training_config, f, indent=4)

    return (
        dataset_config,
        training_config,
        dataset_dir,
        res_dir,
        results_dir,
        dataset_type,
        dataset_hash,
        training_hash
    )


def load_tokenizer_and_text_model(args):
    """
    Load the tokenizer and text model from the specified pretrained model,
    and determine the appropriate device (CUDA or CPU).

    :param args: The command-line arguments parsed by parse_args().
    :return: A tuple containing:
             - tokenizer (AutoTokenizer)
             - text_model (AutoModel)
             - device (torch.device)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
    text_model = AutoModel.from_pretrained(args.pretrained_model_name).to(device)
    return tokenizer, text_model, device


def run_fin_returns(
    args,
    dataset_type,
    dataset_config,
    training_config,
    dataset_dir,
    res_dir,
    results_dir,
    tokenizer,
    text_model,
    device
):
    """
    Execute the entire pipeline for the 'fin_returns' dataset domain.
    This includes:
     - Loading or creating (balanced/unbalanced) datasets,
     - Splitting them into train/val/test,
     - Creating embeddings if they do not exist,
     - Running autoencoder compression (if specified),
     - Training MLP or Random Forest models,
     - Saving and logging results with Weights & Biases.

    :param args: Command-line arguments from parse_args().
    :param dataset_type: String representing the dataset type (e.g., 'fin_returns').
    :param dataset_config: Dictionary of dataset configuration parameters.
    :param training_config: Dictionary of training configuration parameters.
    :param dataset_dir: Directory path where dataset files are stored.
    :param res_dir: Directory path where overall results for this config are stored.
    :param results_dir: Directory path where model/training results are stored.
    :param tokenizer: Tokenizer from HuggingFace transformers.
    :param text_model: Pretrained text model (AutoModel) from HuggingFace transformers.
    :param device: Torch device (CUDA or CPU).
    """
    num_classes = 2  

    # Define data paths
    unbalanced_data_path = os.path.join(dataset_dir, "preprocessed_data.pkl")      
    unbalanced_emb_path = os.path.join(dataset_dir, "embeddings.pkl")             

    balanced_data_path = os.path.join(dataset_dir, "balanced_preprocessed_data.pkl")   
    balanced_emb_path = os.path.join(dataset_dir, "balanced_embeddings.pkl")          

    if args.task == "classification":
        data_path = balanced_data_path
        emb_path = balanced_emb_path
    else:
        data_path = unbalanced_data_path
        emb_path = unbalanced_emb_path

    # Check for procesed embeddings
    if os.path.exists(emb_path):
        print(f"[FIN_RETURNS] Found existing embeddings in {emb_path}")
        with open(emb_path, "rb") as f:
            (
                train_embeddings,
                train_targets,
                val_embeddings,
                val_targets,
                test_embeddings,
                test_targets
            ) = pickle.load(f)

        if os.path.exists(data_path):
            print(f"[FIN_RETURNS] Loading dataset from {data_path} to retrieve tickers")
            with open(data_path, "rb") as f:
                train_dataset, test_dataset, tickers = pickle.load(f)

    else:
        # If embeddings don't exist, check for dataset
        if os.path.exists(data_path):
            print(f"[FIN_RETURNS] Loading dataset from {data_path}")
            with open(data_path, "rb") as f:
                train_dataset, test_dataset, tickers = pickle.load(f)
        else:
            # If the dataset doesn't exist, process dataset: balanced for classificaiton and unblanced for regression
            unbalanced_exists = os.path.exists(unbalanced_data_path)
            if args.task == "classification":
                if unbalanced_exists:
                    print(f"[FIN_RETURNS] Loading unbalanced dataset from {unbalanced_data_path} to create balanced version...")
                    with open(unbalanced_data_path, "rb") as f:
                        train_dataset, test_dataset, tickers = pickle.load(f)
                else:
                    print("[FIN_RETURNS] Merging datasets (unbalanced, first-time)...")
                    print("Training dataset")
                    train_dataset = merge_datasets(args.train_years, [args.outlets], tokenizer, args.single_company)

                    print("Test dataset")
                    test_dataset = merge_datasets(args.test_years, [args.outlets], tokenizer, args.single_company)
                    tickers = train_dataset.tickers
                    print(f"[FIN_RETURNS] Saving unbalanced dataset to {unbalanced_data_path}")
                    with open(unbalanced_data_path, "wb") as f:
                        pickle.dump((train_dataset, test_dataset, tickers), f)

                print("[FIN_RETURNS] Balancing the training dataset for classification...")
                balanced_train_dataset = balance_binary_train_set(train_dataset, args)

                print(f"[FIN_RETURNS] Saving balanced dataset to {balanced_data_path}")
                with open(balanced_data_path, "wb") as f:
                    pickle.dump((balanced_train_dataset, test_dataset, tickers), f)
                train_dataset = balanced_train_dataset
            else:
                if not os.path.exists(unbalanced_data_path):
                    print("[FIN_RETURNS] Merging datasets (unbalanced, first time)...")
                    print("Training dataset")
                    train_dataset = merge_datasets(args.train_years, [args.outlets], tokenizer, args.single_company)
                    print("Test dataset")
                    test_dataset = merge_datasets(args.test_years, [args.outlets], tokenizer, args.single_company)
                    tickers = train_dataset.tickers
                    with open(unbalanced_data_path, "wb") as f:
                        pickle.dump((train_dataset, test_dataset, tickers), f)
                else:
                    with open(unbalanced_data_path, "rb") as f:
                        train_dataset, test_dataset, tickers = pickle.load(f)

        # Train/validation split
        print("[FIN_RETURNS] Splitting dataset into train and validation...")
        if args.temporal_split:
            train_dataset.sort_by_date()
            dataset_size = len(train_dataset)
            val_size = int(0.1 * dataset_size)
            train_size = dataset_size - val_size

            train_indices = range(train_size)
            val_indices = range(train_size, dataset_size)
        else:
            train_indices, val_indices = train_test_split(
                range(len(train_dataset)),
                test_size=0.1,
                random_state=args.random_seed
            )
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=equal_chunks_collate
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=equal_chunks_collate
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=equal_chunks_collate
        )

        # Create embeddings
        print("[FIN_RETURNS] Creating embeddings, please wait...")

        (
            train_embeddings,
            train_targets,
            val_embeddings,
            val_targets,
            test_embeddings,
            test_targets
        ) = create_train_val_test_representation(
            model=text_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_dataset=train_subset,
            val_dataset=val_subset,
            test_dataset=test_dataset,
            device=device,
            tickers=tickers,
            args=args
        )
        # Save the embeddings
        print(f"[FIN_RETURNS] Saving embeddings to {emb_path}...")
        with open(emb_path, "wb") as f:
            pickle.dump(
                (
                    train_embeddings,
                    train_targets,
                    val_embeddings,
                    val_targets,
                    test_embeddings,
                    test_targets
                ),
                f
            )
        print("[FIN_RETURNS] Done creating embeddings.")

    compression_ratios = ["raw"]
    mlp_hidden_dims = [64]
    all_results = []
    one_pass = False

    test_dataset_size = len(test_targets) * len(tickers)
    sample_size = min(10000, test_dataset_size)
    sampled_test_indices = np.random.choice(test_dataset_size, sample_size, replace=False)

    for latent_dim in tqdm(compression_ratios, desc="Comp ratios"):
        combined_results_list = []
        
        print(f"Generating latent representations for latent dimension {latent_dim}...")
        if latent_dim == "raw":
            latent_dim_str = "raw"
            train_z = train_embeddings
            val_z = val_embeddings
            test_z = test_embeddings
        elif args.emotion:
            latent_dim = latent_dim_str = "emotion"
            train_z = train_embeddings
            val_z = val_embeddings
            test_z = test_embeddings
            one_pass = True
        elif args.sentiment:
            latent_dim = latent_dim_str = "sentiment"
            train_z = train_embeddings
            val_z = val_embeddings
            test_z = test_embeddings
            one_pass = True
        else:
            latent_dim_str = str(latent_dim)
            print(f"Training Autoencoder with latent dimension {latent_dim}...")
            start_time = time.time()
            ae, train_losses, val_losses = train_autoencoder_with_metrics(
                train_embeddings=train_embeddings,
                val_embeddings=val_embeddings,
                latent_dim=latent_dim,
                device=device,
                args=args,
            )
            total_ae_time = time.time() - start_time
            print(f"Autoencoder (Latent={latent_dim}) training completed in {total_ae_time:.2f} seconds.")
            train_z = generate_latent_representations(ae, train_embeddings, device)
            val_z = generate_latent_representations(ae, val_embeddings, device)
            test_z = generate_latent_representations(ae, test_embeddings, device)

        combined_train_z = torch.cat([train_z for _ in tickers])
        combined_val_z = torch.cat([val_z for _ in tickers])
        combined_test_z = torch.cat([test_z for _ in tickers])
        combined_train_targets = torch.cat([train_targets[t] for t in tickers])
        combined_val_targets = torch.cat([val_targets[t] for t in tickers])
        combined_test_targets = torch.cat([test_targets[t] for t in tickers])

        for mlp_hidden_dim in tqdm(mlp_hidden_dims, desc="MLP hidden dims"):

            summary_file = os.path.join(results_dir, f"summary_latent_{latent_dim}_mlp_{mlp_hidden_dim}.csv")
            if os.path.exists(summary_file):
                print(f"File {summary_file} already exists. Skipping latent_dim={latent_dim}, mlp={mlp_hidden_dim}")
                continue

            print(f"Training MLP with hidden dimension {mlp_hidden_dim}...")
            run_name = f"Combined_MLP_hidden_{mlp_hidden_dim}_latent_{latent_dim}"
            wandb.init(
                project="",
                entity="",
                name=run_name,
                group=""
            )    

            wandb.config.update({
                "mlp_hidden_dim": mlp_hidden_dim,
                "task": args.task,
                "latent_dim": latent_dim,
            })

            combined_transform_params = get_transform_params(
                combined_train_targets,
                standardize=args.standardize_targets,
                normalize=args.normalize_targets
            )
            combined_train_targets_trans = apply_transform(combined_train_targets, combined_transform_params)
            combined_val_targets_trans = apply_transform(combined_val_targets, combined_transform_params)
            combined_test_targets_trans = apply_transform(combined_test_targets, combined_transform_params)

            combined_results_file = os.path.join(
                results_dir, f"combined_results_latent_{latent_dim}_mlp_{mlp_hidden_dim}.txt"
            )
            metrics_combined, preds_combined, reg_val_losses_combined = train_and_evaluate_with_early_stopping(
                combined_train_z, combined_train_targets_trans,
                combined_val_z, combined_val_targets_trans,
                combined_test_z, combined_test_targets_trans,
                args=args,
                device=device,
                results_file=combined_results_file,
                mlp_hidden_dim=mlp_hidden_dim,
                num_classes=num_classes,
                task=args.task,
                combined=True,
                use_random_forest=args.random_forest
            )

            if args.task == "regression":
                filtered_combined_metrics = {
                    f"combined_MAE": metrics_combined["MAE"],
                    f"combined_MSE": metrics_combined["MSE"],
                    f"combined_R²": metrics_combined["R²"],
                    f"combined_Huber": metrics_combined["Huber"]
                }
            else:
                filtered_combined_metrics = metrics_combined
            wandb.log(filtered_combined_metrics)

            if args.task == "regression":
                preds_combined_unscaled = inverse_transform(preds_combined, combined_transform_params)
                combined_test_targets_unscaled = inverse_transform(combined_test_targets_trans, combined_transform_params)

                y_combined_mean_unscaled = inverse_transform(
                    combined_train_targets.numpy(),
                    combined_transform_params
                ).mean()

                unscaled_metrics_combined = compute_regression_metrics(
                    combined_test_targets_unscaled, preds_combined_unscaled, y_combined_mean_unscaled
                )
                unscaled_metrics_combined = {f"unscaled_{k}": v for k, v in unscaled_metrics_combined.items()}

                wandb.log(unscaled_metrics_combined)
                filtered_unscaled_combined_metrics = {
                    f"unscaled_combined_MAE": unscaled_metrics_combined["unscaled_MAE"],
                    f"unscaled_combined_MSE": unscaled_metrics_combined["unscaled_MSE"],
                    f"unscaled_combined_R²": unscaled_metrics_combined["unscaled_R²"],
                    f"unscaled_combined_Huber": unscaled_metrics_combined["unscaled_Huber"]
                }
                wandb.log(filtered_unscaled_combined_metrics)
            else:
                unscaled_metrics_combined = metrics_combined

            wandb.finish()

            combined_results_list.append({
                "Latent Dim": latent_dim,
                "Task": args.task,
                "MLP Hidden Dim": mlp_hidden_dim,
                **metrics_combined,
                **unscaled_metrics_combined,
                "Final Regressor Validation Loss": reg_val_losses_combined[-1] if reg_val_losses_combined else None,
                "Prediction Variance": np.var(preds_combined),
                "Regressor Training Time (s)": metrics_combined.get("Training Time", None),
                "Preds": preds_combined.tolist(),
                "Y_true": combined_test_targets_trans.tolist(),
                "Preds_unscaled": preds_combined_unscaled.tolist() if args.task == "regression" else None,
                "Y_true_unscaled": combined_test_targets_unscaled.tolist() if args.task == "regression" else None,
            })
        combined_results_filename = f"combined_results_latent_{latent_dim_str}_mlp_{mlp_hidden_dim}.csv"
        combined_results_path = os.path.join(results_dir, combined_results_filename)
        pd.DataFrame(combined_results_list).to_csv(combined_results_path, index=False)
        print(f"Saved combined results for latent_dim={latent_dim_str} to {combined_results_path}")
        if one_pass:
            break

    all_results_file = os.path.join(results_dir, "all_results.csv")
    pd.DataFrame(all_results).to_csv(all_results_file, index=False)

    print(f"All individual results saved to {all_results_file}.")
    print(f"Combined results saved to {combined_results_file}.")


def run_huggingface_dataset(
    args,
    dataset_type,
    dataset_config,
    training_config,
    dataset_dir,
    res_dir,
    results_dir,
    tokenizer,
    text_model,
    device
):
    """
    Execute the pipeline for a HuggingFace dataset. This includes:
     - Loading or creating (balanced/unbalanced) splits,
     - Creating embeddings if they do not exist,
     - (Optionally) training an AutoEncoder,
     - Training MLP or RandomForest,
     - Logging results with Weights & Biases.

    :param args: Command-line arguments from parse_args().
    :param dataset_type: String representing the dataset type.
    :param dataset_config: Dictionary of dataset configuration parameters.
    :param training_config: Dictionary of training configuration parameters.
    :param dataset_dir: Directory path where dataset files are stored.
    :param res_dir: Directory path where overall results for this config are stored.
    :param results_dir: Directory path where model/training results are stored.
    :param tokenizer: Tokenizer from HuggingFace transformers.
    :param text_model: Pretrained text model (AutoModel) from HuggingFace transformers.
    :param device: Torch device (CUDA or CPU).
    """
    if args.task == "classification":
        num_classes = args.num_classes  
    else:
        num_classes = 1 

    unbalanced_data_path = os.path.join(dataset_dir, "hf_unbalanced_data.pkl")
    unbalanced_emb_path = os.path.join(dataset_dir, "hf_unbalanced_embeddings.pkl")

    balanced_data_path = os.path.join(dataset_dir, "hf_balanced_data.pkl")
    balanced_emb_path = os.path.join(dataset_dir, "hf_balanced_embeddings.pkl")

    if args.task == "classification":
        data_path = balanced_data_path
        emb_path = balanced_emb_path
    else:
        data_path = unbalanced_data_path
        emb_path = unbalanced_emb_path

    if os.path.exists(emb_path):
        print(f"[HF-DATASET] Found existing embeddings in {emb_path}")
        with open(emb_path, "rb") as f:
            (
                train_embeddings,
                train_targets,
                val_embeddings,
                val_targets,
                test_embeddings,
                test_targets
            ) = pickle.load(f)
    else:
        if os.path.exists(data_path):
            print(f"[HF-DATASET] Loading dataset from {data_path}")
            with open(data_path, "rb") as f:
                (
                    train_split,
                    val_split,
                    test_split
                ) = pickle.load(f)
        else:
            print(f"[HF-DATASET] Loading HuggingFace dataset from {args.dataset_path}")
            hf_dataset = load_from_disk(args.dataset_path)

            if "full" in hf_dataset:
                print("[HF-DATASET] Detected 'full' dataset format.")
                if args.dataset_path == "data/amazon_book_reviews":
                    print("[HF-DATASET] Sampling 100,000 instances from Amazon Book Reviews dataset...")
                    total_samples = 100000
                    hf_dataset = hf_dataset["full"].shuffle(seed=args.random_seed).select(range(total_samples))
                
                train_valid = hf_dataset.train_test_split(test_size=0.2, seed=args.random_seed)
                raw_train = train_valid["train"]
                test_valid = train_valid["test"].train_test_split(test_size=0.5, seed=args.random_seed)
                
                raw_val_split = test_valid["train"]
                raw_test = test_valid["test"]

                print("[HF-DATASET] Manually split dataset into Train (80%) / Val (10%) / Test (10%).")
            
            elif "train" in hf_dataset:
                raw_train = hf_dataset["train"]
                
                if "test" in hf_dataset:
                    raw_test = hf_dataset["test"]
                else:
                    print("[HF-DATASET] No 'test' split found. Creating one from train split.")
                    train_valid = raw_train.train_test_split(test_size=0.1, seed=args.random_seed)
                    raw_train = train_valid["train"]
                    raw_test = train_valid["test"]

                if "validation" in hf_dataset:
                    raw_val_split = hf_dataset["validation"]
                else:
                    print("[HF-DATASET] No 'validation' split found. Creating one from train split.")
                    val_fraction = 0.05
                    train_valid = raw_train.train_test_split(test_size=val_fraction, seed=args.random_seed)
                    raw_train_split = train_valid["train"]
                    raw_val_split = train_valid["test"]
            else:
                raise ValueError("[HF-DATASET] Dataset must have a 'train' or 'full' split!")

            train_split = raw_train_split if "raw_train_split" in locals() else raw_train
            val_split = raw_val_split
            test_split = raw_test

            if args.task == "regression":
                print("[HF-DATASET] Saving unbalanced dataset to", unbalanced_data_path)
                with open(unbalanced_data_path, "wb") as f:
                    pickle.dump((train_split, val_split, test_split), f)

            else:
                if not os.path.exists(unbalanced_data_path):
                    print("[HF-DATASET] Saving unbalanced dataset to", unbalanced_data_path)
                    with open(unbalanced_data_path, "wb") as f:
                        pickle.dump((raw_train_split, raw_val_split, raw_test), f)

                print("[HF-DATASET] Balancing the training portion for classification...")

                train_labels_np = np.array([row[args.label_column] for row in raw_train_split])
                balanced_indices = balance_classes(train_labels_np)  
                balanced_train_split = raw_train_split.select(balanced_indices)

                print("[HF-DATASET] Saving balanced dataset to", balanced_data_path)
                with open(balanced_data_path, "wb") as f:
                    pickle.dump((balanced_train_split, raw_val_split, raw_test), f)

                train_split = balanced_train_split
                val_split = raw_val_split
                test_split = raw_test

        # Build datasets
        print("[HF-DATASET] Building GenericHFDataset objects...")

        train_ds = GenericHFDataset(
            hf_dataset=train_split,
            text_column=args.text_column,
            target_column=args.label_column,
            tokenizer=tokenizer,
            task=args.task,
            label_shift=args.label_shift
        )
        val_ds = GenericHFDataset(
            hf_dataset=val_split,
            text_column=args.text_column,
            target_column=args.label_column,
            tokenizer=tokenizer,
            task=args.task,
            label_shift=args.label_shift
        )
        test_ds = GenericHFDataset(
            hf_dataset=test_split,
            text_column=args.text_column,
            target_column=args.label_column,
            tokenizer=tokenizer,
            task=args.task,
            label_shift=args.label_shift
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=equal_chunks_collate
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=equal_chunks_collate
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=equal_chunks_collate
        )

        # Create embeddings
        print("[HF-DATASET] Creating embeddings, please wait...")

        (
            train_embeddings,
            train_targets,
            val_embeddings,
            val_targets,
            test_embeddings,
            test_targets
        ) = create_train_val_test_representation(
            model=text_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            device=device,
            tickers=None,
            args=args
        )

        # Save embeddings
        print(f"[HF-DATASET] Saving embeddings to {emb_path}...")
        with open(emb_path, "wb") as f:
            pickle.dump(
                (
                    train_embeddings,
                    train_targets,
                    val_embeddings,
                    val_targets,
                    test_embeddings,
                    test_targets
                ),
                f
            )

    print("[HF-DATASET] Done creating embeddings.")

    compression_ratios = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    mlp_hidden_dims = [64]
    all_results = []
    one_pass = False

    for latent_dim in tqdm(compression_ratios, desc="Comp ratios"):
        print(f"Generating latent representations for latent dimension {latent_dim}...")

        if latent_dim == "raw":
            latent_dim_str = "raw"
            train_z = train_embeddings
            val_z = val_embeddings
            test_z = test_embeddings
        elif args.emotion:
            latent_dim = latent_dim_str = "emotion"
            train_z = train_embeddings
            val_z = val_embeddings
            test_z = test_embeddings
            one_pass = True
        elif args.sentiment:
            latent_dim = latent_dim_str = "sentiment"
            train_z = train_embeddings
            val_z = val_embeddings
            test_z = test_embeddings
            one_pass = True
        else:
            latent_dim_str = str(latent_dim)
            print(f"Training Autoencoder with latent dimension {latent_dim}...")
            start_time = time.time()
            ae, train_losses, val_losses = train_autoencoder_with_metrics(
                train_embeddings=train_embeddings,
                val_embeddings=val_embeddings,
                latent_dim=latent_dim,
                device=device,
                args=args,
            )
            total_ae_time = time.time() - start_time
            print(f"Autoencoder (Latent={latent_dim}) training completed in {total_ae_time:.2f} seconds.")
            train_z = generate_latent_representations(ae, train_embeddings, device)
            val_z = generate_latent_representations(ae, val_embeddings, device)
            test_z = generate_latent_representations(ae, test_embeddings, device)

        for mlp_hidden_dim in tqdm(mlp_hidden_dims, desc="MLP hidden dims"):
            run_name = f"HF_MLP_hidden_{mlp_hidden_dim}_latent_{latent_dim}"
            if dataset_type == "writing_quality":
                project_name = f"Regression_{dataset_type}_{args.label_column}_RF"
            else:
                project_name = f"Regression_{dataset_type}_RF"

            wandb.init(
                project=project_name,
                entity="",
                name=run_name,
                group=""
            )    

            wandb.config.update({
                "mlp_hidden_dim": mlp_hidden_dim,
                "task": args.task,
                "latent_dim": latent_dim,
            })

            summary_file = os.path.join(
                results_dir, f"hf_summary_latent_{latent_dim}_mlp_{mlp_hidden_dim}.csv"
            )
            if os.path.exists(summary_file):
                print(f"File {summary_file} exists, skipping latent_dim={latent_dim}, mlp={mlp_hidden_dim}")
                wandb.finish()
                continue

            print(f"[HF-DATASET] Training MLP with hidden dim {mlp_hidden_dim}...")
            train_targets_all = train_targets["ALL"]
            val_targets_all = val_targets["ALL"]
            test_targets_all = test_targets["ALL"]

            if args.task == "regression":
                transform_params = get_transform_params(
                    train_targets_all,
                    standardize=args.standardize_targets,
                    normalize=args.normalize_targets
                )
                train_targets_trans = apply_transform(train_targets_all, transform_params)
                val_targets_trans = apply_transform(val_targets_all, transform_params)
                test_targets_trans = apply_transform(test_targets_all, transform_params)
            else:
                train_targets_trans = train_targets_all
                val_targets_trans = val_targets_all
                test_targets_trans = test_targets_all

            results_file = os.path.join(
                results_dir,
                f"hf_results_latent_{latent_dim}_mlp_{mlp_hidden_dim}.txt"
            )

            metrics, preds, reg_val_losses = train_and_evaluate_with_early_stopping(
                train_z, train_targets_trans,
                val_z, val_targets_trans,
                test_z, test_targets_trans,
                args=args,
                device=device,
                results_file=results_file,
                mlp_hidden_dim=mlp_hidden_dim,
                num_classes=num_classes,
                task=args.task,
                use_random_forest=args.random_forest
            )
            wandb.log(metrics)

            if args.task == "regression":
                preds_unscaled = inverse_transform(preds, transform_params)
                test_targets_unscaled = apply_transform(test_targets_all, transform_params)
                test_targets_unscaled = inverse_transform(test_targets_unscaled, transform_params)
                test_targets_unscaled = test_targets_unscaled

                y_train_mean_unscaled = inverse_transform(
                    train_targets_all.numpy(),
                    transform_params
                ).mean()

                unscaled_metrics = compute_regression_metrics(
                    test_targets_unscaled, preds_unscaled, y_train_mean_unscaled
                )
                unscaled_metrics = {f"unscaled_{k}": v for k, v in unscaled_metrics.items()}
                preds = preds_unscaled
                y_true = test_targets_unscaled
                wandb.log(unscaled_metrics)
            else:
                if hasattr(test_targets_all, "numpy"):
                    y_true = test_targets_all.numpy()
                else:
                    y_true = test_targets_all

            preds_serialized = preds.tolist() if isinstance(preds, np.ndarray) else preds
            test_targets_serialized = y_true.tolist() if isinstance(y_true, np.ndarray) else y_true

            wandb.finish()

            summary_row = {
                "Task": args.task,
                **metrics,
                "Latent Dim": latent_dim,
                "MLP Hidden Dim": mlp_hidden_dim,
                "Preds": preds_serialized,
                "Y True": test_targets_serialized,
                "Final AE Training Loss": train_losses[-1] if latent_dim not in ["raw", "emotion", "sentiment"] else None,
                "Final AE Validation Loss": val_losses[-1] if latent_dim not in ["raw", "emotion", "sentiment"] else None,
                "Validation Convergence Epoch": len(val_losses) if latent_dim not in ["raw", "emotion", "sentiment"] else None,
                "Prediction Variance": np.var(preds_serialized),
                "Regressor Training Time (s)": metrics.get("Training Time", None),
                "AE Training Time (s)": total_ae_time if latent_dim not in ["raw", "emotion", "sentiment"] else 0.0,
            }

            pd.DataFrame([summary_row]).to_csv(summary_file, index=False)
            all_results.append(summary_row)

        if one_pass:
            break

    all_results_file = os.path.join(results_dir, "hf_all_results.csv")
    pd.DataFrame(all_results).to_csv(all_results_file, index=False)
    print(f"[HF-DATASET] All results saved to {all_results_file}.")
    print("Done!")


def main():
    """
    The main entry point of the script. It parses command-line arguments,
    prepares directories and configs, loads models/tokenizers, and routes
    execution to the appropriate branch (e.g., fin_returns, HuggingFace dataset).
    """
    args = parse_args()

    (
        dataset_config,
        training_config,
        dataset_dir,
        res_dir,
        results_dir,
        dataset_type,
        dataset_hash,
        training_hash
    ) = prepare_directories_and_configs(args)

    tokenizer, text_model, device = load_tokenizer_and_text_model(args)

    if not args.huggingface_dataset:
        run_fin_returns(
            args,
            dataset_type,
            dataset_config,
            training_config,
            dataset_dir,
            res_dir,
            results_dir,
            tokenizer,
            text_model,
            device
        )

    elif args.huggingface_dataset:
        run_huggingface_dataset(
            args,
            dataset_type,
            dataset_config,
            training_config,
            dataset_dir,
            res_dir,
            results_dir,
            tokenizer,
            text_model,
            device
        )

    else:
        pass


if __name__ == "__main__":
    main()