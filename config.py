import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate models with autoencoder for regression/classification.")
    
    # General Arguments
    parser.add_argument("--single_company", action="store_true", help="Filters the dataset for samples that only include one mentioned company.")
    parser.add_argument("--temporal_split", action="store_true", help="Bool for whether the train and validation sets should have a temporal split.")
    parser.add_argument("--task", type=str, required=True, choices=["regression", "classification"],
                        help="Specify whether the task is regression or classification.")
    parser.add_argument("--pretrained_model_name", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Name of the pre-trained transformer model.")
    parser.add_argument("--train_years", nargs='+', required=True, type=int,
                        help="List of years to use for training (e.g., 2020 2021).")
    parser.add_argument("--test_years", nargs='+', required=True,
                        help="Years to use for testing (e.g., 2022).")
    parser.add_argument("--outlets", type=str, default="finance.yahoo.com",
                        help="Either 'all' or a specific news outlet.")    
    parser.add_argument("--dataset_path", type=str, default="data/writing_quality",
                        help="Directory of huggingface dataset.")
    parser.add_argument("--text_column", type=str, default="full_text",
                        help="Directory of huggingface dataset.")    
    parser.add_argument("--label_column", type=str, default="cohesion",
                        help="Directory of huggingface dataset.")
    parser.add_argument("--label_shift", type=int, default=0,
                        help="Directory of huggingface dataset.")
    # Training Parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs_ae", type=int, default=25, help="Number of epochs for autoencoder training.")
    parser.add_argument("--epochs_reg", type=int, default=25, help="Number of epochs for regression/classification training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer.")
    parser.add_argument("--ae_hidden_dim", type=int, default=256, help="Latent dimensionality of the autoencoder.")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="Hidden dimensionality of the MLP.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer.")
    parser.add_argument("--grad_acc_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout probability for the MLP.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum token length for transformer models.")
    parser.add_argument("--ae_patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--mlp_patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--huggingface_dataset", action="store_true", help="Use generic huggingface dataset instead of stock returns")
    parser.add_argument("--standardize_targets", action="store_true", help="Apply per-ticker standardization to the targets")
    parser.add_argument("--normalize_targets", action="store_true", help="Apply per-ticker min-max normalization to the targets")
    parser.add_argument("--random_forest", action="store_true", help="Random Forest model boolean. MLP if False.")
    parser.add_argument("--use_raw_embeddings", action="store_true", help="Boolean for using raw embeddings in prediction.")
    parser.add_argument("--emotion", action="store_true", help="Boolean for using emotion features in prediction.")
    parser.add_argument("--sentiment", action="store_true", help="Boolean for using sentiment in prediction.")

    return parser.parse_args()
