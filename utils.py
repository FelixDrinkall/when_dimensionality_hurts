import numpy as np
from sklearn.decomposition import PCA
import torch
import json
import hashlib

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve."""
    def __init__(self, patience=3, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def generate_latent_representations(ae, embeddings, device, batch_size=256):
    """
    Generate latent representations in mini-batches to avoid memory issues.

    Args:
        ae (Autoencoder): Trained autoencoder model.
        embeddings (torch.Tensor): Input embeddings to compress.
        device (torch.device): Device to perform computations on.
        batch_size (int): Size of mini-batches.

    Returns:
        torch.Tensor: Latent representations.
    """
    ae.eval()
    latent_representations = []

    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size].to(device) 
            _, z = ae(batch)
            latent_representations.append(z.cpu())

    return torch.cat(latent_representations, dim=0)

def build_dataset_config(args):
    """
    Build a dictionary of dataset-related arguments only.
    For 'fin_returns', include 'outlets' as well.
    """
    dataset_config = {
        "pretrained_model_name": args.pretrained_model_name,
        "sentiment": args.sentiment,
        "emotion": args.emotion,
        "dataset_path": args.dataset_path,
        "huggingface_dataset": args.huggingface_dataset,
        "label_column": args.label_column

    }
    if not args.yelp:
        dataset_config["outlets"] = args.outlets
        dataset_config["train_years"] = args.train_years
        dataset_config["test_years"] = args.test_years
        dataset_config["single_company"] = args.single_company
        dataset_config["temporal_split"] = args.temporal_split

    return dataset_config

def build_training_config(args, dataset_config):
    """
    Create a dict of all other args not in dataset_config.
    """
    args_dict = vars(args).copy()
    training_config = {**args_dict, **dataset_config}

    return training_config

def get_hash(config):
    """
    Hash only the dataset config (so classification/regression changes won't re-generate data).
    """
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()
