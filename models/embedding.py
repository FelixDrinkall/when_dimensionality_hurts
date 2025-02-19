import torch
from tqdm import tqdm


def build_features_and_targets_tickers(
    model,
    loader,
    device,
    tickers,
    use_embeddings=False,
    use_sentiment=False,
    use_emotion=False
):
    """
    Process data through a transformer model to extract embeddings or use sentiment/emotion.
    Uses tqdm to show progress during batch processing.
    
    Args:
        model (torch.nn.Module): Transformer model for embeddings.
        loader (DataLoader): DataLoader to provide batches.
        device (torch.device): Device to perform computation.
        tickers (list): List of stock tickers.
        use_embeddings (bool): If True, extract embeddings from the model.
        use_sentiment (bool): If True, use sentiment vectors instead.
        use_emotion (bool): If True, use emotion vectors instead.
    
    Returns:
        torch.Tensor: Processed features (either embeddings or sentiment/emotion vectors).
        dict: Regression targets per ticker.
        dict: Classification labels per ticker.
    """
    model.eval()

    all_features = []
    all_regression_targets = {t: [] for t in tickers}
    all_classification_labels = {t: [] for t in tickers}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing Batches", unit="batch"):
            for t in tickers:
                all_regression_targets[t].append(batch["regression_targets"][t])
                all_classification_labels[t].append(batch["classification_labels"][t])

            if use_embeddings:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                B, num_chunks, max_len = input_ids.shape
                flat_input_ids = input_ids.view(-1, max_len)
                flat_attention_mask = attention_mask.view(-1, max_len)

                outputs = model(flat_input_ids, attention_mask=flat_attention_mask)
                chunk_embeddings = outputs.last_hidden_state.mean(dim=1)
                chunk_embeddings = chunk_embeddings.view(B, num_chunks, -1).mean(dim=1)

                all_features.append(chunk_embeddings.cpu())
            else:
                if use_emotion:
                    if "emotion" not in batch:
                        raise KeyError("[ERROR] 'emotion' key missing in batch! Available keys:", batch.keys())
                    all_features.append(batch["emotion"])
                elif use_sentiment:
                    if "sentiment" not in batch:
                        raise KeyError("[ERROR] 'sentiment' key missing in batch! Available keys:", batch.keys())
                    all_features.append(batch["sentiment"])
                else:
                    raise ValueError("No embeddings, no sentiment, no emotion. No representation defined.")

    all_features = torch.cat(all_features, dim=0)

    for t in tickers:
        all_regression_targets[t] = torch.cat(all_regression_targets[t], dim=0)
        all_classification_labels[t] = torch.cat(all_classification_labels[t], dim=0)

    return all_features, all_regression_targets, all_classification_labels


def build_features_and_targets(
    model,
    loader,
    device,
    use_embeddings=False,
    use_sentiment=False,
    use_emotion=False
):
    """
    A 'tickerless' version of build_features_and_targets for datasets (e.g. Yelp) with only one target set.
    Expects each batch has:
        batch["regression_targets"]  => Tensor
        batch["classification_labels"] => Tensor
    Returns:
        all_features (Tensor)
        all_regression_targets (dict with one key "ALL")
        all_classification_labels (dict with one key "ALL")
    """
    model.eval()

    all_features = []
    all_regression_targets = {"ALL": []}
    all_classification_labels = {"ALL": []}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing Batches", unit="batch"):
            all_regression_targets["ALL"].append(batch["regression_targets"]["ALL"])
            all_classification_labels["ALL"].append(batch["classification_labels"]["ALL"])

            if use_embeddings:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                B, num_chunks, max_len = input_ids.shape
                flat_input_ids = input_ids.view(-1, max_len)
                flat_attention_mask = attention_mask.view(-1, max_len)

                outputs = model(flat_input_ids, attention_mask=flat_attention_mask)
                chunk_embeddings = outputs.last_hidden_state.mean(dim=1)
                chunk_embeddings = chunk_embeddings.view(B, num_chunks, -1).mean(dim=1)

                all_features.append(chunk_embeddings.cpu())
            else:
                if use_emotion:
                    if "emotion" not in batch:
                        raise KeyError("[ERROR] 'emotion' key missing in batch!")
                    all_features.append(batch["emotion"])
                elif use_sentiment:
                    if "sentiment" not in batch:
                        raise KeyError("[ERROR] 'sentiment' key missing in batch!")
                    all_features.append(batch["sentiment"])
                else:
                    raise ValueError("No embeddings, sentiment, or emotion requested.")

    all_features = torch.cat(all_features, dim=0)

    all_regression_targets["ALL"] = torch.cat(all_regression_targets["ALL"], dim=0)
    all_classification_labels["ALL"] = torch.cat(all_classification_labels["ALL"], dim=0)

    return all_features, all_regression_targets, all_classification_labels