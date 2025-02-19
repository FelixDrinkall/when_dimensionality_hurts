import json
from torch.utils.data import Dataset, Subset
import torch
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from models.embedding import build_features_and_targets, build_features_and_targets_tickers


class GenericHFDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        text_column,
        target_column,
        tokenizer,
        task="classification",
        label_shift=0,
    ):
        """
        Args:
            hf_dataset (Dataset):
                A Hugging Face dataset split (e.g., dataset["train"]).
            text_column (str):
                The column name containing text.
            target_column (str):
                The column name containing the label/target.
            tokenizer:
                A Hugging Face tokenizer (e.g. from AutoTokenizer.from_pretrained(...)).
            task (str):
                "classification" or "regression".
            label_shift (int):
                If your labels start at 1 (like Yelp's stars), and you
                want them to start at 0, set this to 1, etc. Otherwise, set to 0.
        """
        self.dataset = hf_dataset
        self.text_column = text_column
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.task = task
        self.label_shift = label_shift

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        text = row[self.text_column]
        raw_label = row[self.target_column]

        if self.task == "classification":
            label = int(raw_label) - self.label_shift  
            classification_label = torch.tensor(label, dtype=torch.long)
            regression_label = float(label)  
        else:  
            val = float(raw_label)
            regression_label = torch.tensor(val, dtype=torch.float)
            classification_label = torch.tensor(0, dtype=torch.long)

        tokens = self.tokenizer(
            text, truncation=False, return_tensors="pt"  
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        max_len = self.tokenizer.model_max_length
        if len(input_ids) > max_len:
            chunks, chunk_masks = [], []
            for i in range(0, len(input_ids), max_len):
                chunk_ids = input_ids[i : i + max_len]
                chunk_mask = attention_mask[i : i + max_len]
                if len(chunk_ids) < max_len:
                    pad_len = max_len - len(chunk_ids)
                    chunk_ids = torch.cat([chunk_ids, torch.zeros(pad_len, dtype=torch.long)])
                    chunk_mask = torch.cat([chunk_mask, torch.zeros(pad_len, dtype=torch.long)])
                chunks.append(chunk_ids)
                chunk_masks.append(chunk_mask)
            input_ids = torch.stack(chunks, dim=0)        
            attention_mask = torch.stack(chunk_masks, dim=0) 
        else:
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            input_ids = input_ids.unsqueeze(0)       
            attention_mask = attention_mask.unsqueeze(0)  

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "regression_targets": {"ALL": regression_label},
            "classification_labels": {"ALL": classification_label},
        }

def balance_classes(targets):
    """
    Balances a *single* class distribution by undersampling 
    the majority class. Skips any None values in `targets`.

    :param targets: list/array of class labels (0 or 1). May contain None.
    :return: A list of indices (relative to the input array) that form a balanced subset.
    """
    targets = np.array(targets, dtype=object)
    valid_indices = [i for i, v in enumerate(targets) if v is not None]
    if not valid_indices:
        return []

    valid_targets = targets[valid_indices]
    valid_targets = valid_targets.astype(int)

    labels, counts = np.unique(valid_targets, return_counts=True)
    if len(labels) < 2:
        return valid_indices

    min_count = counts.min()
    balanced_indices_relative = []
    for label in labels:
        label_positions = np.where(valid_targets == label)[0]
        chosen = np.random.choice(label_positions, min_count, replace=False)
        balanced_indices_relative.extend(chosen)

    np.random.shuffle(balanced_indices_relative)
    balanced_indices = [valid_indices[i] for i in balanced_indices_relative]
    return balanced_indices


def balance_binary_train_set(train_dataset, args):
    """
    Balances each ticker's 0/1 distribution separately, 
    then merges (unions) those subsets.

    This is intended for the 'fin_returns' dataset with --single_company=True,
    where each sample mentions exactly *one* ticker.

    Steps:
        1) For each ticker T:
           a) Identify all samples that belong to T (non-None label).
           b) Use 'balance_classes' to undersample so that #Up == #Down for T.
           c) Collect these balanced indices in a set.

        2) Union all per-ticker sets (because single-company means no overlap).

        3) Return a single Subset containing all balanced samples.
    """
    if args.yelp:
        labels = train_dataset.classification_labels["ALL"]  
        labels_np = np.array(labels, dtype=object)
        balanced_indices = balance_classes(labels_np)
        return Subset(train_dataset, balanced_indices)

    final_indices = set()
    for i, ticker in enumerate(train_dataset.tickers):
        labels = train_dataset.classification_labels[ticker]
        labels_np = np.array(labels, dtype=object)
        balanced_rel_indices = balance_classes(labels_np)
        balanced_rel_indices = set(balanced_rel_indices)
        if i == 0:
            final_indices = balanced_rel_indices
        else:
            final_indices = final_indices.union(balanced_rel_indices)

    balanced_indices = sorted(final_indices)
    balanced_subset = Subset(train_dataset, balanced_indices)
    return balanced_subset

def equal_chunks_collate(batch):
    """
    Custom collate function to handle variable-length chunked sequences.
    
    This function ensures that:
    - Input sequences (`input_ids` and `attention_mask`) are correctly padded.
    - Regression and classification targets are stacked.
    - Emotion and sentiment vectors are included if they exist.
    
    Args:
        batch (list): A batch of data samples, each being a dictionary.
    
    Returns:
        dict: Batched data with padded inputs, stacked targets, and optional sentiment/emotion.
    """
    batch_input_ids = []
    batch_attention_masks = []
    batch_regression_targets = {}
    batch_classification_labels = {}
    batch_sentiments = []
    batch_emotions = []

    tickers = batch[0]['regression_targets'].keys()

    max_chunks = max(sample['input_ids'].size(0) for sample in batch)
    max_length = max(sample['input_ids'].size(1) for sample in batch)

    for sample in batch:
        input_ids = sample['input_ids']
        attention_mask = sample['attention_mask']

        if input_ids.size(1) < max_length:
            pad_size = (0, max_length - input_ids.size(1))
            input_ids = torch.nn.functional.pad(input_ids, pad_size, value=0)
            attention_mask = torch.nn.functional.pad(attention_mask, pad_size, value=0)

        if input_ids.size(0) < max_chunks:
            pad_size = (0, 0, 0, max_chunks - input_ids.size(0))
            input_ids = torch.nn.functional.pad(input_ids, pad_size, value=0)
            attention_mask = torch.nn.functional.pad(attention_mask, pad_size, value=0)

        batch_input_ids.append(input_ids)
        batch_attention_masks.append(attention_mask)

        if "sentiment" in sample:
            batch_sentiments.append(sample["sentiment"])
        if "emotion" in sample:
            batch_emotions.append(sample["emotion"])

    batch_input_ids = torch.stack(batch_input_ids, dim=0)  
    batch_attention_masks = torch.stack(batch_attention_masks, dim=0)

    for ticker in tickers:
        batch_regression_targets[ticker] = torch.stack([sample['regression_targets'][ticker] for sample in batch])
        batch_classification_labels[ticker] = torch.stack([sample['classification_labels'][ticker] for sample in batch])

    batch_sentiments = torch.stack(batch_sentiments) if batch_sentiments else None
    batch_emotions = torch.stack(batch_emotions) if batch_emotions else None

    batch_dict = {
        'input_ids': batch_input_ids,
        'attention_mask': batch_attention_masks,
        'regression_targets': batch_regression_targets,
        'classification_labels': batch_classification_labels,
        'emotion': batch_emotions,
        'sentiment': batch_sentiments,

    }

    return batch_dict

def merge_datasets(years, outlets, tokenizer, single_company, base_dir="processed_data_final"):
    """
    Merge datasets dynamically based on years and outlets.

    Args:
        years (list): List of years to include (e.g., [2020, 2021]).
        outlets (list): List of outlets to include (e.g., ["finance.yahoo.com"]).
        tokenizer (Tokenizer): Pre-trained tokenizer.
        base_dir (str): Base directory for processed data.

    Returns:
        TextRegressionDataset: Combined dataset for all specified years and outlets.
    """
    merged_data = []
    for outlet in tqdm(outlets, desc="Outlets"):
        for year in tqdm(years, desc="years"):
            file_path = os.path.join(base_dir, outlet, f"{year}_processed.json")
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue
            with open(file_path, 'r') as f:
                merged_data.extend(json.load(f))
            
    print("Processing dataset")
    return TextRegressionDataset.from_data(merged_data, tokenizer, single_company)

class TextRegressionDataset(Dataset):
    def __init__(self, json_path, tokenizer, single_company=False):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = tokenizer.model_max_length
        self.single_company = single_company

        self.dates = self._parse_dates(self.data)
        self.sentiments = [item['sentiment'] for item in self.data]
        self.emotions   = [item['emotion'] for item in self.data]
        self.tickers = self._extract_unique_tickers()
        self.data = self._filter_data()
        self.texts = [item['maintext'] for item in self.data]
        self.regression_targets = self._extract_targets("next_day_return")
        self.classification_labels = self._extract_targets("direction", classification=True)

    def _parse_dates(self, data_list):
        """Parse 'date_publish' for each item into a datetime object. 
        If missing or invalid, fallback to a default or None."""
        dates = []
        for item in data_list:
            date_str = item.get("date_publish", None)
            if date_str:
                try:
                    parsed_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    parsed_date = None
            else:
                parsed_date = None
            dates.append(parsed_date)
        return dates

    def _sort_by_date(self):
        """Sort self.data and self.dates simultaneously by ascending date."""
        zipped = list(zip(self.data, self.dates))
        zipped.sort(key=lambda x: x[1] if x[1] is not None else datetime.min)
        self.data, self.dates = zip(*zipped)
        self.data = list(self.data)
        self.dates = list(self.dates)

    def _extract_unique_tickers(self):
        """Extract all unique tickers from the dataset."""
        tickers = set()
        for item in self.data:
            tickers.update(item.get("mentioned_companies", []))
        return sorted(tickers)

    def _filter_data(self):
        """Filter out entries with NaN or None in regression or classification targets and enforce single_company if required."""
        filtered_data = []
        filtered_dates = []

        for i, item in enumerate(self.data):
            if self.single_company and len(item.get("mentioned_companies", [])) != 1:
                continue

            valid = True
            for ticker in self.tickers:
                reg_key = f"next_day_return_{ticker}"
                class_key = f"direction_{ticker}"
                if reg_key in item:
                    reg_value = item[reg_key]
                    if reg_value is None:
                        valid = False
                        break
                    if isinstance(reg_value, float) and torch.isnan(torch.tensor(reg_value)):
                        valid = False
                        break
                if class_key in item:
                    class_value = item[class_key]
                    if class_value is None:
                        valid = False
                        break
                    if isinstance(class_value, float) and torch.isnan(torch.tensor(class_value)):
                        valid = False
                        break

            if valid:
                filtered_data.append(item)
                filtered_dates.append(self.dates[i])

        self.dates = filtered_dates
        return filtered_data

    def _extract_targets(self, prefix, classification=False):
        """Extract regression targets or classification labels for all tickers."""
        targets = {ticker: [] for ticker in self.tickers}
        for item in self.data:
            for ticker in self.tickers:
                key = f"{prefix}_{ticker}"
                if key in item:
                    if classification:
                        targets[ticker].append(1 if item[key] == "Up" else 0)
                    else:
                        try:
                            targets[ticker].append(float(item[key]))
                        except (ValueError, TypeError):
                            targets[ticker].append(0.0)
                else:
                    targets[ticker].append(None)
        return targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self._chunk_and_tokenize(text)

        regression_targets = {
            ticker: self._safe_tensor(self.regression_targets[ticker][idx], dtype=torch.float)
            for ticker in self.tickers
        }
        classification_labels = {
            ticker: self._safe_tensor(self.classification_labels[ticker][idx], dtype=torch.long)
            for ticker in self.tickers
        }

        s_dict = self.sentiments[idx]
        e_dict = self.emotions[idx]

        sentiment_vec = [
            s_dict.get("negative", 0.0),
            s_dict.get("neutral",  0.0),
            s_dict.get("positive", 0.0),
        ]
        emotion_vec = [
            e_dict.get("neutral", 0.0),
            e_dict.get("surprise", 0.0),
            e_dict.get("fear", 0.0),
            e_dict.get("anger", 0.0),
            e_dict.get("disgust", 0.0),
            e_dict.get("joy", 0.0),
            e_dict.get("sadness", 0.0),
        ]

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "regression_targets": regression_targets,
            "classification_labels": classification_labels,
            "sentiment": torch.tensor(sentiment_vec, dtype=torch.float),
            "emotion":   torch.tensor(emotion_vec,   dtype=torch.float),
        }

    def _chunk_and_tokenize(self, text):
        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        if len(input_ids) > self.max_length:
            chunks = [input_ids[i:i + self.max_length] for i in range(0, len(input_ids), self.max_length)]
            attention_chunks = [attention_mask[i:i + self.max_length] for i in range(0, len(input_ids), self.max_length)]

            if len(chunks[-1]) < self.max_length:
                pad_length = self.max_length - len(chunks[-1])
                chunks[-1] = torch.cat([chunks[-1], torch.zeros(pad_length, dtype=torch.long)])
                attention_chunks[-1] = torch.cat([attention_chunks[-1], torch.zeros(pad_length, dtype=torch.long)])

            return {
                "input_ids": torch.stack(chunks),
                "attention_mask": torch.stack(attention_chunks),
            }
        else:
            return {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
            }

    def _safe_tensor(self, value, dtype):
        return torch.tensor(value, dtype=dtype) if value is not None else torch.tensor(0, dtype=dtype)

    @classmethod
    def from_data(cls, data, tokenizer, single_company=False):
        obj = cls.__new__(cls)
        obj.data = data
        obj.tokenizer = tokenizer
        obj.max_length = tokenizer.model_max_length
        obj.single_company = single_company
        obj.dates = obj._parse_dates(obj.data)
        obj.tickers = obj._extract_unique_tickers()
        obj.data = obj._filter_data()
        obj.texts = [item['maintext'] for item in obj.data]
        obj.sentiments = [item['sentiment'] for item in obj.data]
        obj.emotions   = [item['emotion'] for item in obj.data]
        obj.regression_targets = obj._extract_targets("next_day_return")
        obj.classification_labels = obj._extract_targets("direction", classification=True)
        return obj

    def sort_by_date(self):
        """Public method to allow external sorting by date if needed."""
        self._sort_by_date()


class YelpDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        """
        Args:
            texts (List[str]): List of review texts.
            labels (List[int]): Integer star ratings in [1..5].
            tokenizer: A Hugging Face tokenizer.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        star_label = self.labels[idx]

        tokens = self.tokenizer(text, truncation=False, return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        max_len = self.tokenizer.model_max_length
        if len(input_ids) > max_len:
            chunks, chunk_masks = [], []
            for i in range(0, len(input_ids), max_len):
                chunk_ids = input_ids[i : i + max_len]
                chunk_mask = attention_mask[i : i + max_len]
                if len(chunk_ids) < max_len:
                    pad_len = max_len - len(chunk_ids)
                    chunk_ids = torch.cat([chunk_ids, torch.zeros(pad_len, dtype=torch.long)])
                    chunk_mask = torch.cat([chunk_mask, torch.zeros(pad_len, dtype=torch.long)])
                chunks.append(chunk_ids)
                chunk_masks.append(chunk_mask)
            input_ids = torch.stack(chunks, dim=0)
            attention_mask = torch.stack(chunk_masks, dim=0)
        else:
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        classification_label = star_label - 1   # shift label to 0..4
        regression_label = float(star_label)    # convert label to float (1..5)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "regression_targets": {
                "ALL": torch.tensor(regression_label, dtype=torch.float)
            },
            "classification_labels": {
                "ALL": torch.tensor(classification_label, dtype=torch.long)
            },
        }



def get_transform_params(targets_array, standardize, normalize):
    """
    Returns the parameters needed for standardization or normalization
    based on the train set only.
    """
    if hasattr(targets_array, "numpy"):
        train_targets_np = targets_array.numpy()
    else:
        train_targets_np = np.array(targets_array)
    
    params = {}
    if standardize:
        mean_val = np.mean(train_targets_np)
        std_val = np.std(train_targets_np)
        if std_val < 1e-12:
            std_val = 1.0  
        params["mode"] = "standardize"
        params["mean"] = mean_val
        params["std"] = std_val
    elif normalize:
        min_val = np.min(train_targets_np)
        max_val = np.max(train_targets_np)
        if abs(max_val - min_val) < 1e-12:
            max_val = min_val + 1.0
        params["mode"] = "normalize"
        params["min"] = min_val
        params["max"] = max_val
    else:
        params["mode"] = "none"
    return params


def apply_transform(targets, params):
    """ Apply either standardization, normalization, or do nothing. """
    if hasattr(targets, "numpy"):
        targets_np = targets.numpy()
    else:
        targets_np = np.array(targets)
    
    if params["mode"] == "standardize":
        mean_val = params["mean"]
        std_val = params["std"]
        return (targets_np - mean_val) / std_val
    elif params["mode"] == "normalize":
        min_val = params["min"]
        max_val = params["max"]
        return (targets_np - min_val) / (max_val - min_val)
    else:
        return targets_np 

def inverse_transform(preds, params):
    """ Inverse transform back to original scale. """
    if params["mode"] == "standardize":
        return preds * params["std"] + params["mean"]
    elif params["mode"] == "normalize":
        return preds * (params["max"] - params["min"]) + params["min"]
    else:
        return preds


def create_train_val_test_representation(
    model,
    train_loader,
    val_loader,
    test_loader,
    train_dataset,
    val_dataset,
    test_dataset,
    device,
    tickers,
    args
):
    """
    Build final representation + targets for train/val/test.
    If 'tickers' is a non-empty list, we assume multi-ticker data
    and call build_features_and_targets (ticker-based).
    Otherwise, we call build_features_and_targets_tickerless.
    """

    use_embeddings = not (args.sentiment or args.emotion)
    use_sentiment  = args.sentiment
    use_emotion    = args.emotion

    def build_fn(loader):
        if tickers is not None and len(tickers) > 0:
            return build_features_and_targets_tickers(
                model=model,
                loader=loader,
                device=device,
                tickers=tickers,
                use_embeddings=use_embeddings,
                use_sentiment=use_sentiment,
                use_emotion=use_emotion
            )
        else:
            return build_features_and_targets(
                model=model,
                loader=loader,
                device=device,
                use_embeddings=use_embeddings,
                use_sentiment=use_sentiment,
                use_emotion=use_emotion
            )

    train_embedding, train_reg_targets, train_cls_targets = build_fn(train_loader)
    if args.task == "regression":
        train_targets = train_reg_targets
    else:
        train_targets = train_cls_targets

    val_embedding, val_reg_targets, val_cls_targets = build_fn(val_loader)
    if args.task == "regression":
        val_targets = val_reg_targets
    else:
        val_targets = val_cls_targets

    test_embedding, test_reg_targets, test_cls_targets = build_fn(test_loader)
    if args.task == "regression":
        test_targets = test_reg_targets
    else:
        test_targets = test_cls_targets

    return (
        train_embedding, train_targets,
        val_embedding,   val_targets,
        test_embedding,  test_targets
    )