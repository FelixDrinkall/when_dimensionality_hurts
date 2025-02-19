import torch.nn as nn
from torch.utils.data import DataLoader
from utils import EarlyStopping
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import wandb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)

def symmetric_mape(y_true, y_pred, epsilon=1e-8):
    """
    Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (numpy.ndarray or torch.Tensor): True values.
        y_pred (numpy.ndarray or torch.Tensor): Predicted values.
        epsilon (float): Small value to prevent division by zero. Default is 1e-8.

    Returns:
        float: SMAPE value in percentage.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon

    smape = 100 * np.mean(numerator / denominator)
    return smape


def compute_regression_metrics(y_true, y_pred, y_train_mean):
    """
    Compute various regression error metrics.

    Args:
        y_true (np.ndarray): Ground truth targets.
        y_pred (np.ndarray): Predicted targets.
        y_train_mean (float): Mean of the training targets used for naive forecasting.

    Returns:
        dict: Dictionary containing all error metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    def huber(y_true, y_pred):
        delta = 1.0
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error**2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))
    
    # Compute metrics
    metrics = {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Huber": huber(y_true, y_pred)
    }
    
    return metrics

def compute_classification_metrics(y_true, preds, probs=None):
    """
    Compute various classification metrics.

    Args:
        y_true (np.ndarray): Ground truth labels.
        preds (np.ndarray): Predicted labels (class indices).
        probs (np.ndarray, optional): Predicted probabilities (for ROC-AUC and log-loss).

    Returns:
        dict: Dictionary containing all classification metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, preds),
        "Balanced Accuracy": balanced_accuracy_score(y_true, preds),
        "Precision (Macro)": precision_score(y_true, preds, average="macro", zero_division=0),
        "Recall (Macro)": recall_score(y_true, preds, average="macro", zero_division=0),
        "F1 (Macro)": f1_score(y_true, preds, average="macro"),
        "Precision (Weighted)": precision_score(y_true, preds, average="weighted", zero_division=0),
        "Recall (Weighted)": recall_score(y_true, preds, average="weighted", zero_division=0),
        "F1 (Weighted)": f1_score(y_true, preds, average="weighted"),
        "Matthews Correlation Coefficient": matthews_corrcoef(y_true, preds),
    }

    if probs is not None:
        try:
            if probs.ndim == 2:
                if probs.shape[1] > 2:
                    metrics["ROC-AUC (Macro)"] = roc_auc_score(
                        y_true, probs, multi_class="ovr", average="macro"
                    )
                    metrics["Log-Loss"] = log_loss(y_true, probs)
                elif probs.shape[1] == 2:
                    positive_probs = probs[:, 1]
                    metrics["ROC-AUC"] = roc_auc_score(y_true, positive_probs)
                    metrics["Log-Loss"] = log_loss(y_true, positive_probs)
                else:
                    metrics["ROC-AUC"] = roc_auc_score(y_true, probs)
                    metrics["Log-Loss"] = log_loss(y_true, probs)
            else:
                metrics["ROC-AUC"] = roc_auc_score(y_true, probs)
                metrics["Log-Loss"] = log_loss(y_true, probs)

        except Exception as e:
            print(f"Error computing ROC-AUC or Log-Loss: {e}")

    return metrics

class RandomForestModel:
    """
    A simple wrapper that unifies RandomForestRegressor and RandomForestClassifier
    under a single interface similar to PyTorch models.
    """
    def __init__(self, task="regression", n_estimators=100, max_depth=None, random_state=42):
        self.task = task
        if task == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.task == "classification":
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("predict_proba is not applicable to regression")


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate, num_classes=2, task="regression"):
        super().__init__()
        self.task = task
        out_dim = 1 if task == "regression" else num_classes
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )
        if task == "classification":
            self.activation = nn.Softmax(dim=1)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        if self.task == "classification":
            x = self.activation(x)
        return x


def train_and_evaluate_with_early_stopping(
        train_z, train_targets, 
        val_z, val_targets, 
        test_z, test_targets,
        args, 
        device, 
        results_file, 
        mlp_hidden_dim, 
        num_classes, 
        sampled_test_z=None, 
        sampled_test_targets=None, 
        task="regression", 
        combined=False,
        use_random_forest=False
    ):
    """
    Train and evaluate either a RandomForest model or an MLP with early stopping 
    using validation data.
    """
    if use_random_forest:
        print("Using RandomForest Model.")
        model = RandomForestModel(
            task=task,
            n_estimators=args.n_estimators if hasattr(args, 'n_estimators') else 100,
            max_depth=args.max_depth if hasattr(args, 'max_depth') else None,
            random_state=42
        )
    else:
        print("Using MLP Model.")
        model = MLPModel(
            input_dim=train_z.size(1),
            hidden_dim=mlp_hidden_dim,
            dropout_rate=args.dropout_prob,
            num_classes=num_classes,
            task=task
        ).to(device)

    criterion = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()

    if not use_random_forest:
        optimizer = torch.optim.AdamW(model.parameters(), 
                                      lr=args.learning_rate, 
                                      weight_decay=args.weight_decay)

    train_reg_loader = DataLoader(list(zip(train_z, train_targets)), 
                                  batch_size=args.batch_size, 
                                  shuffle=True)
    val_reg_loader = DataLoader(list(zip(val_z, val_targets)), 
                                batch_size=args.batch_size, 
                                shuffle=False)
    test_reg_loader = DataLoader(list(zip(test_z, test_targets)), 
                                 batch_size=args.batch_size, 
                                 shuffle=False)
    if sampled_test_z is not None:
        sampled_test_reg_loader = DataLoader(list(zip(sampled_test_z, sampled_test_targets)), 
                                             batch_size=args.batch_size, 
                                             shuffle=False)

    y_train_mean = train_targets.mean().item() if task == "regression" else None

    if use_random_forest:
        train_z_np = train_z.cpu().numpy() if torch.is_tensor(train_z) else train_z
        train_targets_np = train_targets.cpu().numpy() if torch.is_tensor(train_targets) else train_targets
        
        model.fit(train_z_np, train_targets_np)

        val_z_np = val_z.cpu().numpy() if torch.is_tensor(val_z) else val_z
        val_targets_np = val_targets.cpu().numpy() if torch.is_tensor(val_targets) else val_targets
        val_preds = model.predict(val_z_np)
        test_z_np = test_z.cpu().numpy() if torch.is_tensor(test_z) else test_z
        test_targets_np = test_targets.cpu().numpy() if torch.is_tensor(test_targets) else test_targets

        # Predict
        preds_ticker_test = model.predict(test_z_np)
        if combined:
            preds_ticker_test = np.squeeze(preds_ticker_test, axis=-1) if preds_ticker_test.ndim > 1 else preds_ticker_test
            test_targets_np = np.squeeze(test_targets_np, axis=-1) if test_targets_np.ndim > 1 else test_targets_np

        # Compute metrics
        if task == "classification":
            ticker_probs = model.predict_proba(test_z_np)
            ticker_preds = preds_ticker_test
            ticker_metrics = compute_classification_metrics(test_targets_np, ticker_preds, ticker_probs)
        else:
            ticker_metrics = compute_regression_metrics(test_targets_np, preds_ticker_test, y_train_mean)

        # Evaluate on combined sampled test set
        preds_combined_test = []
        if not combined and not args.yelp and not args.huggingface_dataset:
            if sampled_test_z is not None:
                sampled_test_z_np = sampled_test_z.cpu().numpy() if torch.is_tensor(sampled_test_z) else sampled_test_z
                preds_combined_test = model.predict(sampled_test_z_np)
            return ticker_metrics, preds_ticker_test, preds_combined_test, []  
        else:
            
            return ticker_metrics, preds_ticker_test, []

    else:
        # Early stopping mechanism
        early_stopping = EarlyStopping(patience=args.mlp_patience)
        val_losses = []
        best_model_state = None
        best_val_loss = float('inf')

        # Training
        for epoch in range(args.epochs_reg):
            model.train()
            total_loss = 0.0
            optimizer.zero_grad()
            for step, (z, y) in enumerate(train_reg_loader):
                z, y = z.to(device), y.to(device)
                if task == "classification":
                    y = y.long()

                preds = model(z)
                loss = criterion(preds, y.unsqueeze(-1) if task == "regression" else y)
                loss.backward()
                total_loss += loss.item()

                if (step + 1) % args.grad_acc_steps == 0 or (step + 1) == len(train_reg_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_reg_loader)

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for z, y in val_reg_loader:
                    z, y = z.to(device), y.to(device)
                    if task == "classification":
                        y = y.long()
                    preds = model(z)
                    val_loss += criterion(preds, y.unsqueeze(-1) if task == "regression" else y).item()

            val_loss /= len(val_reg_loader)
            val_losses.append(val_loss)

            wandb.log({"Epoch": epoch + 1, "Train Loss": avg_train_loss, "Validation Loss": val_loss})
            print(f"Epoch {epoch + 1}/{args.epochs_reg}, "
                  f"Train Loss: {avg_train_loss:.7f}, Val Loss: {val_loss:.7f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Load the best model
        if best_model_state:
            model.load_state_dict(best_model_state)

        # Eval on test set
        preds_ticker_test = []
        model.eval()
        with torch.no_grad():
            for z, _ in test_reg_loader:
                z = z.to(device)
                preds_ticker_test.append(model(z).cpu().numpy())

        preds_ticker_test = np.concatenate(preds_ticker_test, axis=0)

        if combined:
            preds_ticker_test = preds_ticker_test.squeeze(-1) if preds_ticker_test.ndim == 2 else preds_ticker_test
            test_targets = test_targets.squeeze(-1) if test_targets.ndim == 2 else test_targets
            
        if task == "classification":
            ticker_probs = torch.softmax(torch.tensor(preds_ticker_test), dim=1).numpy()
            ticker_preds = np.argmax(ticker_probs, axis=1)
            ticker_metrics = compute_classification_metrics(test_targets, ticker_preds, ticker_probs)
        else:
            ticker_metrics = compute_regression_metrics(test_targets, preds_ticker_test, y_train_mean)

        # Eval on combined test set
        if not combined:
            preds_combined_test = []
            if sampled_test_z is not None:
                with torch.no_grad():
                    for z, _ in sampled_test_reg_loader:
                        z = z.to(device)
                        preds_combined_test.append(model(z).cpu().numpy())
                preds_combined_test = np.concatenate(preds_combined_test)

            return ticker_metrics, preds_ticker_test, preds_combined_test, val_losses
        else:
            return ticker_metrics, preds_ticker_test, val_losses