import argparse
import os
import joblib
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torch import nn, optim
import datetime
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

### Importing the datasets and model
from data.dataset_loader import UciOccupancyDataset
from data.dataset_loader import Lab42Dataset
from data.dataset_loader import load_lab42_from_influxdb, add_contextual_features
from utils.train import train_model, evaluate_model
from utils.save_metrics import save_metrics_to_csv, save_metrics_to_json

### Importing model class and definitions
from models.randomforest import RandomForestModel
from models.lstm import LSTMModel
from models.transformer import TransformerModel
from models.hybrid import HybridLSTMTransformerModel

def get_model(name, input_dim, dropout, seq_len=None, lstm_path=None, transformer_path=None):
    """
    Helper function to return the appropriate model based on the name argument.

    :param name: Name of the model to return. Options are 'randomforest', 'lstm', 'transformer', or 'hybrid'.
    :param input_dim: Input dimension of the model, which is the number of features in the input data.
    :param dropout: Dropout rate for the model, used for regularization.
    :param seq_len: Sequence length for LSTM and Transformer models. Not used for RandomForest.
    :param lstm_path: Path to the pre-trained LSTM model weights. Only used for hybrid model.
    :param transformer_path: Path to the pre-trained Transformer model weights. Only used for hybrid model.
    :return: An instance of the specified model class.
    """
    if name == 'randomforest':
        return RandomForestModel(input_dim, seq_len=seq_len)
    elif name == 'lstm':
        # LSTM model with 64 hidden units and 2 layers
        # 64 is used since dataset is of moderate size
        # 2 layers prevents underfitting and is a good starting point
        return LSTMModel(input_dim, hidden_dim=64, num_layers=2, dropout=dropout)
    elif name == 'transformer':
        # Transformer model with 64 dimensions, 4 attention heads, and 2 encoder layers
        # 64 is used since dataset is of moderate size, matching the LSTM hidden size
        # 4 attention heads is a good starting point for moderate size datasets and splits d_model into 4 chunks of 16 each
        # 2 encoder blocks for shallow sequence modelling
        return TransformerModel(input_dim, d_model=64, nhead=4, num_encoder_layers=2, dropout=dropout)
    elif name == 'hybrid':
        # Load individual models
        lstm_model = LSTMModel(input_dim, hidden_dim=64, num_layers=2, dropout=dropout)
        transformer_model = TransformerModel(input_dim, d_model=64, nhead=4, num_encoder_layers=2, dropout=dropout)

        # Load weights
        lstm_model.load_state_dict(torch.load(f"{lstm_path}.pt", map_location='cpu'))
        transformer_model.load_state_dict(torch.load(f"{transformer_path}.pt", map_location='cpu'))

        # Create hybrid model
        return HybridLSTMTransformerModel(lstm_model, transformer_model)
    else:
        raise ValueError(f"Unsupported model: {name}")

def make_loader(dataset, batch_size):
    """
    Create a DataLoader for the given dataset.

    :param dataset: Dataset to create a DataLoader for.
    :param batch_size: Batch size for the DataLoader.
    :return: DataLoader instance for the dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

def run_uci(args, model, device, loss_fn):
    """
    Run the models on UCI dataset only

    :param args: Parse arguments
    :param model: Model to train
    :param device: Device to use (CPU or GPU)
    :param loss_fn: Loss function to use
    """

    print("Running UCI Dataset only training")

    # Load UCI dataset splits
    uci_train_df = pd.read_csv(args.uci_train_csv)
    uci_val_df = pd.read_csv(args.uci_val_csv)
    uci_test_df = pd.read_csv(args.uci_test_csv)

    # Rename to match LAB42
    rename_map = {
        "CO2": "airquality",
        "Light": "light",

    }
    uci_train_df.rename(columns=rename_map, inplace=True)
    uci_val_df.rename(columns=rename_map, inplace=True)
    uci_test_df.rename(columns=rename_map, inplace=True)
    uci_train_df["_time"] = pd.to_datetime(uci_train_df["date"])
    uci_val_df["_time"] = pd.to_datetime(uci_val_df["date"])
    uci_test_df["_time"] = pd.to_datetime(uci_test_df["date"])

    # Add contextual features
    normalize_features = args.model in ["lstm", "transformer", "hybrid"]
    uci_train_df = add_contextual_features(uci_train_df, normalize=normalize_features)
    uci_val_df = add_contextual_features(uci_val_df, normalize=normalize_features)
    uci_test_df = add_contextual_features(uci_test_df, normalize=normalize_features)

    # Create datasets
    uci_train_dataset = UciOccupancyDataset(uci_train_df, args.features, args.uci_label_col, args.seq_len, include_capacity=False)
    uci_val_dataset = UciOccupancyDataset(uci_val_df, args.features, args.uci_label_col, args.seq_len, include_capacity=False)
    uci_test_dataset = UciOccupancyDataset(uci_test_df, args.features, args.uci_label_col, args.seq_len, include_capacity=False)

    # Create DataLoaders
    uci_train_loader = make_loader(uci_train_dataset, args.batch_size)
    uci_val_loader = make_loader(uci_val_dataset, args.batch_size)
    uci_test_loader = make_loader(uci_test_dataset, args.batch_size)

    # Create optimizer
    if isinstance(model, torch.nn.Module):
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        labels = uci_train_df[args.uci_label_col].values
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
        class_weights = torch.tensor([weights[1]/weights[0]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        optimizer = None  # RandomForest doesn't use an optimizer

    # Train the model
    train_model(
        model=model,
        train_dataloader=uci_train_loader,
        val_dataloader=uci_val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        grad_clip=args.grad_clip,
        save_path=args.save_path
    )

    if isinstance(model, torch.nn.Module):
        # Load the best model state for PyTorch models
        model.load_state_dict(torch.load(args.save_path + ".pt"))

    # Evaluate the model
    uci_metrics = evaluate_model(model, uci_test_loader, loss_fn, device, split_name="test")
    print(f"UCI Test Metrics: {uci_metrics}")

    # Save metrics to JSON and CSV files
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    uci_metrics_path_json = f"{args.metrics_path}_{current_time}.json"
    uci_metrics_path_csv = f"{args.metrics_path}_{current_time}.csv"
    save_metrics_to_json(uci_metrics, uci_metrics_path_json)
    save_metrics_to_csv(uci_metrics, uci_metrics_path_csv)


def run_lab42(args, model, device, loss_fn):
    """
    Run the models on Lab42 dataset only. Model might be pretrained with UCI dataset

    :param loss_fn: Loss function to use
    :param args: Parse arguments
    :param model: Model to train
    :param device: Device to use (CPU or GPU)
    """

    print("Running Lab42 Dataset only training or finetuning on pretrained model " + args.pretrain_path)
    torch_models = ["lstm", "transformer", "hybrid"]

    # Load Lab42 dataset splits
    lab42_train_df = load_lab42_from_influxdb(
        url=args.influx_url,
        token=args.influx_token,
        org=args.influx_org,
        bucket=args.influx_bucket,
        start=args.lab42_train_start,
        stop=args.lab42_train_stop
    )

    lab42_val_df = load_lab42_from_influxdb(
        url=args.influx_url,
        token=args.influx_token,
        org=args.influx_org,
        bucket=args.influx_bucket,
        start=args.lab42_val_start,
        stop=args.lab42_val_stop
    )

    lab42_train_val_df = load_lab42_from_influxdb(
        url=args.influx_url,
        token=args.influx_token,
        org=args.influx_org,
        bucket=args.influx_bucket,
        start=args.lab42_train_start,
        stop=args.lab42_val_stop
    )

    lab42_test_df = load_lab42_from_influxdb(
        url=args.influx_url,
        token=args.influx_token,
        org=args.influx_org,
        bucket=args.influx_bucket,
        start=args.lab42_test_start,
        stop=args.lab42_test_stop
    )

    # Drop any missing values
    lab42_train_df.dropna()
    lab42_val_df.dropna()
    lab42_train_val_df.dropna(inplace=True)
    lab42_test_df.dropna(inplace=True)

    # Filter by room if specified
    if args.room_filter:
        lab42_train_df = lab42_train_df[lab42_train_df["room_number"].isin(args.room_filter)]
        lab42_val_df = lab42_val_df[lab42_val_df["room_number"].isin(args.room_filter)]
        lab42_train_val_df = lab42_train_val_df[lab42_train_val_df["room_number"].isin(args.room_filter)]
        lab42_test_df = lab42_test_df[lab42_test_df["room_number"].isin(args.room_filter)]

    # Add contextual features
    normalize_features = args.model in torch_models
    lab42_train_df = add_contextual_features(lab42_train_df, normalize=normalize_features)
    lab42_val_df = add_contextual_features(lab42_val_df, normalize=normalize_features)
    lab42_train_val_df = add_contextual_features(lab42_train_val_df, normalize=normalize_features)
    lab42_test_df = add_contextual_features(lab42_test_df, normalize=normalize_features)

    # Split the train_val dataset into train and val using StratifiedShuffleSplit for torch models
    if args.model in torch_models:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        X = lab42_train_val_df.drop(columns=[args.lab42_label_col])
        y = lab42_train_val_df[args.lab42_label_col]
        train_idx, val_idx = next(sss.split(X, y))
        lab42_train_df = lab42_train_val_df.iloc[train_idx].reset_index(drop=True)
        lab42_val_df = lab42_train_val_df.iloc[val_idx].reset_index(drop=True)

    # Create datasets
    lab42_train_dataset = Lab42Dataset(lab42_train_df, args.features, args.lab42_label_col, args.seq_len, include_capacity=args.include_capacity)
    lab42_val_dataset = Lab42Dataset(lab42_val_df, args.features, args.lab42_label_col, args.seq_len, include_capacity=args.include_capacity)
    lab42_test_dataset = Lab42Dataset(lab42_test_df, args.features, args.lab42_label_col, args.seq_len, include_capacity=args.include_capacity)

    # Create DataLoaders
    lab42_train_loader = make_loader(lab42_train_dataset, args.batch_size)
    lab42_val_loader = make_loader(lab42_val_dataset, args.batch_size)
    lab42_test_loader = make_loader(lab42_test_dataset, args.batch_size)

    if isinstance(model, torch.nn.Module):
        # Create optimizer for PyTorch models
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        labels = lab42_train_df[args.lab42_label_col].values
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
        class_weights = torch.tensor([weights[1]/weights[0]], dtype=torch.float32).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        optimizer = None  # RandomForest doesn't use an optimizer

    # Load the best model state for models
    if isinstance(model, torch.nn.Module): # For PyTorch models
        pretrain_path = args.pretrain_path + ".pt"
    else: # For RandomForest, we use joblib
        pretrain_path = args.pretrain_path + ".joblib"

    if os.path.exists(pretrain_path):
        if isinstance(model, torch.nn.Module): # If it's a PyTorch model
            model.load_state_dict(torch.load(pretrain_path))
            print(f"Loaded pretrained weights from {pretrain_path}")
        else: # If it's a RandomForest model
            model.model = joblib.load(pretrain_path)
            print(f"Loaded pretrained weights from {pretrain_path}")
    else: # If the pretrain path does not exist, we train from scratch
        print(f"Pretrain path {pretrain_path} not found. Training from scratch.")

    # Train the model
    train_model(
        model=model,
        train_dataloader=lab42_train_loader,
        val_dataloader=lab42_val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        grad_clip=args.grad_clip,
        save_path=args.save_path
    )

    # Load the best model state for PyTorch models
    lab42_metrics = evaluate_model(model, lab42_test_loader, loss_fn, device)
    print(f"Lab42 Test Metrics: {lab42_metrics}")

    # Save metrics to JSON and CSV files
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lab42_metrics_path_json = f"{args.metrics_path}_{current_time}.json"
    lab42_metrics_path_csv = f"{args.metrics_path}_{current_time}.csv"
    save_metrics_to_json(lab42_metrics, lab42_metrics_path_json)
    save_metrics_to_csv(lab42_metrics, lab42_metrics_path_csv)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train an occupancy prediction model.')

    # Model and dataset options
    parser.add_argument("--model", type=str, required=True, choices=['randomforest', 'lstm', 'transformer', 'hybrid'],
                        help="Model to train: logreg, lstm, transformer, or hybrid")
    parser.add_argument("--label_col", type=str, default='Occupancy', help="Name of the label column in the dataset")
    parser.add_argument("--features", nargs='+', default=['CO2', 'light'], help="Sensor features to use")
    parser.add_argument("--include_capacity", default=False, action="store_true", help="Include capacity as a static feature")
    parser.add_argument("--simulate_strategy", type=str, choices=["mask", "shift"], default="mask", help="Strategy to simulate delayed data")
    parser.add_argument("--simulate_random", action="store_true", help="Simulate random delayed data")
    parser.add_argument("--seq_len", type=int, default=60, help="Number of timesteps in each input sequence")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the model")
    parser.add_argument("--target_interval", type=str, default="1min", help="Target interval for the model")
    parser.add_argument("--lstm_path", type=str, default=None, help="Path to the LSTM model weights")
    parser.add_argument("--transformer_path", type=str, default=None, help="Path to the Transformer model weights")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for the optimizer")
    parser.add_argument("--save_path", type=str, default='best_model.pth', help="Path to save the trained model")
    parser.add_argument("--grad_clip", type=float, default=0.5, help="Gradient clipping value")
    parser.add_argument("--pretrain_path", type=str, default="pretrained_model.pt")
    parser.add_argument("--metrics_path", type=str, default="metrics.json", help="Path to save metrics")

    # UCI dataset specific parameters
    parser.add_argument("--uci_train_csv", type=str, help="Path to UCI training CSV file")
    parser.add_argument("--uci_test_csv", type=str, help="Path to UCI testing CSV file")
    parser.add_argument("--uci_val_csv", type=str, help="Path to UCI validation CSV file")
    parser.add_argument("--uci_label_col", type=str, default='Occupancy', help="Name of the label column in the UCI dataset")

    # Lab42 dataset specific parameters
    parser.add_argument("--lab42_label_col", type=str, default="Occupancy")
    parser.add_argument("--influx_url", type=str, default=None)
    parser.add_argument("--influx_token", type=str, default=None)
    parser.add_argument("--influx_org", type=str, default=None)
    parser.add_argument("--influx_bucket", type=str, default=None)
    parser.add_argument("--lab42_train_start", type=str, default=None)
    parser.add_argument("--lab42_train_stop", type=str, default=None)
    parser.add_argument("--lab42_val_start", type=str, default=None)
    parser.add_argument("--lab42_val_stop", type=str, default=None)
    parser.add_argument("--lab42_test_start", type=str, default=None)
    parser.add_argument("--lab42_test_stop", type=str, default=None)
    parser.add_argument("--room_filter", nargs="+", default=None, help="Room number to filter the Lab42 dataset")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(args.features) + (1 if args.include_capacity else 0)  # Number of features + room capacity if included

    # Initialize the model
    model = get_model(args.model, input_dim, args.dropout, seq_len=args.seq_len, lstm_path=args.lstm_path, transformer_path=args.transformer_path)

    # Only move model to device if it's a PyTorch model
    if isinstance(model, torch.nn.Module):
        model = model.to(device)

    loss_fn = nn.BCELoss()

    if args.uci_train_csv: # UCI dataset is specified
        run_uci(args, model, device, loss_fn)

    if args.lab42_train_start: # Lab42 dataset is specified
        run_lab42(args, model, device, loss_fn)

if __name__ == "__main__":
    main()
