import os
import copy

import torch
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_random_forest(df_long, test_size=0.2, random_state=42, n_jobs=None):
    """
    Trains a Random Forest classifier on time series data in long format.

    :param df_long: DataFrame in long format with columns ['id', 'time', 'value', 'label']
    :param test_size: Fraction of data to use as test set
    :param random_state: Random seed for reproducibility
    :param n_jobs: Number of parallel jobs (default: all cores)
    :return: trained classifier, LabelEncoder, selected feature names, (X_train, X_test, y_train, y_test)
    """

    if n_jobs is None:
        n_jobs = os.cpu_count()

    # Feature extraction
    X = extract_features(
        df_long,
        column_id="id",
        column_sort="time",
        column_value="value",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=n_jobs,
        disable_progressbar=True
    )

    # Labels
    y_raw = df_long.groupby("id").first()["label"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Impute missing values
    X = impute(X)

    # Select significant features
    X_selected = select_features(X, y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Define and train classifier
    clf = RandomForestClassifier(
        random_state=random_state,
        bootstrap=True,
        max_depth=23,
        max_features='sqrt',
        min_samples_leaf=8,
        min_samples_split=6,
        n_estimators=70
    )
    clf.fit(X_train, y_train)

    return clf, le, X_selected.columns.tolist(), (X_train, X_test, y_train, y_test)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50, is_sequence=False, patience=5, save_path=None):
    """
    Trains the model using the provided data loaders and optimizer. It also handles validation and early stopping.

    :param model: The model to be trained.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    :param criterion: Loss function to be used.
    :param optimizer: Optimizer for updating model parameters.
    :param device: Device to run the model on (CPU or GPU).
    :param num_epochs: Number of epochs to train the model.
    :param is_sequence: Boolean indicating if the model is for sequence data.
    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param save_path: Path to save the best model state.
    :return: The trained model and a tuple containing training and validation losses and accuracies.
    """
    model.to(device)

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []

    best_val_acc = 0.0
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)

            if is_sequence:
                loss = criterion(outputs.view(-1, outputs.shape[-1]), y_batch.view(-1))
                preds = outputs.argmax(dim=-1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.numel()
            else:
                loss = criterion(outputs, y_batch)
                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)

                if is_sequence:
                    loss = criterion(outputs.view(-1, outputs.shape[-1]), y_batch.view(-1))
                    preds = outputs.argmax(dim=-1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.numel()
                else:
                    loss = criterion(outputs, y_batch)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += y_batch.size(0)

                val_loss += loss.item()

        val_acc = correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1:2d}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        # Early Stopping Check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if save_path:
                torch.save(best_model_state, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    model.load_state_dict(best_model_state)
    return model, (train_losses, val_losses, train_accuracies, val_accuracies)