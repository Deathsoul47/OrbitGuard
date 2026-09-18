import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Dict

# Configure module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)


class OrbitalDataset(Dataset):
    """
    PyTorch Dataset for orbital telemetry data.
    Converts 3D numpy arrays of sequences and 2D arrays of targets into PyTorch tensors.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def train_orbital_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 10,
    checkpoint_dir: str = 'checkpoints'
) -> Dict[str, list]:
    """
    Executes the training loop for the OrbitalLSTM.
    Includes validation, learning rate scheduling, gradient clipping, and checkpoint saving.

    Args:
        model (nn.Module): The PyTorch model to train.
        X_train (np.ndarray): Training sequences (batch, seq_len, features).
        y_train (np.ndarray): Training targets (batch, targets).
        X_val (np.ndarray): Validation sequences.
        y_val (np.ndarray): Validation targets.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Mini-batch size.
        learning_rate (float): Initial learning rate for Adam optimizer.
        patience (int): Number of epochs with no validation improvement before early stopping.
        checkpoint_dir (str): Directory path to save best model weights.

    Returns:
        Dict[str, list]: History dictionary containing 'train_loss' and 'val_loss' across epochs.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Initiating training on device: {device}")
    model = model.to(device)
    
    # Setup DataLoaders
    train_dataset = OrbitalDataset(X_train, y_train)
    val_dataset = OrbitalDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup Loss, Optimizer, and LR Scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        running_train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in LSTMs
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        
        # --- VALIDATION PHASE ---
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        
        logger.info(f"Epoch [{epoch}/{epochs}] - Train MSE: {epoch_train_loss:.6f} | Val MSE: {epoch_val_loss:.6f}")
        
        # Step the learning rate scheduler based on validation loss
        scheduler.step(epoch_val_loss)
        
        # --- MODEL CHECKPOINTING & EARLY STOPPING ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(checkpoint_dir, 'best_orbit_lstm.pth')
            torch.save(model.state_dict(), checkpoint_path)
            logger.debug(f"Validation loss decreased. Checkpoint saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs. No improvement for {patience} epochs.")
                break

    # Load best weights before returning
    best_checkpoint = os.path.join(checkpoint_dir, 'best_orbit_lstm.pth')
    if os.path.exists(best_checkpoint):
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        logger.info(f"Loaded best model weights with Validation MSE: {best_val_loss:.6f}")
        
    return history
