"""
GPU-Accelerated Deep Learning Models for CSP Timing
Uses LSTM, Transformer, and TabNet with PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_collector import CSPDataCollector


class SequenceDataset(Dataset):
    """Dataset that creates sequences for LSTM"""
    def __init__(self, features, labels, seq_length=20):
        self.features = features
        self.labels = labels
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        # Get sequence of features
        X = self.features[idx:idx + self.seq_length]
        # Get current label
        y = self.labels[idx + self.seq_length - 1]
        return torch.FloatTensor(X), torch.FloatTensor([y])


class LSTMModel(nn.Module):
    """LSTM-based model for temporal pattern recognition"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Apply attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # Final prediction
        output = self.fc(context)
        return output


class TransformerModel(nn.Module):
    """Transformer-based model with temporal encoding"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.2):
        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 100, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)

        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer forward pass
        x = self.transformer(x)

        # Use the last timestep for classification
        x = x[:, -1, :]

        # Final prediction
        output = self.fc(x)
        return output


class HybridModel(nn.Module):
    """Hybrid model combining LSTM and Transformer"""
    def __init__(self, input_size, hidden_size=128, d_model=128, dropout=0.3):
        super(HybridModel, self).__init__()

        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        # Transformer branch
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size + d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # LSTM branch
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_features = h_n[-1]

        # Transformer branch
        trans_x = self.input_projection(x)
        trans_out = self.transformer(trans_x)
        trans_features = trans_out[:, -1, :]

        # Combine features
        combined = torch.cat([lstm_features, trans_features], dim=1)

        # Final prediction
        output = self.fusion(combined)
        return output


class DeepLearningTrainer:
    """Trainer for deep learning models with GPU acceleration"""
    def __init__(self, model_type='hybrid', seq_length=20, force_cpu=False):
        self.model_type = model_type
        self.seq_length = seq_length
        self.force_cpu = force_cpu
        # Check CUDA availability
        if self.force_cpu:
            self.device = torch.device('cpu')
        else:
            cuda_available = torch.cuda.is_available()

            # For RTX 5070 Ti (sm_120), disable cuDNN for RNNs
            # Use PyTorch's native CUDA implementation instead
            if cuda_available:
                try:
                    # Test if CUDA works
                    torch.zeros(1).cuda()
                    self.device = torch.device('cuda')

                    # Disable cuDNN for RNNs (workaround for sm_120)
                    torch.backends.cudnn.enabled = False
                    print("Note: cuDNN disabled for RTX 5070 Ti compatibility")
                    print("      Using PyTorch native CUDA kernels (slightly slower but works)")
                except Exception as e:
                    print(f"Warning: CUDA device found but not compatible. Using CPU. ({e})")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')

        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("Note: Training on CPU. For RTX 5070 Ti, install PyTorch 2.6+ for GPU support.")
            print("      Install with: pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124")

    def load_data(self, ticker='NVDA', period='10y'):
        """Load and prepare data"""
        print(f"\nLoading data for {ticker}...")
        collector = CSPDataCollector(ticker, period=period)
        df, self.feature_cols = collector.get_training_data()

        X = df[self.feature_cols].values
        y = df['Good_CSP_Time'].values

        print(f"Total samples: {len(df)}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Positive class: {y.mean():.2%}")

        return X, y, df

    def prepare_data(self, X, y, test_size=0.2, val_size=0.1):
        """Prepare train/val/test splits"""
        # Chronological split
        n = len(X)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nTrain: {len(X_train)} samples")
        print(f"Val: {len(X_val)} samples")
        print(f"Test: {len(X_test)} samples")

        return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test)

    def create_model(self, input_size):
        """Create the specified model"""
        if self.model_type == 'lstm':
            model = LSTMModel(input_size, hidden_size=128, num_layers=2, dropout=0.3)
        elif self.model_type == 'transformer':
            model = TransformerModel(input_size, d_model=128, nhead=8, num_layers=3)
        elif self.model_type == 'hybrid':
            model = HybridModel(input_size, hidden_size=128, d_model=128, dropout=0.3)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model.to(self.device)

    def train_epoch(self, model, dataloader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        total_loss = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, model, dataloader, criterion):
        """Validate the model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, lr=0.001):
        """Train the model"""
        # Create datasets
        train_dataset = SequenceDataset(X_train, y_train, self.seq_length)
        val_dataset = SequenceDataset(X_val, y_val, self.seq_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        input_size = X_train.shape[1]
        self.model = self.create_model(input_size)

        # Loss and optimizer
        # Use weighted loss for imbalanced data
        pos_weight = torch.tensor([y_train.mean() / (1 - y_train.mean())]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # For models with sigmoid already, use BCELoss
        criterion = nn.BCELoss()

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        print(f"\nTraining {self.model_type} model...")
        print("="*70)

        for epoch in range(epochs):
            train_loss = self.train_epoch(self.model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.validate(self.model, val_loader, criterion)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'best_{self.model_type}_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Load best model
        self.model.load_state_dict(torch.load(f'best_{self.model_type}_model.pth'))
        print(f"\nBest validation loss: {best_val_loss:.4f}")

        return self.model

    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        test_dataset = SequenceDataset(X_test, y_test, self.seq_length)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        self.model.eval()
        predictions = []
        probabilities = []
        actuals = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)

                predicted = (outputs > 0.5).float()
                predictions.extend(predicted.cpu().numpy())
                probabilities.extend(outputs.cpu().numpy())
                actuals.extend(y_batch.numpy())

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        actuals = np.array(actuals)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        accuracy = accuracy_score(actuals, predictions)
        precision = precision_score(actuals, predictions, zero_division=0)
        recall = recall_score(actuals, predictions)
        f1 = f1_score(actuals, predictions)
        roc_auc = roc_auc_score(actuals, probabilities)

        print("\n" + "="*70)
        print(f"{self.model_type.upper()} MODEL - TEST SET RESULTS")
        print("="*70)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy
        axes[1].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.model_type}_training_history.png', dpi=300, bbox_inches='tight')
        print(f"\nTraining history saved to {self.model_type}_training_history.png")
        plt.close()

    def save_model(self, filename=None):
        """Save model and scaler"""
        if filename is None:
            filename = f'csp_{self.model_type}_model.pkl'

        model_package = {
            'model_state': self.model.state_dict(),
            'model_type': self.model_type,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'seq_length': self.seq_length,
            'input_size': len(self.feature_cols)
        }

        joblib.dump(model_package, filename)
        print(f"\nModel saved to {filename}")


if __name__ == "__main__":
    # Train LSTM model
    print("\n" + "="*70)
    print("TRAINING LSTM MODEL")
    print("="*70)
    lstm_trainer = DeepLearningTrainer(model_type='lstm', seq_length=20)
    X, y, df = lstm_trainer.load_data('NVDA', period='10y')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = lstm_trainer.prepare_data(X, y)
    lstm_trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.001)
    lstm_results = lstm_trainer.evaluate(X_test, y_test)
    lstm_trainer.plot_training_history()
    lstm_trainer.save_model()

    # Train Transformer model
    print("\n" + "="*70)
    print("TRAINING TRANSFORMER MODEL")
    print("="*70)
    trans_trainer = DeepLearningTrainer(model_type='transformer', seq_length=20)
    X, y, df = trans_trainer.load_data('NVDA', period='10y')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = trans_trainer.prepare_data(X, y)
    trans_trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.001)
    trans_results = trans_trainer.evaluate(X_test, y_test)
    trans_trainer.plot_training_history()
    trans_trainer.save_model()

    # Train Hybrid model
    print("\n" + "="*70)
    print("TRAINING HYBRID MODEL")
    print("="*70)
    hybrid_trainer = DeepLearningTrainer(model_type='hybrid', seq_length=20)
    X, y, df = hybrid_trainer.load_data('NVDA', period='10y')
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = hybrid_trainer.prepare_data(X, y)
    hybrid_trainer.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.001)
    hybrid_results = hybrid_trainer.evaluate(X_test, y_test)
    hybrid_trainer.plot_training_history()
    hybrid_trainer.save_model()

    # Compare all models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    results_df = pd.DataFrame([
        {'Model': 'LSTM', **lstm_results},
        {'Model': 'Transformer', **trans_results},
        {'Model': 'Hybrid', **hybrid_results}
    ])
    print(results_df.to_string(index=False))
