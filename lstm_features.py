"""
LSTM Time Series Feature Generator
Trains an LSTM to predict future price movements, then uses predictions as features for XGBoost
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class LSTMPredictor(nn.Module):
    """LSTM model for predicting future returns"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)  # Predict: 5-day return, 10-day return, direction probability
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


class LSTMFeatureGenerator:
    """Generate LSTM-based features for the main model"""

    def __init__(self, sequence_length=20, hidden_size=64, num_layers=2, device=None):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaler = StandardScaler()
        self.model = None
        self.feature_cols = None

        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"LSTM using device: {self.device}")

    def _prepare_sequences(self, df, feature_cols):
        """Convert dataframe to sequences for LSTM"""
        data = df[feature_cols].values

        sequences = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)

        return np.array(sequences)

    def _prepare_targets(self, df):
        """Prepare target variables (future returns)"""
        close = df['Close'].values

        # Calculate forward returns
        returns_5d = np.zeros(len(close))
        returns_10d = np.zeros(len(close))
        direction = np.zeros(len(close))

        for i in range(len(close) - 10):
            returns_5d[i] = (close[i + 5] - close[i]) / close[i]
            returns_10d[i] = (close[i + 10] - close[i]) / close[i]
            direction[i] = 1.0 if close[i + 5] > close[i] else 0.0

        # Align with sequences (offset by sequence_length)
        targets = np.column_stack([
            returns_5d[self.sequence_length:],
            returns_10d[self.sequence_length:],
            direction[self.sequence_length:]
        ])

        return targets

    def fit(self, df, feature_cols, epochs=50, batch_size=64, lr=0.001, verbose=True):
        """Train the LSTM model"""
        self.feature_cols = feature_cols

        if verbose:
            print(f"\nTraining LSTM Feature Generator...")
            print(f"  Sequence length: {self.sequence_length}")
            print(f"  Features: {len(feature_cols)}")
            print(f"  Samples: {len(df)}")

        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaled_data

        # Prepare sequences and targets
        X = self._prepare_sequences(df_scaled, feature_cols)
        y = self._prepare_targets(df)

        # Trim to match lengths
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]

        # Remove any NaN targets
        valid_mask = ~np.isnan(y).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        if verbose:
            print(f"  Training sequences: {len(X)}")

        # Initialize model first to test GPU compatibility
        self.model = LSTMPredictor(
            input_size=len(feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

        # Try to move model to GPU, fall back to CPU if it fails
        try:
            self.model = self.model.to(self.device)
            # Test with a small forward pass
            test_input = torch.zeros(1, self.sequence_length, len(feature_cols)).to(self.device)
            _ = self.model(test_input)
            if verbose:
                print(f"  Using device: {self.device}")
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'kernel' in str(e) or 'no kernel image' in str(e):
                print(f"  GPU not compatible (new GPU?), using CPU instead...")
                self.device = torch.device('cpu')
                self.model = self.model.to(self.device)
            else:
                raise e

        # Keep data on CPU, move batches to GPU during training (saves VRAM)
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)

        # Create data loader (data stays on CPU, batches moved to GPU in training loop)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                # Move batch to GPU only when needed (saves VRAM)
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

                # Free GPU memory aggressively
                del batch_X, batch_y, outputs, loss
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            avg_loss = total_loss / len(loader)
            scheduler.step(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

            # Early stopping
            if patience_counter >= 10:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

        if verbose:
            print(f"  Training complete. Best loss: {best_loss:.6f}")

        return self

    def generate_features(self, df, batch_size=64):
        """Generate LSTM predictions as features for the main model"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Scale features
        scaled_data = self.scaler.transform(df[self.feature_cols])
        df_scaled = df.copy()
        df_scaled[self.feature_cols] = scaled_data

        # Prepare sequences
        X = self._prepare_sequences(df_scaled, self.feature_cols)

        # Predict in batches to avoid OOM
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i + batch_size]).to(self.device)
                batch_pred = self.model(batch).cpu().numpy()
                all_predictions.append(batch_pred)
                del batch  # Free GPU memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        predictions = np.concatenate(all_predictions, axis=0)

        # Create feature dataframe
        # Pad the beginning with NaN (no predictions for first sequence_length rows)
        lstm_features = np.full((len(df), 3), np.nan)
        lstm_features[self.sequence_length:self.sequence_length + len(predictions)] = predictions

        feature_df = pd.DataFrame(
            lstm_features,
            index=df.index,
            columns=['LSTM_Return_5D', 'LSTM_Return_10D', 'LSTM_Direction_Prob']
        )

        return feature_df

    def save(self, path):
        """Save the model"""
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers
        }, path)
        print(f"LSTM model saved to {path}")

    def load(self, path):
        """Load a saved model"""
        checkpoint = torch.load(path, map_location=self.device)

        self.scaler = checkpoint['scaler']
        self.feature_cols = checkpoint['feature_cols']
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']

        self.model = LSTMPredictor(
            input_size=len(self.feature_cols),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        print(f"LSTM model loaded from {path}")

        return self


if __name__ == "__main__":
    # Test the LSTM feature generator
    from data_collector import CSPDataCollector

    print("Testing LSTM Feature Generator...")
    print("=" * 60)

    # Collect sample data
    collector = CSPDataCollector('NVDA', period='5y')
    collector.fetch_data()
    collector.calculate_technical_indicators()

    df = collector.data.dropna()

    # Features to use for LSTM
    lstm_input_features = [
        'Close', 'Volume', 'RSI', 'MACD', 'BB_Position',
        'ATR_Pct', 'Return_1D', 'Return_5D', 'Volatility_20D'
    ]

    # Train LSTM
    lstm_gen = LSTMFeatureGenerator(sequence_length=20)
    lstm_gen.fit(df, lstm_input_features, epochs=30, verbose=True)

    # Generate features
    lstm_features = lstm_gen.generate_features(df)
    print("\nGenerated LSTM features:")
    print(lstm_features.dropna().head(10))

    print("\nFeature statistics:")
    print(lstm_features.describe())
