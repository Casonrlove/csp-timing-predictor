"""
TabNet training - Attention-based model specifically designed for tabular data
Excellent for financial features with built-in feature importance
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import joblib
from data_collector import CSPDataCollector


class TabNetTrainer:
    """TabNet model trainer with GPU acceleration"""
    def __init__(self, force_cpu=False):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None

        # Try to use GPU with cuDNN disabled for RTX 5070 Ti
        if force_cpu:
            self.device = 'cpu'
        else:
            try:
                if torch.cuda.is_available():
                    # Test actual computation
                    torch.randn(10, 10).cuda()
                    self.device = 'cuda'
                    # Disable cuDNN for compatibility
                    torch.backends.cudnn.enabled = False
                    print("Note: cuDNN disabled for RTX 5070 Ti compatibility")
                else:
                    self.device = 'cpu'
            except Exception as e:
                print(f"Warning: CUDA not fully compatible. Using CPU. ({e})")
                self.device = 'cpu'

        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    def load_multi_ticker_data(self, tickers=['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT'], period='10y'):
        """Load data from multiple tickers for better generalization"""
        print(f"\nLoading data from multiple tickers: {tickers}")

        all_X = []
        all_y = []

        for ticker in tickers:
            try:
                print(f"  Loading {ticker}...")
                collector = CSPDataCollector(ticker, period=period)
                df, feature_cols = collector.get_training_data()

                if self.feature_cols is None:
                    self.feature_cols = feature_cols

                X = df[self.feature_cols].values
                y = df['Good_CSP_Time'].values

                all_X.append(X)
                all_y.append(y)
                print(f"    {ticker}: {len(df)} samples, {y.mean():.2%} positive")

            except Exception as e:
                print(f"    ERROR loading {ticker}: {str(e)}")
                continue

        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)

        print(f"\nTotal combined samples: {len(X_combined)}")
        print(f"Overall positive class: {y_combined.mean():.2%}")

        return X_combined, y_combined

    def prepare_data(self, X, y, test_size=0.2):
        """Prepare train/test splits"""
        # Chronological split
        split_idx = int(len(X) * (1 - test_size))

        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        # Don't scale for TabNet (it handles this internally)
        # But we'll still save scaler for compatibility
        self.scaler.fit(X_train)

        print(f"\nTrain: {len(X_train)} samples ({y_train.mean():.2%} positive)")
        print(f"Test: {len(X_test)} samples ({y_test.mean():.2%} positive)")

        return X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, X_val=None, y_val=None,
              max_epochs=200, patience=50, batch_size=1024):
        """Train TabNet model"""

        # Split validation set if not provided
        if X_val is None:
            val_size = int(len(X_train) * 0.15)
            X_val = X_train[-val_size:]
            y_val = y_train[-val_size:]
            X_train = X_train[:-val_size]
            y_train = y_train[:-val_size]

        # Calculate class weights for imbalanced data
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        weights = {0: 1.0, 1: n_neg / n_pos}

        print(f"\nClass weights: {weights}")
        print(f"Training with {len(X_train)} samples")

        # Create TabNet model
        self.model = TabNetClassifier(
            n_d=64,                    # Width of decision prediction layer
            n_a=64,                    # Width of attention embedding
            n_steps=5,                 # Number of steps in the architecture
            gamma=1.5,                 # Coefficient for feature reusage
            n_independent=2,           # Number of independent GLU layers
            n_shared=2,                # Number of shared GLU layers
            lambda_sparse=1e-4,        # Sparsity regularization
            momentum=0.3,              # Momentum for batch normalization
            clip_value=2.0,            # Gradient clipping
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"mode": "min", "patience": 10, "factor": 0.5},
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            mask_type='entmax',        # Mask type (sparsemax or entmax)
            device_name=self.device,
            verbose=10,
            seed=42
        )

        print("\nTraining TabNet model...")
        print("="*70)

        # Train
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['validation'],
            eval_metric=['accuracy', 'auc'],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            weights=1,  # Can pass class weights here if needed
        )

        print("\nTraining complete!")

    def evaluate(self, X_test, y_test):
        """Evaluate on test set"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print("\n" + "="*70)
        print("TABNET MODEL - TEST SET RESULTS")
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

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance from TabNet"""
        if self.model is None:
            print("No model trained yet")
            return

        # Get feature importances
        feature_importances = self.model.feature_importances_

        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)

        # Plot top N features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'TabNet - Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('tabnet_feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved to tabnet_feature_importance.png")
        plt.close()

        # Print top features
        print("\nTop 10 Most Important Features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df

    def plot_training_history(self):
        """Plot training history"""
        if not hasattr(self.model, 'history'):
            print("No training history available")
            return

        history = self.model.history

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        if 'loss' in history:
            axes[0].plot(history['loss'], label='Train Loss')
            if 'val_0_loss' in history:
                axes[0].plot(history['val_0_loss'], label='Val Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True)

        # Metrics
        if 'val_0_auc' in history:
            axes[1].plot(history['val_0_auc'], label='Val AUC')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('AUC')
            axes[1].set_title('Validation AUC')
            axes[1].legend()
            axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('tabnet_training_history.png', dpi=300, bbox_inches='tight')
        print("Training history saved to tabnet_training_history.png")
        plt.close()

    def save_model(self, filename='csp_tabnet_model.pkl'):
        """Save model"""
        if self.model is None:
            print("No model to save")
            return

        # Save TabNet model
        self.model.save_model('tabnet_checkpoint')

        # Save metadata
        model_package = {
            'model_path': 'tabnet_checkpoint',
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': 'TabNet'
        }

        joblib.dump(model_package, filename)
        print(f"\nModel saved to {filename}")

    def load_model(self, filename='csp_tabnet_model.pkl'):
        """Load saved model"""
        model_package = joblib.load(filename)

        self.scaler = model_package['scaler']
        self.feature_cols = model_package['feature_cols']

        # Load TabNet model
        self.model = TabNetClassifier()
        self.model.load_model(model_package['model_path'] + '.zip')

        print(f"Model loaded from {filename}")


if __name__ == "__main__":
    # Try GPU with cuDNN disabled for RTX 5070 Ti compatibility
    trainer = TabNetTrainer(force_cpu=False)

    # Option 1: Train on single ticker
    print("\n" + "="*70)
    print("OPTION 1: SINGLE TICKER (NVDA)")
    print("="*70)
    X, y, df = trainer.load_data('NVDA', period='10y')
    X_train, y_train, X_test, y_test = trainer.prepare_data(X, y)
    trainer.train(X_train, y_train)
    results = trainer.evaluate(X_test, y_test)
    trainer.plot_feature_importance()
    trainer.plot_training_history()
    trainer.save_model('csp_tabnet_nvda.pkl')

    # Option 2: Train on multiple tickers (better generalization)
    print("\n" + "="*70)
    print("OPTION 2: MULTI-TICKER TRAINING")
    print("="*70)
    trainer2 = TabNetTrainer(force_cpu=False)
    X, y = trainer2.load_multi_ticker_data(['NVDA', 'AMD', 'TSLA', 'AAPL', 'MSFT'], period='10y')
    X_train, y_train, X_test, y_test = trainer2.prepare_data(X, y)
    trainer2.train(X_train, y_train, max_epochs=200, batch_size=2048)
    results2 = trainer2.evaluate(X_test, y_test)
    trainer2.plot_feature_importance()
    trainer2.plot_training_history()
    trainer2.save_model('csp_tabnet_multi.pkl')

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Single ticker ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Multi-ticker ROC-AUC:  {results2['roc_auc']:.4f}")
