"""
Model training for CSP timing prediction
Trains and evaluates multiple models to find best parameters
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from data_collector import CSPDataCollector


class CSPModelTrainer:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_cols = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """Load prepared data"""
        if self.data_path:
            df = pd.read_csv(self.data_path, index_col=0)
        else:
            # Generate fresh data
            print("Generating fresh data...")
            collector = CSPDataCollector('NVDA', period='5y')
            df, self.feature_cols = collector.get_training_data()

        if self.feature_cols is None:
            # Infer feature columns
            self.feature_cols = [col for col in df.columns
                                if col not in ['Good_CSP_Time', 'Max_Drawdown_35D']]

        X = df[self.feature_cols]
        y = df['Good_CSP_Time']

        print(f"\nDataset: {len(df)} samples")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Target distribution:\n{y.value_counts()}")
        print(f"Positive class rate: {y.mean():.2%}")

        return X, y, df

    def split_data(self, X, y, test_size=0.2):
        """Split data into train and test sets (chronological split for time series)"""
        # Use chronological split since we're dealing with time series
        split_idx = int(len(X) * (1 - test_size))

        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]

        print(f"\nTrain set: {len(self.X_train)} samples ({self.y_train.mean():.2%} positive)")
        print(f"Test set: {len(self.X_test)} samples ({self.y_test.mean():.2%} positive)")

        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test

    def train_logistic_regression(self):
        """Train Logistic Regression with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Logistic Regression...")
        print("="*50)

        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'class_weight': ['balanced', None],
            'max_iter': [1000]
        }

        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train_scaled, self.y_train)

        self.models['logistic_regression'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def train_random_forest(self):
        """Train Random Forest with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training Random Forest...")
        print("="*50)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4],
            'class_weight': ['balanced', None]
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.models['random_forest'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def train_xgboost(self):
        """Train XGBoost with hyperparameter tuning"""
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'scale_pos_weight': [1, scale_pos_weight]
        }

        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.models['xgboost'] = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def evaluate_model(self, model, model_name, use_scaled=False):
        """Evaluate a trained model"""
        X_test = self.X_test_scaled if use_scaled else self.X_test

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print(f"\n{'='*50}")
        print(f"{model_name} - Test Set Evaluation")
        print(f"{'='*50}")

        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)

        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)

        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        print(f"Average Precision Score: {avg_precision:.4f}")

        return {
            'model_name': model_name,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'y_pred_proba': y_pred_proba
        }

    def compare_models(self):
        """Compare all trained models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        results = []

        # Evaluate each model
        if 'logistic_regression' in self.models:
            result = self.evaluate_model(
                self.models['logistic_regression'],
                'Logistic Regression',
                use_scaled=True
            )
            results.append(result)

        if 'random_forest' in self.models:
            result = self.evaluate_model(
                self.models['random_forest'],
                'Random Forest',
                use_scaled=False
            )
            results.append(result)

        if 'xgboost' in self.models:
            result = self.evaluate_model(
                self.models['xgboost'],
                'XGBoost',
                use_scaled=False
            )
            results.append(result)

        # Summary comparison
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        results_df = pd.DataFrame([{
            'Model': r['model_name'],
            'ROC-AUC': r['roc_auc'],
            'Avg Precision': r['avg_precision']
        } for r in results])
        print(results_df.to_string(index=False))

        # Select best model based on ROC-AUC
        best_result = max(results, key=lambda x: x['roc_auc'])
        self.best_model = self.models[best_result['model_name'].lower().replace(' ', '_')]
        print(f"\nBest Model: {best_result['model_name']} (ROC-AUC: {best_result['roc_auc']:.4f})")

        return results

    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        if self.best_model is None:
            print("No best model selected yet")
            return

        model_type = type(self.best_model).__name__

        plt.figure(figsize=(12, 8))

        if hasattr(self.best_model, 'feature_importances_'):
            # Tree-based models
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features

            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [self.feature_cols[i] for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top 20 Feature Importances - {model_type}')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance plot saved to feature_importance.png")

        elif hasattr(self.best_model, 'coef_'):
            # Linear models
            coefs = np.abs(self.best_model.coef_[0])
            indices = np.argsort(coefs)[::-1][:20]

            plt.barh(range(len(indices)), coefs[indices])
            plt.yticks(range(len(indices)), [self.feature_cols[i] for i in indices])
            plt.xlabel('Absolute Coefficient Value')
            plt.title(f'Top 20 Feature Coefficients - {model_type}')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("\nFeature importance plot saved to feature_importance.png")

        plt.close()

        # Print top features
        print("\nTop 10 Most Important Features:")
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]
            for i, idx in enumerate(indices, 1):
                print(f"{i}. {self.feature_cols[idx]}: {importances[idx]:.4f}")

    def save_model(self, filename='csp_model.pkl'):
        """Save the best model and scaler"""
        if self.best_model is None:
            print("No model to save")
            return

        model_package = {
            'model': self.best_model,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'model_type': type(self.best_model).__name__
        }

        joblib.dump(model_package, filename)
        print(f"\nModel saved to {filename}")

    def full_training_pipeline(self):
        """Run complete training pipeline"""
        # Load and prepare data
        X, y, df = self.load_data()
        self.split_data(X, y)

        # Train all models
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()

        # Compare and select best
        results = self.compare_models()

        # Analyze feature importance
        self.plot_feature_importance()

        # Save best model
        self.save_model()

        return self.best_model, results


if __name__ == "__main__":
    trainer = CSPModelTrainer()
    best_model, results = trainer.full_training_pipeline()
