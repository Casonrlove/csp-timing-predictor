"""
Quick training script - trains only the Hybrid model
Use this for faster initial testing
"""

import torch
from deep_learning_model import DeepLearningTrainer

print("="*70)
print("QUICK TRAINING - HYBRID MODEL ONLY")
print("="*70)

# Check device
if torch.cuda.is_available():
    try:
        torch.zeros(1).cuda()
        print("GPU will be used for training (RTX 5070 Ti)")
        print("Note: cuDNN disabled, using native PyTorch CUDA kernels")
    except:
        print("GPU detected but not compatible. Using CPU.")
else:
    print("Training on CPU (slower but works)")

print("\nTraining Hybrid model (LSTM + Transformer)...")
print("With GPU: 3-5 minutes | With CPU: 15-30 minutes")
print("="*70)

# Try to use GPU (cuDNN disabled for RTX 5070 Ti compatibility)
trainer = DeepLearningTrainer(model_type='hybrid', seq_length=20, force_cpu=False)
X, y, df = trainer.load_data('NVDA', period='10y')
(X_train, y_train), (X_val, y_val), (X_test, y_test) = trainer.prepare_data(X, y)

# Train with reduced epochs for faster testing
trainer.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=64, lr=0.001)

# Evaluate
results = trainer.evaluate(X_test, y_test)

# Save
trainer.plot_training_history()
trainer.save_model('csp_hybrid_model.pkl')

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Test ROC-AUC: {results['roc_auc']:.4f}")
print(f"Test Accuracy: {results['accuracy']:.4f}")
print("\nModel saved as: csp_hybrid_model.pkl")
print("\nTo make predictions:")
print("  python deep_predictor.py csp_hybrid_model.pkl NVDA")
print("="*70)
