#!/usr/bin/env python3
"""
Main runner script for CSP timing model
"""

import sys
import os
from pathlib import Path


def main():
    print("="*70)
    print("CSP TIMING MODEL - NVDA Cash Secured Put Optimizer")
    print("="*70)

    # Check if model exists
    model_exists = Path('csp_model.pkl').exists()

    if not model_exists:
        print("\nNo trained model found. Starting training process...")
        print("\nThis will:")
        print("  1. Download 5 years of NVDA data")
        print("  2. Calculate 24+ technical indicators")
        print("  3. Train and compare 3 different models")
        print("  4. Select the best performer")
        print("\nThis may take 5-10 minutes...\n")

        response = input("Proceed with training? (y/n): ").lower()
        if response != 'y':
            print("Training cancelled.")
            return

        # Train the model
        from model_trainer import CSPModelTrainer
        trainer = CSPModelTrainer()
        trainer.full_training_pipeline()

        print("\n" + "="*70)
        print("MODEL TRAINING COMPLETE!")
        print("="*70)

    # Make predictions
    print("\n" + "="*70)
    print("MAKING PREDICTIONS")
    print("="*70)

    from predictor import CSPPredictor
    predictor = CSPPredictor('csp_model.pkl')

    if len(sys.argv) > 1:
        ticker_input = sys.argv[1].upper()
        if ',' in ticker_input:
            tickers = [t.strip() for t in ticker_input.split(',')]
            predictor.scan_multiple_tickers(tickers)
        else:
            predictor.predict(ticker_input, show_details=True)
    else:
        # Default to NVDA
        predictor.predict('NVDA', show_details=True)

    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("  Single ticker:   python run_csp_model.py NVDA")
    print("  Multiple tickers: python run_csp_model.py NVDA,TSLA,AMD")
    print("  Retrain model:   python model_trainer.py")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
