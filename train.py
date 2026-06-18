"""
train.py  —  Training Engine and Accuracy Evaluator
==========================================================================
Run:  python train.py
"""

import os
import numpy as np
import data_pipeline as dp  

def calculate_mse_loss(Y_true, Y_pred):
    """Computes Mean Squared Error."""
    return np.mean((Y_pred - Y_true) ** 2)

def calculate_r2_score(Y_true, Y_pred):
    """
    Computes R² Score (Coefficient of Determination).
    1.0 is a perfect score. 0.0 means the model is just guessing averages.
    """
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    ss_tot = np.sum((Y_true - np.mean(Y_true)) ** 2)
    return 1.0 - (ss_res / (ss_tot + 1e-8))

def train_model(batch_dir="processed_batches", epochs=15, lr=0.01, hidden_dim=64):
    if not os.path.exists(batch_dir) or len(os.listdir(batch_dir)) == 0:
        raise FileNotFoundError(f"No batches found in '{batch_dir}/'. Run data_pipeline.py first!")

    x_files = sorted([f for f in os.listdir(batch_dir) if f.startswith('X_batch_')])
    num_batches = len(x_files)
    num_train_batches = int(num_batches * 0.8)
    
    print(f"Found {num_batches} data blocks.")
    print(f"  -> {num_train_batches} Batches allocated for Training")
    print(f"  -> {num_batches - num_train_batches} Batches allocated for Final Accuracy Check\n")

    weights = dp.initialize_weights(input_dim=5, hidden_dim=hidden_dim, output_dim=1)
    weights['W1'] = np.random.randn(5, hidden_dim) * 0.1
    weights['W2'] = np.random.randn(hidden_dim, 1) * 0.1
    weights['b2'] = np.full((1, 1), 11.5)
    
    cache = dp.initialize_cache()

    # Establish global scaling bounds
    sample_X = np.load(os.path.join(batch_dir, x_files[0]))
    x_max = np.max(sample_X, axis=0) + 1e-8
    x_min = np.min(sample_X, axis=0)

    print("========================================================================")
    print("    STARTING NEURAL NETWORK TRAINING LOOPS")
    print("========================================================================")

    for epoch in range(epochs):
        train_losses = []
        for b in range(num_train_batches):
            X_batch = np.load(os.path.join(batch_dir, f"X_batch_{b}.npy"))
            Y_batch = np.load(os.path.join(batch_dir, f"Y_batch_{b}.npy"))

            X_batch_scaled = (X_batch - x_min) / (x_max - x_min)

            Y_pred, cache = dp.forward_pass(X_batch_scaled, weights, cache)
            grads = dp.backward_pass(Y_batch, Y_pred, weights, cache)

            for key in weights:
                weights[key] -= lr * grads['d' + key]

            loss = calculate_mse_loss(Y_batch, Y_pred)
            train_losses.append(loss)

        avg_train_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Training MSE Loss: {avg_train_loss:.5f}")

    # --- EXPORT MODEL ARTIFACTS AND SCALING FIELDS ---
    print("\n Saving optimized neural network parameters and scaling fields to drive...")
    np.save("W1_optimized.npy", weights['W1'])
    np.save("b1_optimized.npy", weights['b1'])
    np.save("W2_optimized.npy", weights['W2'])
    np.save("b2_optimized.npy", weights['b2'])
    np.save("x_min.npy", x_min)
    np.save("x_max.npy", x_max)
    print(" All model parameters safely exported.")

    print("\n========================================================================")
    print("    RUNNING ACCURACY EVALUATION (On Unseen Data Splits)")
    print("========================================================================")
    
    val_losses = []
    all_y_true = []
    all_y_pred = []

    for b in range(num_train_batches, num_batches):
        X_val = np.load(os.path.join(batch_dir, f"X_batch_{b}.npy"))
        Y_val = np.load(os.path.join(batch_dir, f"Y_batch_{b}.npy"))

        X_val_scaled = (X_val - x_min) / (x_max - x_min)
        Y_val_pred, _ = dp.forward_pass(X_val_scaled, weights, cache)
        
        val_losses.append(calculate_mse_loss(Y_val, Y_val_pred))
        all_y_true.append(Y_val)
        all_y_pred.append(Y_val_pred)

    global_true = np.vstack(all_y_true)
    global_pred = np.vstack(all_y_pred)

    final_val_mse = np.mean(val_losses)
    r2_accuracy = calculate_r2_score(global_true, global_pred)

    print(f"  Validation Mean Squared Error (MSE):  {final_val_mse:.5f}")
    print(f"  Validation Model R² Accuracy Score:   {r2_accuracy:.2%}")
    print("========================================================================\n")

if __name__ == "__main__":
    train_model(batch_dir="processed_batches", epochs=200, lr=0.002, hidden_dim=64)
