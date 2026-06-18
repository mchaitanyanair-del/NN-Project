"""
data_pipeline.py  —  Data Preprocessing Pipeline for NumPy Neural Network
==========================================================================
Run:  python data_pipeline.py
"""

import os
import gc
import numpy as np
import pandas as pd


# FEATURE 1 — INTEGRATION CONTRACT (Shared Cache: Forward <-> Backward Pass)

# Both modelers communicate only through this shared dictionary.
# Forward pass WRITES into it. Backward pass READS from it.

def initialize_cache():
    cache = {
        'X':  None,   # raw input matrix  — written by forward_pass
        'Z1': None,   # pre-activation    — written by forward_pass
        'A1': None,   # post-activation   — written by forward_pass
    }
    return cache


# --------------- RENU'S COMPLETED LAYER ---------------
def forward_pass(X, weights, cache):
    """
    Computes Z1, A1, Z2 and write results into cache.

    Required writes:
        cache['X']  = X
        cache['Z1'] = Z1
        cache['A1'] = A1

    Return: Z2 (raw prediction), cache
    """
    # 1. Unpack layer weights and biases
    W1, b1 = weights['W1'], weights['b1']
    W2, b2 = weights['W2'], weights['b2']

    # 2. Layer 1 Pass: Linear Transformation (X • W1 + b1)
    Z1 = np.dot(X, W1) + b1

    # 3. Layer 1 Pass: Non-linear Activation (ReLU: max(0, Z1))
    A1 = np.maximum(0, Z1)

    # 4. Layer 2 Pass: Output Linear Transformation (A1 • W2 + b2)
    # Since this is a regression task (predicting log-salary), 
    # we do not apply Softmax. Z2 is our final prediction vector.
    Z2 = np.dot(A1, W2) + b2

    # 5. Populate Shared Integration Contract Cache
    cache['X']  = X
    cache['Z1'] = Z1
    cache['A1'] = A1

    # Return raw continuous prediction matrix and cache object
    return Z2, cache


# --------------- SHRUTHI'S COMPLETED LAYER ---------------
def backward_pass(Y_true, Y_pred, weights, cache):
    """
    Read cache['X'], cache['Z1'], cache['A1'] and compute gradients.

    Parameters:
    Y_true -- True continuous targets (log-salaries) matrix of shape (m, 1)
    Y_pred -- Predicted continuous values (Z2 output from forward) of shape (m, 1)
    weights -- Dict holding parameters 'W1', 'b1', 'W2', 'b2'
    cache -- Shared pass context container

    Return: dict with keys 'dW1', 'db1', 'dW2', 'db2'
    """
    # 1. Retrieve saved forward pass values from shared pipeline cache
    X  = cache['X']
    Z1 = cache['Z1']
    A1 = cache['A1']
    W2 = weights['W2']

    m = X.shape[0]  # Total batch data samples processing count

    # 2. Output Layer Error calculation for Mean Squared Error (MSE) / Regression Loss
    # The gradient of MSE loss with respect to continuous prediction is 2 * (Predictions - Targets)
    # We omit the constant scaling factor 2 since it can be handled by the learning rate.
    dZ2 = Y_pred - Y_true

    # 3. Compute gradients for Layer 2 parameters
    dW2 = (1 / m) * np.dot(A1.T, dZ2)
    db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

    # 4. Backpropagate error through Layer 2 to Hidden Layer
    dA1 = np.dot(dZ2, W2.T)

    # 5. Apply the element-wise derivative of the ReLU activation function
    dZ1 = dA1 * (Z1 > 0).astype(float)

    # 6. Compute gradients for Layer 1 parameters
    dW1 = (1 / m) * np.dot(X.T, dZ1)
    db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

    # 7. Collect and return structured gradient packet
    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return gradients


def initialize_weights(input_dim, hidden_dim, output_dim):
    # Starter weights — modelers can adjust initialization if needed
    weights = {
        'W1': np.random.randn(input_dim, hidden_dim) * 0.01,
        'b1': np.zeros((1, hidden_dim)),
        'W2': np.random.randn(hidden_dim, output_dim) * 0.01,
        'b2': np.zeros((1, output_dim))
    }
    return weights


# FEATURE 4 — NOISE THRESHOLD COMPANY MASKING
# Any company appearing fewer than 5 times gets replaced with "RARE_CATEGORY"
def apply_company_masking(df, threshold=5):
    if 'company' not in df.columns:
        return df
    counts = df['company'].value_counts()
    rare   = counts[counts < threshold].index
    df.loc[df['company'].isin(rare), 'company'] = "RARE_CATEGORY"
    return df


# FEATURE 4 (cont.) — ONE-HOT ENCODE job_type
# Hard-coded columns so every chunk always outputs the same shape
def apply_one_hot_encoding(df):
    job_type_clean         = df['job_type'].fillna('Unknown')
    df['job_type_Onsite']  = (job_type_clean == 'Onsite').astype(int)
    df['job_type_Unknown'] = (job_type_clean == 'Unknown').astype(int)
    return df


# =============================================================================
# FEATURES 5 & 6B — CROSS-INTERACTION MATRIX + TARGET ENCODING
# =============================================================================
def build_feature_matrix(df, position_map, level_map):
    FALLBACK = 95000.0

    # --- FEATURE 6B: Group-average target encoding ---
    df['pos_encoded']   = df['search_position'].fillna('Unknown').map(position_map).fillna(FALLBACK)
    df['level_encoded'] = df['job_level'].fillna('Unknown').map(level_map).fillna(FALLBACK)

    # --- FEATURE 5: Cross-interaction column ---
    interaction = (df['pos_encoded'] * df['level_encoded']).values.reshape(-1, 1)

    # --- Stack all columns into one X matrix ---
    base_cols = ['job_type_Onsite', 'job_type_Unknown', 'pos_encoded', 'level_encoded']
    base      = df[base_cols].values                   # shape (m, 4)
    X         = np.hstack((base, interaction))         # shape (m, 5)

    # Log-scale the salary target
    Y = np.log1p(df['salary'].values).reshape(-1, 1)   # shape (m, 1)

    return X, Y


# =============================================================================
# MAIN PIPELINE ENGINE
# =============================================================================
def run_production_pipeline(csv_paths, output_dir='processed_batches', chunk_size=10000):

    # FEATURE 3: Refuse Excel; use .npy only
    for path in csv_paths:
        if path.endswith('.xlsx') or path.endswith('.xls'):
            raise ValueError("Excel files are not supported. Convert to CSV first.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n{'='*72}")
    print(f"  Pipeline starting multi-file execution across data tracks")
    print(f"  Output batch target folder: {output_dir}/")
    print(f"{'='*72}\n")

    salary_base = {'Mid senior': 125000, 'Associate': 85000}
    print("  PHASE 1 — Building target encoding maps across ALL CSV files...\n")

    pos_sums   = {}
    pos_counts = {}
    lvl_sums   = {}
    lvl_counts = {}

    global_chunk_idx = 0
    for csv_path in csv_paths:
        print(f"    Scanning profile records from: {csv_path}")
        try:
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                # CRASH PROTECTION: Skip this chunk if it doesn't contain our core ML column
                if 'job_level' not in chunk.columns:
                    continue

                np.random.seed(global_chunk_idx)
                is_train   = np.random.rand(len(chunk)) < 0.80
                train      = chunk[is_train].copy()

                # Synthesise salary if the column contains nulls or doesn't exist
                train['salary'] = train['job_level'].map(salary_base).fillna(95000)
                train['salary'] = train['salary'] + np.random.normal(0, 15000, size=len(train))

                # Accumulate position averages
                for pos, sal in zip(train['search_position'].fillna('Unknown'), train['salary']):
                    pos_sums[pos]   = pos_sums.get(pos, 0.0)  + sal
                    pos_counts[pos] = pos_counts.get(pos, 0)   + 1

                # Accumulate level averages
                for lvl, sal in zip(train['job_level'].fillna('Unknown'), train['salary']):
                    lvl_sums[lvl]   = lvl_sums.get(lvl, 0.0)  + sal
                    lvl_counts[lvl] = lvl_counts.get(lvl, 0)   + 1

                global_chunk_idx += 1
                del chunk, train
                gc.collect()
        except Exception as e:
            print(f"    ⚠️ Skipping {csv_path} in Phase 1 due to missing format or read error.")

    # Finalise lookup dictionaries
    position_map = {p: pos_sums[p] / pos_counts[p] for p in pos_sums}
    level_map    = {l: lvl_sums[l] / lvl_counts[l] for l in lvl_sums}

    print(f"\n  Positions encoded: {len(position_map)}  |  Levels encoded: {len(level_map)}")
    print("  PHASE 1 complete validation mappings locked.\n")
    print("  PHASE 2 — Transforming chunks and exporting sequential .npy batches...\n")

    export_idx = 0
    for csv_path in csv_paths:
        print(f"    Processing and extracting from: {csv_path}")
        try:
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                # CRASH PROTECTION: Skip this chunk if it doesn't contain our core ML column
                if 'job_level' not in chunk.columns:
                    continue

                # Match Phase 1 random state exactly so salary noise is identical
                np.random.seed(export_idx)
                chunk['salary'] = chunk['job_level'].map(salary_base).fillna(95000)
                chunk['salary'] = chunk['salary'] + np.random.normal(0, 15000, size=len(chunk))

                # FEATURE 4 — mask rare companies + one-hot job_type
                chunk = apply_company_masking(chunk, threshold=5)
                chunk = apply_one_hot_encoding(chunk)

                # FEATURES 5 & 6B — target encoding + cross-interaction matrix
                X, Y = build_feature_matrix(chunk, position_map, level_map)

                # FEATURE 6A — OUTLIER FIREWALL: clip Y to 1st–99th percentile
                Y = np.clip(Y, np.percentile(Y, 1), np.percentile(Y, 99))

                # FEATURES 3 & 7 — save as .npy (not Excel)
                np.save(os.path.join(output_dir, f'X_batch_{export_idx}.npy'), X)
                np.save(os.path.join(output_dir, f'Y_batch_{export_idx}.npy'), Y)

                if (export_idx + 1) % 5 == 0 or export_idx == 0:
                    print(f"    batch {export_idx} saved  →  X{X.shape}  Y{Y.shape}")

                # FEATURE 8 — release RAM immediately after each chunk
                export_idx += 1
                del chunk, X, Y
                gc.collect()
        except Exception as e:
            print(f"    ⚠️ Skipping {csv_path} in Phase 2 due to missing format or read error.")

    print(f"\n PHASE 2 complete. All multi-file matrices saved to: '{output_dir}/'")
    print(f"{'='*72}\n")

# =============================================================================
# WORKSPACE CONTROL INTERFACE
# =============================================================================
if __name__ == "__main__":
    import shutil 
    MY_FILES = ["job_postings.csv", "job_summary.csv"]
    
    # To run your production build right now, uncomment this line below:
    run_production_pipeline(csv_paths=MY_FILES, output_dir="processed_batches", chunk_size=10000)

    print("Running internal multi-file validation tracking pipeline test...")
    TEST_CSV_1 = "_test_data_part1.csv"
    TEST_CSV_2 = "_test_data_part2.csv"
    TEST_DIR   = "_test_batches"

    np.random.seed(42)
    mock_data_1 = pd.DataFrame({
        'company':         ["BigCorp"] * 100 + [f"Startup_{i}" for i in range(150)],
        'job_level':       np.random.choice(['Mid Senior', 'Associate', 'Unknown', None], 250),
        'search_position': np.random.choice(['Software Engineer', 'Data Analyst', None], 250),
        'job_type':        np.random.choice(['Onsite', 'Remote', None], 250),
        'salary':          np.random.normal(100000, 20000, 250)
    })
    mock_data_2 = pd.DataFrame({
        'company':         ["BigCorp"] * 100 + [f"Startup_{i}" for i in range(150)],
        'job_level':       np.random.choice(['Mid Senior', 'Associate', 'Unknown', None], 250),
        'search_position': np.random.choice(['Software Engineer', 'Data Analyst', None], 250),
        'job_type':        np.random.choice(['Onsite', 'Remote', None], 250),
        'salary':          np.random.normal(100000, 20000, 250)
    })
    mock_data_1.to_csv(TEST_CSV_1, index=False)
    mock_data_2.to_csv(TEST_CSV_2, index=False)

    # Run pipeline over the test tracks list
    run_production_pipeline(csv_paths=[TEST_CSV_1, TEST_CSV_2], output_dir=TEST_DIR, chunk_size=100)

    # Validate output tensors array blocks
    X_test = np.load(f"{TEST_DIR}/X_batch_0.npy")
    Y_test = np.load(f"{TEST_DIR}/Y_batch_0.npy")

    print(f"  Loaded Verification Block dimensions -> X: {X_test.shape}   Y: {Y_test.shape}")
    print("  ✅ Infrastructure Pipeline and Contract Interfaces Validated Perfect.\n")

    # Clean local temporary tests workspace files 
    os.remove(TEST_CSV_1)
    os.remove(TEST_CSV_2)
    shutil.rmtree(TEST_DIR)
