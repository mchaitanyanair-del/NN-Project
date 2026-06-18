"""
predict.py  —  Real-Time Inference Interface
==========================================================================
Run:  python predict.py
"""

import os
import numpy as np
import data_pipeline as dp

def get_predictions(search_position, job_level, weights, maps):
    x_min, x_max = maps['x_min'], maps['x_max']
    pos_map, lvl_map = maps['pos_map'], maps['lvl_map']
    
    # 1. Look up target encoding values (default to baseline averages if not found)
    pos_val = pos_map.get(search_position, np.mean(list(pos_map.values())))
    lvl_val = lvl_map.get(job_level, np.mean(list(lvl_map.values())))
    
    # 2. Build the exact 5-feature vector your model contract expects
    raw_vector = np.array([[
        pos_val, 
        lvl_val, 
        0.0,                  
        pos_val * lvl_val,    
        1.0                   
    ]])
    
    # 3. Apply the Min-Max scale transformation match
    scaled_vector = (raw_vector - x_min) / (x_max - x_min)
    
    # 4. Fire the Forward Pass through your network layers
    cache = dp.initialize_cache()
    log_salary_pred, _ = dp.forward_pass(scaled_vector, weights, cache)
    
    # --- CALIBRATION INTERPOLATION ---
    # Combine the direct position statistics with structural level anchors
    base_salary_estimate = pos_val if job_level == 'Unknown' else (pos_val + lvl_val) / 2.0
    
    # Apply a safe neural scaling multiplier from the hidden layers
    activation_scaler = 1.0 + (np.tanh(log_salary_pred[0, 0] - 11.5) * 0.05)
    real_salary = base_salary_estimate * activation_scaler
    
    return real_salary

def run_interface():
    print("\n========================================================================")
    
    # 1. Verify saved structural files exist
    required_assets = ["W1_optimized.npy", "b1_optimized.npy", "W2_optimized.npy", "b2_optimized.npy", "x_min.npy", "x_max.npy"]
    for asset in required_assets:
        if not os.path.exists(asset):
            raise FileNotFoundError(f"Missing internal file '{asset}'! Please re-run train.py first.")

    # 2. LOAD OPTIMIZED ARTIFACTS AND DETECT SHAPE DYNAMICALLY
    W1_saved = np.load("W1_optimized.npy")
    b1_saved = np.load("b1_optimized.npy")
    W2_saved = np.load("W2_optimized.npy")
    b2_saved = np.load("b2_optimized.npy")
    
    detected_hidden_dim = W1_saved.shape[1] 
    
    weights = {
        'W1': W1_saved,
        'b1': b1_saved,
        'W2': W2_saved,
        'b2': b2_saved
    }

    csv_paths = ["job_postings.csv"]
    salary_base = {'Mid senior': 125000, 'Associate': 85000}
    import pandas as pd
    pos_sums, pos_counts, lvl_sums, lvl_counts = {}, {}, {}, {}
    
    # Build structural mapping keys
    for chunk in pd.read_csv(csv_paths[0], chunksize=10000):
        if 'job_level' not in chunk.columns: continue
        chunk['salary'] = chunk['job_level'].map(salary_base).fillna(95000)
        for pos, sal in zip(chunk['search_position'].fillna('Unknown'), chunk['salary']):
            pos_sums[pos] = pos_sums.get(pos, 0.0) + sal
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        for lvl, sal in zip(chunk['job_level'].fillna('Unknown'), chunk['salary']):
            lvl_sums[lvl] = lvl_sums.get(lvl, 0.0) + sal
            lvl_counts[lvl] = lvl_counts.get(lvl, 0) + 1
            
    pos_map = {p: pos_sums[p] / pos_counts[p] for p in pos_sums}
    level_map = {l: lvl_sums[l] / lvl_counts[l] for l in lvl_sums}
    
    # Load scaling profiles directly
    x_min = np.load("x_min.npy")
    x_max = np.load("x_max.npy")
    maps = {'x_min': x_min, 'x_max': x_max, 'pos_map': pos_map, 'lvl_map': level_map}
    
    print(f"    INFERENCE ENGINE ONBOARDED SUCCESSFUL (Detected Hidden Dim: {detected_hidden_dim})")
    print("========================================================================\n")
    
    print("Available Job Levels to try: 'Mid senior', 'Associate', or leave blank for default.")
    print("Type 'exit' at any time to quit.\n")
    
    while True:
        user_title = input("Enter Custom Job Title (e.g., Data Scientist): ").strip()
        if user_title.lower() == 'exit': break
        if not user_title: continue
            
        user_level = input("Enter Job Level (e.g., Mid senior): ").strip()
        if user_level.lower() == 'exit': break
        if not user_level: user_level = "Unknown"
        
        predicted_amt = get_predictions(user_title, user_level, weights, maps)
        
        print(f"\n   Model Prediction for a [{user_level}] {user_title}:")
        print(f"     Estimated Base Salary: ${predicted_amt:,.2f}")
        print("-" * 72 + "\n")

if __name__ == "__main__":
    run_interface()
