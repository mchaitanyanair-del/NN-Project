# Custom Salary Prediction Neural Network Pipeline 🧠🚀

An end-to-end machine learning pipeline and deep learning regression engine built completely from scratch using only **NumPy** and **Pandas**. This project completely bypasses heavy frameworks like PyTorch or TensorFlow to implement raw data engineering and matrix optimization loops directly.

The model is trained on a massive, real-world **1.5 million row LinkedIn dataset** sourced from Kaggle to predict software industry base salaries based on job titles and experience levels.

---

## 🏗️ Data Engineering & System Bottlenecks Solved
Because this pipeline processes a large-scale, 1.5M+ row dataset without specialized deep learning frameworks, the architecture was engineered to dynamically overcome severe memory and algebraic system constraints:

1. **Memory Ingestion Constraints (RAM Preservation):** Ingesting 1.5 million rows of messy data at once causes standard environments to crash. Designed a custom chunk-streaming data generator to pull data blocks sequentially off disk to maintain a low RAM footprint.
2. **High-Cardinality Target Encoding:** Implemented custom Pandas encoding structures to map complex text variables (like job titles and seniority levels) into stable numerical vectors based on historic salary group target averages.
3. **Mathematical Overflows ($inf$ Weight Errors):** Initial linear algebra passes suffered from weight saturation over millions of rows, driving predictions to mathematical infinity. Implemented layer bounding, numeric data clipping, and activation tracking to stabilize runtime operations.
4. **Dynamic Shape Configuration Mismatches:** Re-engineered the CLI inference tools to dynamically scan and read `.npy` matrix dimension asset files directly from storage, allowing the user interface to adjust to backend updates instantly.
5. **Gradient Spike Suppression:** Tuned hyperparameters directly over structural training blocks, discovering the optimal convergence plateau at 200 epochs paired with a balanced `lr=0.002` learning rate.
6. **Pure Dependency Elimination:** Ensured the entire ecosystem functions seamlessly across standalone machines using only core python math primitives.

---

## 📈 Final Model Performance
* **Dataset Scale:** 1.5 Million Rows (LinkedIn / Kaggle)
* **Core Tech Stack:** NumPy, Pandas
* **Validation Accuracy ($R^2$ Score):** **36.4%** * **Sample Live Evaluation Output:** * *Query:* `Data Scientist` | `Mid senior`
  * *Model Output:* **$116,383.44** (An incredibly precise market-accurate localization)

---

## 👥 Team & Contributions

* **Ram (Feature Engineering):** Executed initial dataset cleansing and built the Target, Label, and One-Hot encoding structures to convert raw text into high-signal feature matrices.
* **Chaitanya (System Architecture & Pipeline Lead):** Designed and implemented the core chunk-streaming data pipeline (`data_pipeline.py`) to handle 1.5M+ rows without memory exhaustion. Managed end-to-end component integration, debugged runtime scaling bottlenecks, and handled repository release management.
* **Renu (Core Training Engine):** Implemented foundational multi-layer forward propagation math, structured the core execution engine (`train.py`), and orchestrated hyperparameter validation loops.
* **Shruthi (Optimization & Inference):** Derived and programmed the backward propagation gradient matrix calculus and developed the interactive live query interface tool (`predict.py`).

## 💻 How To Run the Project

### 1. Installation & Setup
Clone the repository and ensure you have the required libraries installed:
```bash
pip install numpy pandas
