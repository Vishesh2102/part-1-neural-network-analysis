
# Project: Customer Churn Neural Network Analysis

## 1. Overview
This project involves building a feed-forward neural network to predict customer churn. The focus was on analyzing how different architectures and learning rates perform on a highly imbalanced dataset (98.5% retention vs 1.5% churn).

---

## 2. Data Preparation
*   **Cleaning:** Dropped `customer_id` as it is a non-predictive identifier.
*   **Encoding:** Applied **One-Hot Encoding** to categorical columns (region, plan_type, etc.).
*   **Scaling:** Used **StandardScaler** to normalize features like `monthly_charges_inr` and `tenure_months`.
*   **Split:** Utilized an 80/20 train-test split with **stratification** to ensure the rare churn cases were present in both sets.

---

## 3. Architecture & Training
*   **Structure:** Input layer (24 features) → 2 Hidden Layers (32 neurons) → Output Layer.
*   **Activations:** **ReLU** for hidden layers; **Sigmoid** for the output layer.
*   **Optimization:** **Adam** optimizer with **Binary Crossentropy** loss.

---

## 4. Experiment Results
My testing showed that a slower learning rate with higher neuron density provided the most stable accuracy.

| Run Name | Layers | Neurons | Learning Rate | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **Simple_Fast** | 1 | 16 | 0.01 | 98.0% |
| **Deep_Standard** | 4 | 64 | 0.001 | 97.5% |
| **Complex_Slow** | 2 | 128 | 0.0001 | 98.5% |

## 5. Final Reflection

### Weights, Biases, and Activations
*   **Weights & Biases:** Weights scale the importance of features (e.g., payment delays), while biases allow the model to shift its activation threshold.
*   **Activation Functions:** **ReLU** was critical for learning non-linear patterns. Without it, the model would effectively be a limited linear regressor.

### Learning Rate Sensitivity
*   **High LR (0.01):** The "Simple_Fast" model converged quickly but was less precise.
*   **Low LR (0.0001):** The "Complex_Slow" model achieved the highest accuracy (98.5%) because the smaller steps allowed it to settle into a better local minimum.

### Bias-Variance Tradeoff (Observation)
*   **Overfitting:** The "Deep_Standard" model (4 layers) actually performed slightly worse on the test set than the simpler models. This indicates that adding depth to such an imbalanced dataset can cause the model to memorize noise rather than general patterns.
*   **Imbalance Note:** While accuracy was high (~98%), the confusion matrix shows the model struggled to correctly identify the "Churn" class due to the limited number of positive samples in the data.

---

## 6. Repository Contents
*   `notebook.ipynb`: Full implementation and analysis.
*   `requirements.txt`: Environment dependencies (TensorFlow 2.21.0, etc.).
*   `results/evaluation_outputs.png`: Baseline Confusion Matrix.
*   `results/model_comparison_table.csv`: Hyperparameter test logs.