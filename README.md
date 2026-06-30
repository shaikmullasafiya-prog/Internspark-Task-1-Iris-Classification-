# Internspark-Task-1-Iris-Classification-

Intern: Shaik Mulla Safiya 
Task:Build, Benchmark, and Package an Iris Species Classification Engine

## Project Structure
- `Task1_Iris_Final_Report.pdf`: Document containing all technical explanations, comparative benchmarks, and final outputs.
- `best_iris_model.pkl`: Serialized model engine file (Logistic Regression).
- `iris_scaler.pkl`: Serialized data normalizer mapping original input ranges.

## Model Benchmarking Performance Table
| Model Variant | Training Accuracy | Precision (Macro) | Recall (Macro) |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **100%** | **1.00** | **1.00** |
| **k-Nearest Neighbors (k=5)** | 100% | 1.00 | 1.00 |
| **Decision Tree Classifier** | 96.67% | 0.97 | 0.97 |

---

## Live Inference Execution Code
To run inference workloads utilizing these generated parameters, execute the snippet below in your Python environment:

```python
import joblib
import numpy as np

# Load assets
scaler = joblib.load('iris_scaler.pkl')
model = joblib.load('best_iris_model.pkl')

# Feature formatting Matrix: [sepal_length, sepal_width, petal_length, petal_width]
custom_inputs = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Expected target class: Iris-setosa
    [6.2, 2.9, 4.3, 1.3]   # Expected target class: Iris-versicolor
])

# Scale and Predict
transformed_inputs = scaler.transform(custom_inputs)
predictions = model.predict(transformed_inputs)

for i, outcome in enumerate(predictions):
    print(f"Sample Input {custom_inputs[i]} -> Target Predicted Class: {outcome}")
