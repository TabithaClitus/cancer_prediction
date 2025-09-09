🩺 Breast Cancer Diagnosis using Machine Learning

This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset to classify tumors as Benign (B) or Malignant (M).
It also includes a Gradio web application with a login page and a prediction interface for real-time diagnosis.

📊 Dataset

Source: UCI Machine Learning Repository

Samples: 569

Features: 30 (tumor characteristics like radius, texture, perimeter, concavity, etc.)

Target:

M → Malignant

B → Benign

🧪 Workflow

Data Preprocessing

Removed duplicates, checked for null values

Scaled features for consistency

Exploratory Analysis

Diagnosis distribution: Benign (63%), Malignant (37%)

PCA for visualization

Model Training

Models tested: Logistic Regression, SVM, Random Forest, Gradient Boosting

Metrics: Accuracy, Precision, Recall, F1-score, AUC

Results

Best Models: SVM & Random Forest (~97% Accuracy, AUC > 0.99)

Cross-validation Accuracy: ~95%

Frontend with Gradio

Login → Username: tabitha, Password: 1234

Prediction Page → Enter tumor features → Predict Benign/Malignant

🖥️ How to Run

Clone or download this repository.

Install dependencies:

pip install pandas numpy scikit-learn matplotlib gradio


Save your trained model as breast_cancer_model.pkl.

Run the Gradio app:

python app.py


📊 Example Results

Confusion Matrix (Random Forest):

[[72  0]
 [ 3 39]]


Accuracy: 97%

Recall (Malignant): 93%