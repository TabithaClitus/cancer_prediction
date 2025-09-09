🩺 Breast Cancer Diagnosis using Machine Learning

APP LINK:https://huggingface.co/spaces/tabithaclitus/breast-cancer-diagnosis

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

<img width="571" height="453" alt="image" src="https://github.com/user-attachments/assets/8b0861be-106b-4529-8773-488147aa2427" />

PCA for visualization

<img width="547" height="470" alt="image" src="https://github.com/user-attachments/assets/cb40858f-2857-49b1-a84a-b5ea99740e49" />

Model Training

Models tested: Logistic Regression, SVM, Random Forest, Gradient Boosting

Metrics: Accuracy, Precision, Recall, F1-score, AUC

Results

<img width="613" height="470" alt="image" src="https://github.com/user-attachments/assets/01e62ea8-4f45-4d64-892b-5dd37256b61e" />

Best Models: SVM & Random Forest (~97% Accuracy, AUC > 0.99)

<img width="688" height="451" alt="image" src="https://github.com/user-attachments/assets/f0f66e05-5ff4-4c3b-9a1c-00ac1424cdf2" />

Cross-validation Accuracy: ~95%

Frontend with Gradio

<img width="1599" height="456" alt="Screenshot 2025-09-09 114134" src="https://github.com/user-attachments/assets/26bf11f3-0244-430e-ad34-945f571116a2" />

<img width="1806" height="840" alt="Screenshot 2025-09-09 114232" src="https://github.com/user-attachments/assets/d7feb31d-7353-4df0-a13b-272e5de58e2e" />

<img width="1800" height="596" alt="Screenshot 2025-09-09 114249" src="https://github.com/user-attachments/assets/a6c75f50-2630-452e-a843-3cebf34c209a" />

Login → Username: tabitha, Password: 1234

Prediction Page → Enter tumor features → Predict Benign/Malignant

🖥️ How to Run

Clone or download this repository.

Install dependencies:

pip install pandas numpy scikit-learn matplotlib gradio

Save trained model as breast_cancer_model.pkl.

Run the Gradio app:

python app.py


📊 Example Results

Confusion Matrix (Random Forest):

[[72  0]
 [ 3 39]]


Accuracy: 97%


Recall (Malignant): 93%
