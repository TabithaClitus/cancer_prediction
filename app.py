import gradio as gr
import pickle
import numpy as np

# --- Load scaler and model ---
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("breast_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

# --- Login credentials ---
USERNAME = "tabitha"
PASSWORD = "1234"

# --- Login Function ---
def login(username, password):
    if username == USERNAME and password == PASSWORD:
        return gr.update(visible=False), gr.update(visible=True), ""
    else:
        return gr.update(visible=True, value="❌ Invalid credentials"), gr.update(visible=False), ""

# --- Prediction Function ---
def predict(*features):
    try:
        features = np.array([features])  
        features_scaled = scaler.transform(features)
        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

        diagnosis = "Malignant (M)" if pred == 1 else "Benign (B)"
        return f"Prediction: {diagnosis}\nProbability of Malignant: {prob:.2f}"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- Important Features Only ---
feature_names = [
    "mean radius",
    "mean texture",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension"
]

# --- Gradio Interface ---
with gr.Blocks() as demo:
    # --- Login Page ---
    login_box = gr.Group(visible=True)
    with login_box:
        gr.Markdown("## 🔐 Login to Breast Cancer Prediction App")
        with gr.Row():
            username = gr.Textbox(label="Username")
            password = gr.Textbox(label="Password", type="password")
        login_button = gr.Button("Login")
        login_msg = gr.Label()

    # --- Prediction Page ---
    pred_box = gr.Group(visible=False)
    with pred_box:
        gr.Markdown("## 🩺 Breast Cancer Diagnosis Prediction (Important Features Only)")

        inputs = []
        with gr.Row():
            for i, feature in enumerate(feature_names):
                inputs.append(gr.Number(label=feature))
                if (i+1) % 2 == 0:  # arrange neatly in 2 columns
                    pass

        pred_button = gr.Button("Predict")
        output = gr.Textbox(label="Result", interactive=False)

    # --- Actions ---
    login_button.click(login, [username, password], [login_msg, pred_box, login_msg])
    pred_button.click(predict, inputs, output)

# --- Launch ---
demo.launch(share=True)


