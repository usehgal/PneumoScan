# 🩺 PneumoScan - Pneumonia Detection using Deep Learning

📌 Project Overview

PneumoScan is a deep learning-based pneumonia detection system that classifies chest X-ray images into Normal or Pneumonia categories. The model was trained and evaluated using two state-of-the-art architectures: EfficientNetB0 and InceptionV3, with InceptionV3 yielding the best performance. The final model was deployed using Streamlit and hosted on Hugging Face Spaces.

📂 Dataset

The dataset used for training and evaluation is the Chest X-Ray Images (Pneumonia) dataset from Kaggle.

📥 Download Dataset: Chest X-Ray Dataset (https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset is structured as follows:

chest_xray/
├── train/
│   ├── NORMAL/
│   ├── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   ├── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   ├── PNEUMONIA/

📊 Model Selection & Evaluation

Two deep learning models were used for comparison:

Model

Test Accuracy

EfficientNetB0

87.34%

InceptionV3

91.51%

The InceptionV3 model achieved the highest accuracy and was chosen for deployment.

Grad-CAM was used for explainability, highlighting the regions the model focused on.

🛠️ Implementation Steps

Training & Evaluation:

Initial model training was done in Google Colab.

The PneumoScan_InceptionV3 notebook was created for final model training.

The trained model was saved as inception_model.keras.

Deployment Setup:

Local Deployment using Streamlit & Ngrok.

Public Deployment on Hugging Face Spaces.

🚀 Deployment

🔹 Local Deployment using Colab

Mount Google Drive and access the saved model.

Install dependencies:

pip install streamlit tensorflow opencv-python numpy pyngrok

Run the app locally using ngrok:

!streamlit run app.py &
!ngrok authtoken YOUR_NGROK_TOKEN
!ngrok http 8501

Open the generated ngrok URL to access the app.

🔹 Public Deployment on Hugging Face Spaces

The model is publicly available at: 👉 Live App: PneumoScan on Hugging Face (https://huggingface.co/spaces/usehgal6/PneumoScan)

Files used in deployment:

app.py: Streamlit application for pneumonia classification.

inception_model.keras: Trained deep learning model.

requirements.txt: Dependencies for Hugging Face deployment.

📜 File Structure

📂 PneumoScan/
├── 📄 PneumoScan.ipynb           # Initial model training (EfficientNetB0 & InceptionV3)
├── 📄 PneumoScan_InceptionV3.ipynb  # Final model training (InceptionV3 only)
├── 📄 Deployment_Main.ipynb      # Colab-based Streamlit deployment (ngrok)
├── 📄 app.py                     # Streamlit application
├── 📄 requirements.txt            # Dependencies
├── 📄 inception_model.keras       # Trained model file

📌 How It Works

Upload a chest X-ray image.

The image is preprocessed and passed through the InceptionV3 model.

The app predicts Normal Lungs or Pneumonia with a confidence score.

🛠️ Technologies Used

Deep Learning: TensorFlow, Keras, InceptionV3, EfficientNetB0

Computer Vision: OpenCV, NumPy

Model Explainability: Grad-CAM

Deployment: Streamlit, Hugging Face, Ngrok

📌 Future Improvements

Enhancing Model Performance: Experimenting with other CNN architectures.

Explainability Enhancements: Integrating SHAP for model interpretation.

Better Deployment Options: Exploring Docker, AWS, or Heroku for scalable hosting.

🏆 Contributors

👨‍💻 Utkarsh Sehgal

📧 Contact: LinkedIn

📢 If you find this project useful, don't forget to ⭐ the repository!

