# 🩺 PneumoScan - Pneumonia Detection using Deep Learning

## 📌 Project Overview

**PneumoScan** is a deep learning-based system designed to detect pneumonia from chest X-ray images. It classifies images into **Normal** or **Pneumonia** categories. The project leverages **EfficientNetB0** and **InceptionV3**, with **InceptionV3** achieving the best performance. The final model is deployed using **Streamlit** and hosted on **Hugging Face Spaces**.

---

## 📂 Dataset

The dataset used for training and evaluation is the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

📥 **Download Dataset**: [Chest X-Ray Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

### Dataset Structure:
```
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
```

---

## 📊 Model Selection & Evaluation

| Model          | Test Accuracy |
|---------------|--------------|
| EfficientNetB0 | 87.34%       |
| InceptionV3   | **91.51%**   |

✅ **InceptionV3** was chosen for deployment due to its superior accuracy.

🔍 **Explainability**: *Grad-CAM* was used to visualize the regions of the image that the model focuses on.

---

## 🛠️ Implementation Steps

### 🔬 Training & Evaluation
✔ Model training was conducted in **Google Colab**.  
✔ The final model training was performed in `PneumoScan_InceptionV3.ipynb`.  
✔ The trained model was saved as **`inception_model.keras`**.  

### 🚀 Deployment Setup
- **Local Deployment**: Streamlit + Ngrok
- **Public Deployment**: Hugging Face Spaces

---

## 🚀 Deployment

### 🔹 Local Deployment using Colab
1️⃣ **Mount Google Drive** and access the saved model.  
2️⃣ Install dependencies:
   ```bash
   pip install streamlit tensorflow opencv-python numpy pyngrok
   ```
3️⃣ Run the app locally:
   ```bash
   !streamlit run app.py &
   !ngrok authtoken YOUR_NGROK_TOKEN
   !ngrok http 8501
   ```
4️⃣ Open the generated **ngrok URL** to access the app.

### 🔹 Public Deployment on Hugging Face Spaces
📍 **Live App**: [PneumoScan on Hugging Face](https://huggingface.co/spaces/usehgal6/PneumoScan)

### Files Used in Deployment:
- `app.py`: Streamlit application for pneumonia classification.
- `inception_model.keras`: Trained deep learning model.
- `requirements.txt`: Dependencies for Hugging Face deployment.

---

## 📜 File Structure
```
📂 PneumoScan/
├── 📄 PneumoScan.ipynb              # Initial model training (EfficientNetB0 & InceptionV3)
├── 📄 PneumoScan_InceptionV3.ipynb  # Final model training (InceptionV3 only)
├── 📄 Deployment_Main.ipynb         # Colab-based Streamlit deployment (ngrok)
├── 📄 app.py                        # Streamlit application
├── 📄 requirements.txt               # Dependencies
├── 📄 inception_model.keras         # Trained model file
```

---

## 📌 How It Works
1️⃣ Upload a **chest X-ray image**.  
2️⃣ The image is **preprocessed** and fed into the **InceptionV3** model.  
3️⃣ The model predicts **Normal** or **Pneumonia** with a confidence score.  

---

## 🛠️ Technologies Used
✅ **Deep Learning**: TensorFlow, Keras, InceptionV3, EfficientNetB0  
✅ **Computer Vision**: OpenCV, NumPy  
✅ **Model Explainability**: Grad-CAM  
✅ **Deployment**: Streamlit, Hugging Face, Ngrok  

---

## 📌 Future Improvements
🚀 **Enhancing Model Performance**: Experimenting with other CNN architectures.  
🔍 **Explainability Enhancements**: Integrating SHAP for better model interpretation.  
☁ **Better Deployment Options**: Exploring Docker, AWS, or Heroku for scalable hosting.  

---

## 🏆 Contributors
👨‍💻 **Utkarsh Sehgal**  
📧 **Contact**: [LinkedIn](www.linkedin.com/in/utkarsh-sehgal/)

💡 If you find this project useful, don't forget to ⭐ **star the repository**! 🌟
