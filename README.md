# 🎗️ Breast Cancer Prediction System

A machine learning-based **Breast Cancer Prediction System** that predicts whether a breast tumor is **Benign (B)** or **Malignant (M)** based on five important tumor-related features.

The project uses a trained machine learning model and provides an interactive **Streamlit web application** where users can enter the required feature values and receive a prediction.

> **Model Accuracy: 96%**

---

## 📌 Project Overview

Breast cancer is one of the most common types of cancer. Early detection can play an important role in diagnosis and treatment.

This project demonstrates how **Machine Learning** can be used to classify breast tumors based on numerical features extracted from diagnostic measurements.

The system takes **5 inputs** from the user and uses a trained classification model to predict whether the tumor is:

* 🟢 **Benign (B)** – Non-cancerous
* 🔴 **Malignant (M)** – Cancerous

---

## 🎯 Objectives

* Build a machine learning model for breast cancer classification.
* Identify whether a tumor is benign or malignant.
* Use important tumor-related features as model inputs.
* Achieve a reliable prediction accuracy.
* Create an easy-to-use web interface using Streamlit.
* Demonstrate the practical application of machine learning in healthcare.

---

## 🧠 Machine Learning

The model is trained using the **Breast Cancer Wisconsin Diagnostic Dataset**.

The target variable is:

| Diagnosis | Meaning   |
| --------- | --------- |
| `M`       | Malignant |
| `B`       | Benign    |

For machine learning, the diagnosis values are converted into numerical labels:

```text
M → 1
B → 0
```

### 📊 Input Features

The application uses the following five features:

1. `radius_mean`
2. `perimeter_mean`
3. `area_mean`
4. `texture_mean`
5. `radius_worst`

These features describe different characteristics of the cell nuclei observed in breast tissue samples.

---

## 📈 Model Performance

The trained model achieved approximately:

### **96% Accuracy**

Accuracy is calculated as:

```text
Accuracy = Correct Predictions / Total Predictions
```

The 96% accuracy indicates that the model correctly classified approximately 96 out of every 100 samples in the evaluation dataset.


---

## 🖥️ Application

The project includes a **Streamlit web application**.

Users enter values for the five required features, and the application provides a prediction.

### Input

```text
Radius Mean
Perimeter Mean
Area Mean
Texture Mean
Radius Worst
```

### Output

The system predicts:

```text
Benign
```

or

```text
Malignant
```

---


## 🛠️ Technologies & Libraries Used

* 🐍 **Python** – Programming language
* 🤖 **Scikit-learn** – Machine learning model development and evaluation
* 📊 **Pandas** – Data loading and data preprocessing
* 🔢 **NumPy** – Numerical computations
* 📈 **Matplotlib** – Data visualization and graphical analysis
* 🌐 **Streamlit** – Interactive web application
* 📦 **Joblib** – Saving and loading the trained machine learning model
* 💻 **VS Code** – Development environment
* 🐙 **Git & GitHub** – Version control and project hosting


---

## 📂 Project Structure

```text
BreastCancer_Prediction/
│
├── .vscode/
│   └── settings.json
│
├── images/
│   ├── benign prediction.png
│   ├── classification matrices.png
│   ├── confusion matrix.png
│   ├── frontend interface.png
│   ├── malignant prediction.png
│   ├── visualization between perimeter...
│   └── visualization between radius...
│
├── .gitignore
├── README.md
├── app.py
├── breast_cancer_model.pkl
├── dataset
└── prediction.py


---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sainkinjal13/BreastCancer_Prediction.git

### 2. Move into the project directory

```bash
cd BreastCancer_Prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

For Windows:

```bash
.venv\Scripts\activate
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually, Streamlit runs at:

```text
http://localhost:8501
```

---

## 📋 Example Input

An example input could look like:

| Feature        | Example Value |
| -------------- | ------------: |
| Radius Mean    |          14.5 |
| Perimeter Mean |          95.0 |
| Area Mean      |         650.0 |
| Texture Mean   |          18.0 |
| Radius Worst   |          16.0 |

The model then processes these values and generates a classification prediction.

**Important:** Example values are only for demonstrating the application and should not be interpreted as medical thresholds.

---

## 🔄 Machine Learning Workflow

```text
        Dataset
           ↓
    Data Preprocessing
           ↓
    Feature Selection
           ↓
    Train-Test Split
           ↓
    Model Training
           ↓
    Model Evaluation
           ↓
    Model Saving
           ↓
    Streamlit Application
           ↓
       Prediction
   ┌───────────────┐
   ↓               ↓
Benign         Malignant
```

---

## 📊 Features of the Application

* Simple and user-friendly interface
* Five numerical inputs
* Machine learning-based prediction
* Fast prediction
* Interactive Streamlit interface
* Approximately 96% model accuracy

---


---

## 🚀 Future Improvements

Some possible improvements include:

* Adding more clinically relevant features.
* Comparing multiple machine learning algorithms.
* Adding confusion matrix visualization.
* Adding precision, recall, F1-score, and ROC-AUC.
* Improving the user interface.
* Adding data visualization.
* Deploying the application online.
* Adding model explainability using techniques such as SHAP.
* Improving validation using cross-validation.


