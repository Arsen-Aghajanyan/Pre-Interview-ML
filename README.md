# Pre-Interview Assessment (Machine Learning Project)

## 📌 Project Overview
In today’s competitive job market, employers increasingly rely on pre-employment
assessments to identify suitable candidates. With the help of artificial intelligence,
the recruitment process becomes more efficient, reliable, and cost-effective.

This project applies machine learning classification techniques to predict whether
a candidate will be accepted for an interview based on their profile data.

---

## 🎯 Project Objective
The main goal of this project is to build and compare multiple classification models
that predict interview acceptance and evaluate their performance using standard
machine learning metrics.

---

## 📊 Dataset
- File: `logatta.csv`
- Location: `data/logatta.csv`
- Target variable: **accepted for the interview**
- The dataset contains both numerical and categorical features describing candidates.

---

## 🛠 Project Workflow
1. Load the dataset
2. Import required Python libraries
3. Separate features and target variable
4. Preprocess the data:
   - Encode categorical features using `OrdinalEncoder`
   - Scale numerical features using `StandardScaler`
5. Split the dataset into training and testing sets
6. Train multiple classification models
7. Evaluate each model using:
   - Accuracy
   - Classification report
   - Confusion matrix
8. Visualize and compare model performance
9. Analyze results and draw conclusions

---

## 🧠 Models Used
- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors (KNN)

---

## 📈 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## 📊 Visualization
- Confusion matrix heatmaps for each model
- Bar chart comparing accuracy of all models

---

## 🛠 Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## 🚀 How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/your-username/pre-interview-assessment-ml.git
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the notebook
```bash
jupyter notebook notebooks/ml.ipynb
```

---

---

## 📌 Results

Three classification models were trained and evaluated on the test dataset:
Logistic Regression, Naive Bayes, and K-Nearest Neighbors (KNN).

---

### 🔹 Logistic Regression

Logistic Regression Metrics
Report: 
               precision    recall  f1-score   support

      False       0.93      0.98      0.95       268
      True        0.65      0.35      0.46        31

    accuracy                           0.91       299
   macro avg       0.79      0.67      0.71       299
weighted avg       0.90      0.91      0.90       299

---

### 🔹 Naive Bayes

Naive Bayes Report
              precision    recall  f1-score   support

       False       1.00      0.86      0.92       268
        True       0.44      0.97      0.61        31

    accuracy                           0.87       299
   macro avg       0.72      0.91      0.76       299
weighted avg       0.94      0.87      0.89       299

---

### 🔹 K-Nearest Neighbors (KNN)

KNN Report
              precision    recall  f1-score   support

       False       0.95      1.00      0.97       268
        True       0.94      0.55      0.69        31

    accuracy                           0.95       299
   macro avg       0.95      0.77      0.83       299
weighted avg       0.95      0.95      0.94       299
