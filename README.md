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
