# 🏢 Absenteeism Prediction

This project builds a machine learning model to predict whether an employee is likely to exhibit excessive absenteeism based on multiple personal and work-related factors. The goal is to help organizations take proactive steps in workforce management.

---

## 📂 Project Structure

├── Absenteeism_data.csv
├── Absenteeism_preprocessed.csv
├── Absenteeism_new_data.csv
├── Absenteeism_predictions.csv
├── absenteeism_module.py
├── preprocessing.ipynb
├── ml.ipynb
├── usage.ipynb
├── model/ (saved trained model)
├── scaler/ (saved scaler object)
├── README.md

---

## 📊 Features

- **Input Features**:

  - Reason for Absence
  - Date
  - Transportation Expense
  - Distance to Work
  - Age
  - Daily Work Load Average
  - Body Mass Index (BMI)
  - Education Level
  - Number of Children
  - Number of Pets
  - Absenteeism Time in Hours

- **Target**:
  - Predict whether an employee will have **extremely high absenteeism**.

---

## 🛠️ Tools & Technologies

- **Python**: Data processing and machine learning
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Tableau**: Final visualization of model predictions
- **Jupyter Notebooks**: Model training and usage workflows

---

## 🧠 Methodology

1. **Data Preprocessing (`preprocessing.ipynb`)**

   - Handled missing values and feature engineering.
   - Scaled numerical features.
   - Encoded categorical features.

2. **Model Training (`ml.ipynb`)**

   - Built and trained a logistic regression model to predict absenteeism levels.
   - Evaluated model performance and fine-tuned as needed.

3. **Deployment & Usage (`usage.ipynb`)**

   - Saved the trained model and scaler.
   - Created prediction scripts to apply the model to new unseen data.

4. **Visualization**
   - Prepared prediction outputs for exploration and visualization in Tableau.

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/absenteeism-prediction.git
   cd absenteeism-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run preprocessing:
   Open and run preprocessing.ipynb to process the raw data.

4. Train the model:
   Open and run ml.ipynb to train and evaluate the model.

5. Make predictions:
   Open and run usage.ipynb to use the model for new absenteeism predictions.

🙌 Acknowledgments
This project is part of ongoing work to apply data science techniques to real-world business problems, making workforce management more efficient and predictive.
