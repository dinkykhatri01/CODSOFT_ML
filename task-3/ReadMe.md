# CODSOFT November 2023 ML Task 3 - CREDIT CARD FRAUD DETECTION


**Objective:** Detect Credit Card Fraud with Precision

Welcome to our Fraud Detection repository, tailored for simplicity and effectiveness. This Python code focuses on detecting credit card fraud using a Random Forest classifier. Let's navigate through the key steps without unnecessary complexity:

## Essential Tools

- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Scikit-Learn:** For machine learning tools like RandomForestClassifier, StandardScaler, LabelEncoder, OneHotEncoder, and IncrementalPCA.
- **Imbalanced-Learn:** For handling class imbalance through SMOTE.
- **TQDM:** For progress bars during data processing.
## Requirements
- Python 3.x
- Pandas (`pandas`)
- NumPy (`numpy`)
- Scikit-Learn (`scikit-learn`)
- TQDM (`tqdm`)
- Imbalanced-Learn (`imblearn`)
## Key Steps

1. **Import Necessary Tools:** Load essential libraries—pandas, numpy, scikit-learn components, imblearn, and tqdm.

2. **Load and Combine Data:** Load training and testing data from CSV files (fraudTrain.csv and fraudTest.csv). Combine them for consistency.

3. **Feature Extraction:** Extract relevant features from the "trans_date_trans_time" column, like the day of the week and hour of the day.

4. **Drop Irrelevant Columns:** Remove columns irrelevant for fraud detection, streamlining your dataset.

5. **Separate Features and Target Variable:** Divide the dataset into features (X_combined) and the target variable (y_combined).

6. **Encode Categorical Columns:** Use LabelEncoder for "merchant" and "category" columns. Apply OneHotEncoder for other categorical variables.

7. **Standardize Numeric Features:** Bring uniformity to numeric features with StandardScaler.

8. **Combine Encoded Categorical and Numeric Features:** Merge one-hot encoded categorical features with standardized numeric features.

9. **Split Data for Training and Testing:** Divide the combined data back into training and test datasets.

10. **Handle Class Imbalance with SMOTE:** Address imbalances in the training data using SMOTE.

11. **Dimensionality Reduction with Incremental PCA:** Apply Incremental PCA to reduce dimensionality.

12. **Train the Random Forest Model:** Define and train a Random Forest classifier with the refined data.

13. **Make Predictions:** Use the trained model to predict fraud detection results.

14. **Evaluate Model Performance:** Calculate accuracy, display a confusion matrix, and generate a classification report.

## Usage

1. **Check Your Toolbox:** Ensure you have Python 3.x and the required libraries installed—pandas, numpy, scikit-learn, tqdm, and imblearn.

2. **Prepare Your Data:** Place your training and testing data in CSV files named fraudTrain.csv and fraudTest.csv.

3. **Run the Code:** Execute the code to preprocess the data, train the Random Forest model, and evaluate fraud detection performance.

4. **Review Results:** Examine the output, including accuracy, the confusion matrix, and the classification report to understand the model's effectiveness.

