# README.md

## Task-1 : ML Movie Genre Prediction System

**CODSOFT NOVEMBER 2023 ML TASK - Movie Genre Classification**

### Overview

Welcome to the Movie Genre Prediction System repository! This project focuses on predicting movie genres based on their plots using a MultiOutput Naive Bayes approach. The system is designed to learn, identify, and classify movie genres with the help of machine learning techniques.

### Libraries and Techniques Used

- **Pandas (pandas):** For efficient data manipulation and analysis.
- **NumPy (numpy):** Essential for numerical operations.
- **Scikit-Learn (scikit-learn):** Utilized for machine learning tools, featuring TfidfVectorizer, MultiOutputClassifier, MultinomialNB, and MultiLabelBinarizer.
- **TQDM (tqdm):** Provides visually appealing progress bars during data processing.

### Code Overview

1. **Import Necessary Libraries:** Start by importing essential libraries, including pandas, numpy, scikit-learn components, and tqdm for progress tracking.

2. **Define Genre List:** Create a list of movie genres that will be employed for classification.

3. **Define Fallback Genre:** Specify a fallback genre for movies with no predicted genre.

4. **Load Training Data:** Load training data from `train_data.txt` using pandas. Each row contains a movie serial number, name, genre(s), and plot.

5. **Data Preprocessing for Training Data:** Prepare training data by converting movie plot text to lowercase and encoding genre labels using MultiLabelBinarizer.

6. **TF-IDF Vectorization:** Utilize TfidfVectorizer to convert movie plot text into TF-IDF features.

7. **Train MultiOutput Naive Bayes Classifier:** Train a MultiOutputClassifier using a Multinomial Naive Bayes classifier with the training data.

8. **Load Test Data:** Load test data from `test_data.txt` using pandas, including movie serial number, name, and plot.

9. **Data Preprocessing for Test Data:** Preprocess test data by converting movie plot text to lowercase.

10. **Vectorize Test Data:** Transform test data using the same TF-IDF vectorizer used for training.

11. **Predict Genres on Test Data:** Predict movie genres on the test data using the trained model.

12. **Create Results DataFrame:** Generate a DataFrame containing movie names and predicted genres.

13. **Replace Empty Predicted Genres:** Replace empty predicted genres with the fallback genre.

14. **Write Results to a Text File:** Save prediction results to `model_evaluation.txt` with proper formatting and UTF-8 encoding.

15. **Calculate Evaluation Metrics:** Calculate evaluation metrics, including accuracy, precision, recall, and F1-score, using training labels as a proxy.

16. **Append Metrics to the Output File:** Append the evaluation metrics to the `model_evaluation.txt` file.

### Usage

1. Specify the configuration file (`config_file`) and frozen model file (`frozen_model`) for the pre-trained model.

2. Load class labels from a text file (`file_name`) to recognize objects.

3. Configure model input size, scaling, and color format.

4. Load an image or a video for object detection and recognition.

5. Display the recognized objects, their bounding boxes, and confidence scores.

### Requirements

- Python 3.x
- Pandas (`pandas`)
- NumPy (`numpy`)
- Scikit-Learn (`scikit-learn`)
- TQDM (`tqdm`)

### TL;DR

This code leverages MultiOutput Naive Bayes to predict movie genres based on plot descriptions, involving data preprocessing, TF-IDF vectorization, and evaluation metrics calculation. To use the code, ensure required libraries are installed, provide training and test data, run the code, and review genre predictions and evaluation metrics in `model_evaluation.txt`.


