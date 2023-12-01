# CODSOFT November 2023 ML Task 2 SMS Spam Detection System

**CODSOFT NOVEMBER 2023 ML TASK 2 - Spam SMS Detection**

### Overview

Welcome to the SMS Spam Detection System repository! This project focuses on identifying and classifying spam SMS messages through advanced Natural Language Processing (NLP) techniques. The primary goal is to create an AI model capable of accurately categorizing SMS messages as spam or legitimate. The implemented techniques involve TF-IDF vectorization and a Multinomial Naive Bayes classifier.

### Libraries and Techniques Utilized

- **Pandas (pandas):** Employed for streamlined data manipulation and analysis.
- **NumPy (numpy):** Utilized for efficient numerical operations.
- **Scikit-Learn (scikit-learn):** Essential for machine learning tools, featuring MultinomialNB, TfidfVectorizer, and train_test_split.
- **TQDM (tqdm):** Integrated for visually appealing progress bars during processing.

### Code Overview

1. **Import Necessary Libraries:** Inclusion of imperative libraries like pandas, numpy, scikit-learn components, and tqdm for visually engaging progress bars.

2. **Load the SMS Spam Collection dataset:** Load the SMS dataset from 'spam.csv', ensuring proper encoding with 'latin-1'.

3. **Preprocess the data:** Tasks include eliminating duplicates, mapping labels to 'ham' (legitimate) and 'spam', and splitting the data into features (X) and labels (y).

4. **Split the data into training and testing sets:** Divide the dataset into training and testing sets using train_test_split.

5. **Create a TF-IDF vectorizer:** Initialize a TF-IDF vectorizer to convert text data into numerical features.

6. **Fit the vectorizer to the training data:** Transform the SMS text data into TF-IDF features for training purposes.

7. **Initialize a Naive Bayes classifier:** Creation of a Multinomial Naive Bayes classifier.

8. **Train the classifier:** Train the classifier using the TF-IDF transformed training data.

9. **Transform the test data:** Utilize the same vectorizer to transform the SMS text data into TF-IDF features for testing.

10. **Make predictions:** Predict whether SMS messages are spam or legitimate using the trained classifier.

11. **Calculate accuracy:** Determine the accuracy of the model's predictions.

12. **Display classification report:** Generate a comprehensive classification report that includes precision, recall, F1-score, and support for both 'Legitimate SMS' and 'Spam SMS'.

### Usage

1. Ensure you have the required libraries installed, as mentioned in the requirements section.

2. Prepare your SMS dataset or utilize the provided 'spam.csv' dataset.

3. Run the provided code to preprocess the data, train the Multinomial Naive Bayes classifier, and evaluate the model's performance.

4. Review the accuracy and the classification report to assess the model's effectiveness in detecting spam SMS messages.

### TL;DR

This code implements a robust spam SMS detection model using TF-IDF vectorization and Multinomial Naive Bayes classification. It encompasses data preprocessing and evaluation, aiming to accurately classify SMS messages as spam or legitimate. To use the code, ensure required libraries are installed, provide your SMS dataset or use the provided one, run the code to train the model, and evaluate its performance using accuracy and a detailed classification report for spam and legitimate SMS classification.

### Requirements

- Python 3.x
- Pandas (`pandas`)
- NumPy (`numpy`)
- Scikit-Learn (`scikit-learn`)
- TQDM (`tqdm`)
