

## CODSOFT NOVEMBER 2023 ML TASK 2 - Customer Review Sentiment Analysis

**Objective:** Analyzing sentiments in customer reviews for enhanced product evaluation and feedback extraction.

This section comprises Python code tailored for sentiment analysis on customer reviews using advanced Natural Language Processing (NLP) techniques. The core objective is to gauge sentiments expressed in textual reviews, providing valuable insights for businesses to assess customer satisfaction and extract meaningful feedback. The techniques employed include data preprocessing, TF-IDF vectorization, and the training of a Multinomial Naive Bayes classifier.

### Libraries and Techniques Utilized:

- **Pandas (`pandas`):** Applied for streamlined data manipulation and analysis.
- **NumPy (`numpy`):** Employed for efficient numerical operations and array handling.
- **Scikit-Learn (`scikit-learn`):** Utilized for essential machine learning tools, encompassing MultinomialNB, TfidfVectorizer, and train_test_split.
- **TQDM (`tqdm`):** Integrated for a visually appealing progress bar during code execution.

### Code Overview:

1. **Import Necessary Libraries:** Inclusion of imperative libraries like pandas, numpy, scikit-learn components, and tqdm for visually engaging progress bars.

2. **Load Customer Reviews Dataset:** Loading of the customer review dataset, encoding it adequately for processing.

3. **Data Preprocessing:** Tasks include removing duplicates, mapping labels to 'positive' and 'negative', and segregating data into features (X) and labels (y).

4. **Train-Test Split:** Division of the dataset into training and testing sets using the `train_test_split` function.

5. **TF-IDF Vectorization:** Initialization of a TF-IDF vectorizer to convert textual data into numerical features.

6. **Fit Vectorizer to Training Data:** Transformation of customer review text data into TF-IDF features for training purposes.

7. **Initialize Naive Bayes Classifier:** Creation of a Multinomial Naive Bayes classifier for sentiment analysis.

8. **Train the Classifier:** Training of the classifier using the TF-IDF transformed training data.

9. **Transform Test Data:** Application of the same vectorizer to transform customer review text data into TF-IDF features for testing.

10. **Make Predictions:** Prediction of sentiment labels for customer reviews using the trained classifier.

11. **Evaluate Model Accuracy:** Calculation of the accuracy of sentiment predictions.

12. **Display Sentiment Analysis Report:** Generation of a comprehensive sentiment analysis report, covering precision, recall, F1-score, and support for both positive and negative sentiments.

### Usage:

1. Ensure the specified libraries are installed as mentioned in the requirements section.

2. Prepare your customer reviews dataset or utilize the provided dataset.

3. Execute the code to preprocess the data, train the Multinomial Naive Bayes classifier, and evaluate the model's performance.

4. Review the accuracy and sentiment analysis report to gauge the model's effectiveness in understanding customer sentiments.

In summary, this code serves as a robust tool for conducting sentiment analysis on customer reviews, offering businesses profound insights into customer satisfaction and valuable feedback. To implement the code, ensure the required libraries are installed, provide your customer reviews dataset, run the code for model training, and evaluate its performance using accuracy metrics and a detailed sentiment analysis report.
