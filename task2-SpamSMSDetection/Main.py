# SMS Guardian - CODSOFT NOVEMBER 2023 ML TASK
## Vigilant Protector of Textual Integrity
# Welcome to SMS Guardian, your silent protector against the intrusion of spam into the realm of SMS conversations. Below is an evolved version of the code, preserving the essence of logic while presenting a fresh perspective.

# Importing Essential allies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Accessing the SMS Archives with Respectful Encoding
data = pd.read_csv('spam.csv', encoding='latin-1')

# Initiating a Pristine Cleansing Ritual
data.drop_duplicates(inplace=True)
data['label'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})
X = data['v2']
y = data['label']

# Dividing the Forces: Training Set and Testing Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crafting a Textual Sorcerer - TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Infusing the Vectorizer with the Wisdom of Training Data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Summoning the Naive Guardian for Training
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Unleashing the Trained Guardian on Testing Grounds
X_test_tfidf = tfidf_vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_tfidf)

# Calculating the Vigilance Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Displaying the Scrolls of Truth
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS']))

# Closing the Guardian's Report with a Progress Salute
for _ in tqdm(range(10), desc='Completion', position=0, leave=True):
    pass
