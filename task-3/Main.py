import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

# Load training and testing datasets
train_data = pd.read_csv("fraudTrain.csv")
test_data = pd.read_csv("fraudTest.csv")

# Combine and preprocess data
def preprocess_data(df):
    # Extract datetime features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['hour_of_day'] = df['trans_date_trans_time'].dt.hour
    df.drop(['trans_date_trans_time', 'first', 'last', 'job', 'dob', 'trans_num', 'street'], axis=1, inplace=True)
    return df

combined_data = preprocess_data(pd.concat([train_data, test_data], axis=0))

# Separate features and target variable
X_combined = combined_data.drop("is_fraud", axis=1)
y_combined = combined_data["is_fraud"]

# Encode categorical columns
label_encoder = LabelEncoder()
X_combined["merchant"] = label_encoder.fit_transform(X_combined["merchant"])
X_combined["category"] = label_encoder.fit_transform(X_combined["category"])

# One-hot encode categorical variables
categorical_columns = ["gender", "city", "state"]
onehot_encoder = OneHotEncoder(sparse=False, drop="first", handle_unknown='ignore')
X_combined_encoded = np.hstack((StandardScaler().fit_transform(X_combined.drop(categorical_columns, axis=1)), 
                               onehot_encoder.fit_transform(X_combined[categorical_columns])))

# Split the combined data back into training and test datasets
X_train = X_combined_encoded[:len(train_data)]
X_test = X_combined_encoded[len(train_data):]
y_train = y_combined[:len(train_data)]
y_test = y_combined[len(train_data):]

# Address class imbalance using SMOTE
X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_train, y_train)

# Apply Incremental PCA for dimensionality reduction
n_components = 100
ipca = IncrementalPCA(n_components=n_components)

# Apply Incremental PCA to training data
for batch in tqdm(np.array_split(X_resampled, 10), desc="Applying Incremental PCA"):
    ipca.partial_fit(batch)

# Transform the training and testing data
X_resampled_pca = ipca.transform(X_resampled)
X_test_pca = ipca.transform(X_test)

# Define and train the Random Forest model
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_resampled_pca, y_resampled)

# Predict using the trained model
y_pred = rf_classifier.predict(X_test_pca)

# Evaluate and display the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the ML Model Metrics
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{report}")
