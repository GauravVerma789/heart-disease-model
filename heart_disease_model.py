import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/datasets/888463/1508992/heart_disease_uci.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250406%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250406T064250Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9c8dd655b8e0d1bb98a744f565e3cb7b76235a086c76b6a7f0e7484af7d326febfb51d7cfa929fe32ec1a37f371eec5a90fdf8bc0ea126adf99375d7a6fe8faad1274d78408c18d91d80da34d038b71b3e3b6531d5d7e2580999b125799c0b10931416d56d0d56cff36dfee0f52adb55bfa4300e1849405d7535f4186a5150a4f6a18a1b80da25a0ae077f036505b6711d5d17ef2d32008dbf9fa9be04b0b20d94684aa9dee7cf87909aeea78978743e67464c2ad07412967f5afbf6353456b93072ff33d1a99914d885155c1d66e92276e1b3a181682cd3d6d2abecbb256093a9ec0c4dec6a0e2f942917d0dcc8f945e9ba25b5a573221fb832bc7393dda736')

# Drop irrelevant columns
df.drop(['id', 'dataset'], axis=1, inplace=True)

# Convert target column to binary
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
df.rename(columns={'num': 'target'}, inplace=True)

# Handle missing values (if any)
df.dropna(inplace=True)

# List of categorical columns to encode
categorical_cols = ['sex', 'cp', 'restecg', 'exang', 'slope', 'thal']

# Encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))  # convert to str just in case

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler


# Save the trained model
with open('heart_disease_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)