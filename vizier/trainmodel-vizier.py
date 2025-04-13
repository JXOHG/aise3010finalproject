import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score
import tensorflow as tf
import argparse
from google.cloud import storage

# Import the correct hypertune library
try:
    import hypertune
except ImportError:
    print("Failed to import hypertune module")
    hypertune = None

# Parse command-line arguments for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_neurons', type=int, default=64)
parser.add_argument('--dropout_rate', type=float, default=0.3)
args = parser.parse_args()

# Download CSV files from GCS
def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}")

bucket_name = "cloud-ai-platform-8c821022-d31d-4832-8a69-5d84538874ed"
try:
    download_blob(bucket_name, "data/train_data.csv", "/tmp/train_data.csv")
    download_blob(bucket_name, "data/test_data.csv", "/tmp/test_data.csv")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit(1)

# Load data
try:
    train_data = pd.read_csv("/tmp/train_data.csv")
    test_data = pd.read_csv("/tmp/test_data.csv")
    print(f"Train data rows: {len(train_data)}")
    print(f"Test data rows: {len(test_data)}")
    if train_data.empty or test_data.empty:
        print("Error: Empty train or test data")
        exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Handle missing values
train_data = train_data.dropna()
test_data = test_data.dropna()

# Separate features and target
try:
    X_train = train_data.drop(columns=['Type'])
    y_train = train_data['Type']
    X_test = test_data.drop(columns=['Type'])
    y_test = test_data['Type']
except KeyError as e:
    print(f"Column error: {e}")
    exit(1)

# Encode target variable
try:
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_train_categorical = to_categorical(y_train_encoded)
    y_test_encoded = label_encoder.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded)
except Exception as e:
    print(f"Encoding error: {e}")
    exit(1)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build model
model = Sequential([
    Dense(args.num_neurons, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(args.dropout_rate),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
try:
    history = model.fit(
        X_train_scaled, y_train_categorical,
        epochs=50,
        batch_size=32,
        validation_split=0.15,
        callbacks=[early_stopping],
        verbose=1
    )
except Exception as e:
    print(f"Training error: {e}")
    exit(1)

# Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test_categorical, verbose=0)
y_pred = model.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_categorical, axis=1)
precision = precision_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
recall = recall_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)

# Print metrics for logging
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

# Report metrics to Vizier
if hypertune:
    try:
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='accuracy',
            metric_value=accuracy
        )
        print("Successfully reported accuracy to Vizier")
    except Exception as e:
        print(f"Error reporting metric to Vizier: {e}")
else:
    print("Hypertune not available, metrics won't be reported to Vizier")