import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def check_gpu_availability():
  """Prints whether TensorFlow is using a GPU for calculations."""
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.list_logical_devices('GPU')
      print(f"{len(gpus)} Physical GPUs available, {len(logical_gpus)} Logical GPUs available.")
      print("TensorFlow is using the GPU for calculations.")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)
      print("TensorFlow might not be using the GPU due to initialization issues.")
  else:
    print("No GPUs found. TensorFlow is using the CPU for calculations.")

check_gpu_availability()

# Connect to the database using the SQL Proxy
engine = create_engine("mysql+pymysql://root:0000@127.0.0.1:3307/aise3010finalproject-db")

# Load the pre-split training and testing data
train_data = pd.read_sql("SELECT * FROM `train_data`", engine)
test_data = pd.read_sql("SELECT * FROM `test_data`", engine)

# Close the connection
engine.dispose()

# Handle any missing values
print("Missing values in train_data:", train_data.isnull().sum())
print("Missing values in test_data:", test_data.isnull().sum())
train_data = train_data.dropna()
test_data = test_data.dropna()

# Separate features and target for both datasets
X_train = train_data.drop(columns=['Type'])  # Adjust column names as needed
y_train = train_data['Type']
X_test = test_data.drop(columns=['Type'])
y_test = test_data['Type']

# Encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_categorical = to_categorical(y_train_encoded)
# Use the same encoder for test data
y_test_encoded = label_encoder.transform(y_test)
y_test_categorical = to_categorical(y_test_encoded)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Use the same scaler for test data
X_test_scaled = scaler.transform(X_test)

# Build the Neural Network model with dropout for regularization
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.3),  # Add dropout to prevent overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Measure training efficiency
start_time = time.time()

# Train the model with validation data from the training set
history = model.fit(
    X_train_scaled, y_train_categorical,
    epochs=50,  # Maximum epochs
    batch_size=32,
    validation_split=0.15,  # Use a portion of training data for validation
    callbacks=[early_stopping]
)

# Save the trained model in TensorFlow SavedModel format
tf.saved_model.save(model, 'trained_model')

training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")


training_time = time.time() - start_time
print(f"Training Time: {training_time:.2f} seconds")

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test_categorical, verbose=0)
print(f"Model Accuracy: {accuracy:.4f}")

# Make predictions for metrics calculation
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_categorical, axis=1)

# Calculate precision and recall
precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_test_classes, 
    y_pred_classes, 
    target_names=label_encoder.classes_
))

# Plot accuracy graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.savefig('model_accuracy.png')
plt.show()

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix, 
    annot=True, 
    fmt='d', 
    cmap='Blues', 
    xticklabels=label_encoder.classes_, 
    yticklabels=label_encoder.classes_
)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()