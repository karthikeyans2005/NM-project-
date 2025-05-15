# NM-project-
Navigation Menu
Import required libraries
import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler, LabelEncoder from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor from sklearn.svm import SVR from sklearn.neural_network import MLPRegressor from sklearn.metrics import mean_squared_error, r2_score from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense, Dropout from tensorflow.keras.callbacks import EarlyStopping import joblib import warnings warnings.filterwarnings('ignore')

Set random seed for reproducibility
np.random.seed(42)

=============================================
Data Loading and Preprocessing
=============================================
def load_data(filepath): """Load and preprocess air quality data""" df = pd.read_csv(filepath)

# Handle missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Convert timestamp to datetime
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month

return df
def feature_engineering(df, target_column='PM2.5'): """Create additional features and prepare data for modeling"""

# Create lag features for time series data
for lag in [1, 2, 3, 24]:
    df[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)

# Create rolling statistics
df[f'{target_column}_rolling_mean_24'] = df[target_column].rolling(window=24).mean()
df[f'{target_column}_rolling_std_24'] = df[target_column].rolling(window=24).std()

# Drop remaining NA values created by lag features
df.dropna(inplace=True)

return df
def prepare_data(df, target_column='PM2.5'): """Split data into features and target, and scale the data"""

X = df.drop(columns=[target_column, 'timestamp'], errors='ignore')
y = df[target_column]

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

return X_train_scaled, X_test_scaled, y_train, y_test, scaler
=============================================
Model Training
=============================================
def train_ml_models(X_train, y_train): """Train multiple machine learning models"""

models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100,50), 
                        max_iter=1000, random_state=42)
}

trained_models = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model

return trained_models
def build_lstm_model(X_train, y_train, epochs=50, batch_size=32): """Build and train an LSTM model for time series prediction"""

# Reshape data for LSTM [samples, timesteps, features]
X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

# Define model
model = Sequential()
model.add(LSTM(100, activation='relu', input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(
    X_train_reshaped, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

return model, history
=============================================
Model Evaluation
=============================================
def evaluate_models(models, X_test, y_test): """Evaluate trained models on test data"""

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}
    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")

return results
def evaluate_lstm(model, X_test, y_test, scaler): """Evaluate LSTM model"""

X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
y_pred = model.predict(X_test_reshaped).flatten()

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"LSTM - MSE: {mse:.4f}, R2: {r2:.4f}")
return {'MSE': mse, 'R2': r2}
=============================================
Main Execution
=============================================
if name == "main": # Load and preprocess data print("Loading and preprocessing data...") data = load_data('data/raw_data.csv') # Replace with your data path data = feature_engineering(data) X_train, X_test, y_train, y_test, scaler = prepare_data(data)

# Train traditional ML models
print("\nTraining machine learning models...")
ml_models = train_ml_models(X_train, y_train)

# Evaluate ML models
print("\nEvaluating machine learning models...")
ml_results = evaluate_models(ml_models, X_test, y_test)

# Train LSTM model
print("\nTraining LSTM model...")
lstm_model, history = build_lstm_model(X_train, y_train)

# Evaluate LSTM
print("\nEvaluating LSTM model...")
lstm_results = evaluate_lstm(lstm_model, X_test, y_test, scaler)

# Save the best model
print("\nSaving the best model...")
joblib.dump(ml_models['Random Forest'], 'models/best_model.pkl')
print("Model saved successfully!")

# Plot training history for LSTM
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training History')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('lstm_training_history.png')
plt.show()
