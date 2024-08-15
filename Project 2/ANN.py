import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import datetime
import os

# Function to build the model
def build_model(input_shape, num_hidden_layers, neurons, dropout_rate, optimizer, task, num_classes):
    model = Sequential()
    
    # Input layer
    model.add(Dense(units=neurons[0], activation='relu', input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Hidden layers
    for i in range(num_hidden_layers):
        model.add(Dense(units=neurons[i+1], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Output layer
    if task == 'regression':
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_squared_error'])
    elif task == 'binary_classification':
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    elif task == 'categorical_classification':
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Streamlit UI
st.title('Generalized ANN Model Trainer')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Overview:")
    st.write(data.head())

    # Select task type
    task = st.selectbox("Select Task", ("regression", "binary_classification", "categorical_classification"))
    target_column = st.selectbox("Select Target Column", data.columns)
    
    # Split data
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Preprocess target column
    if task == 'binary_classification':
        le = LabelEncoder()
        y = le.fit_transform(y)
        num_classes = 1
        y = y.reshape(-1, 1)  # Ensure target shape is (None, 1)
    elif task == 'categorical_classification':
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
        num_classes = y.shape[1]
    else:
        num_classes = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter selection
    num_hidden_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=10, value=1)
    
    neurons = [st.number_input(f"Neurons in Layer {i+1}", value=64) for i in range(num_hidden_layers + 2)]

    batch_size = st.selectbox("Batch Size", [1, 10, 50, 100, 500, 1000])
    epochs = st.selectbox("Number of Epochs", [1, 5, 10, 30, 50, 100])
    dropout_rate = st.selectbox("Dropout Rate", [0.1, 0.2, 0.3])
    optimizer = st.selectbox("Optimizer", ['SGD', 'RMSProp', 'Adam', 'Adagrad', 'Adadelta'])

    # Configure Early Stopping and TensorBoard
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    if st.button("Train Model"):
        hp = {
            'num_hidden_layers': num_hidden_layers,
            'neurons': neurons,
            'batch_size': batch_size,
            'epochs': epochs,
            'dropout_rate': dropout_rate,
            'optimizer': optimizer,
            'task': task,
            'num_classes': num_classes
        }
        
        model = build_model(X_train.shape[1], hp['num_hidden_layers'], hp['neurons'], hp['dropout_rate'], hp['optimizer'], hp['task'], hp['num_classes'])
        
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stopping, tensorboard_callback])

        # Save the model
        model_save_path = "saved_model/my_model.h5"
        model.save(model_save_path)
        st.write(f"Model saved at {model_save_path}")
        
        # Evaluate model
        if task == 'regression':
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse}")
            st.write(f"Mean Absolute Error: {mae}")
            st.write(f"R2 Score: {r2}")
        elif task == 'binary_classification':
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {acc}")
        elif task == 'categorical_classification':
            y_pred = np.argmax(model.predict(X_test), axis=-1)
            y_true = np.argmax(y_test, axis=-1)
            acc = accuracy_score(y_true, y_pred)
            st.write(f"Accuracy: {acc}")

        # Show TensorBoard link
        st.write(f"TensorBoard logs available at: {log_dir}")
        st.write(f"[Click here to view TensorBoard](http://localhost:6006)")

