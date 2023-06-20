import streamlit as st
import pickle
import numpy as np
from linear_regression_model import LinearRegression

# Load the model
with open('model.pkl', 'rb') as file:
    model_data = pickle.load(file)
    model = model_data['model']
    X_train_expanded = model_data['X_train_expanded']
    y_train_mean = model_data['y_train_mean']
    y_train_std = model_data['y_train_std']
    best_X_train = model_data['best_X_train']
    best_y_train = model_data['best_y_train']

# Function to normalize input data
def normalize_input_data(data):
    normalized_data = (data - np.mean(best_X_train, axis=0)) / np.std(best_X_train, axis=0)
    return normalized_data

# Function to expand input features
def expand_input_features(data):
    normalized_data = normalize_input_data(data)
    expanded_data = model.expand_features(normalized_data, degree=2)
    return expanded_data

# Function to denormalize predicted data
def denormalize_data(data):
    denormalized_data = (data * y_train_std) + y_train_mean
    return denormalized_data

# Streamlit app code
def main():
    st.title('Prediksi Harga Rumah')

    # Input form
    input_data_1 = st.text_input('Luas Tanah', '1.0')
    input_data_2 = st.text_input('Luas Bangunan', '2.0')

    # Check if input values are numeric
    if not input_data_1.isnumeric() or not input_data_2.isnumeric():
        st.error('Please enter numeric values for the input features.')
        return

    # Convert input values to float
    input_feature_1 = float(input_data_1)
    input_feature_2 = float(input_data_2)

    # Normalize and expand input features
    input_features = np.array([[input_feature_1, input_feature_2]])
    expanded_input = expand_input_features(input_features)

    # Perform prediction
    normalized_prediction = model.predict(expanded_input)
    prediction = denormalize_data(normalized_prediction)

    # Display the prediction
    st.subheader('Prediction')
    st.write(prediction[0])

if __name__ == '__main__':
    main()
