import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pandas as pd

# Load the trained model
model_filename = 'tuned_random_forest_regressor.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)

# Load dataset to refit scaler
# (If scaler was saved from the training, you can load it instead)
file_path = 'insurance.csv'  # Make sure this file is accessible
data = pd.read_csv(file_path)

label_encoders = {}
categorical_columns = ['sex', 'smoker', 'region']

# Apply label encoding to each categorical column
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Step 2: Set 'charges' as target and normalize the remaining features
X = data.drop('charges', axis=1)
y = data['charges']

# Features for scaling (excluding the target 'charges')
X = data.drop('charges', axis=1)

# Recreate and fit the scaler on the entire training data
scaler = StandardScaler()
scaler.fit(X)

# Streamlit app
st.title('Insurance Charges Prediction')

# Input fields for user to enter data
age = st.number_input('Age', min_value=18, max_value=100, value=25)
sex = st.selectbox('Sex', ('male', 'female'))
bmi = st.number_input('BMI (Body Mass Index)', min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
smoker = st.selectbox('Smoker', ('yes', 'no'))
region = st.selectbox('Region', ('southeast', 'southwest', 'northwest', 'northeast'))

# Preprocess input to match the model requirements
# Note: Make sure to use the same label encoding as during training
sex_encoded = 1 if sex == 'male' else 0
smoker_encoded = 1 if smoker == 'yes' else 0
region_mapping = {'southeast': 0, 'southwest': 1, 'northwest': 2, 'northeast': 3}
region_encoded = region_mapping[region]

# Prepare input data for prediction
input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

# Apply the saved scaler (do not fit again, just transform)
input_data_scaled = scaler.transform(input_data)

# Prediction
if st.button('Predict Insurance Charges'):
    prediction = model.predict(input_data_scaled)
    st.write(f'Estimated Insurance Charges: ${prediction[0]:.2f}')
