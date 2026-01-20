
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the pre-trained model and scaler
@st.cache_resource # Cache the model loading for efficiency
def load_model():
    model = joblib.load('tuned_random_forest_model.pkl')
    return model

@st.cache_resource # Cache the scaler loading for efficiency
def load_scaler():
    scaler = joblib.load('scaler.pkl')
    return scaler

rf_tuned_model = load_model()
scaler = load_scaler()

# Define the order of features as per X_train.columns
# This is crucial for correct prediction after scaling
feature_order = [
    'Country Code',
    'Longitude',
    'Latitude',
    'Average Cost for two',
    'Has Online delivery',
    'Is delivering now',
    'Price range',
    'Votes',
    'City_encoded',
    'Cuisines_encoded',
    'Currency_encoded',
    'Has Table booking_encoded',
    'City_grouped_encoded'
]

# Streamlit app layout
st.title('Restaurant Aggregate Rating Prediction')
st.write('Enter the restaurant features below to predict its aggregate rating.')

st.markdown('---')
st.subheader('Feature Inputs')

# Create input widgets for each feature

# Numerical inputs
country_code = st.number_input(
    'Country Code (mostly capped to 1.0 during preprocessing)',
    value=1.0, min_value=-1000.0, max_value=1000.0, step=0.1, format="%.1f"
)
longitude = st.number_input(
    'Longitude',
    value=0.0, min_value=-180.0, max_value=180.0, step=0.000001, format="%.6f"
)
latitude = st.number_input(
    'Latitude',
    value=0.0, min_value=-90.0, max_value=90.0, step=0.000001, format="%.6f"
)
avg_cost_for_two = st.number_input(
    'Average Cost for two',
    value=500.0, min_value=0.0, max_value=10000.0, step=10.0, format="%.1f"
)
votes = st.number_input(
    'Votes',
    value=100.0, min_value=0.0, max_value=10000.0, step=1.0, format="%.1f"
)

# Binary features with selectbox
has_online_delivery_option = st.selectbox('Has Online delivery', ('No', 'Yes'))
is_delivering_now_option = st.selectbox('Is delivering now', ('No', 'Yes'))
has_table_booking_option = st.selectbox('Has Table booking', ('No', 'Yes'))

has_online_delivery = 1 if has_online_delivery_option == 'Yes' else 0
is_delivering_now = 1 if is_delivering_now_option == 'Yes' else 0
has_table_booking_encoded = 1 if has_table_booking_option == 'Yes' else 0 # Assuming 'Yes' is 1 after encoding

# Price range slider
price_range = st.slider(
    'Price range (1: cheapest, 4: most expensive)',
    min_value=1, max_value=4, value=2, step=1
)

# Encoded categorical features (user needs to know the encoding)
st.write('---')
st.subheader('Encoded Categorical Features (input numerical IDs)')
st.info('Note: These features were label encoded. Please input the corresponding numerical ID.')
city_encoded = st.number_input(
    'City Encoded (e.g., 59 for New Delhi, 24 for Gurgaon, 91 for Noida)',
    value=59, min_value=0, max_value=150, step=1
)
cuisines_encoded = st.number_input(
    'Cuisines Encoded (e.g., 920 for North Indian, 255 for Chinese, 360 for Fast Food)',
    value=920, min_value=0, max_value=2000, step=1
)
currency_encoded = st.number_input(
    'Currency Encoded (e.g., 0 for Botswana Pula, 1 for Brazilian Real, 2 for Dollar)',
    value=0, min_value=0, max_value=10, step=1
)
city_grouped_encoded = st.number_input(
    'City Grouped Encoded (e.g., 5 for New Delhi, 2 for Gurgaon, 8 for Noida, 9 for Other)',
    value=5, min_value=0, max_value=15, step=1
)

# Collect all inputs into a dictionary
input_data = {
    'Country Code': country_code,
    'Longitude': longitude,
    'Latitude': latitude,
    'Average Cost for two': avg_cost_for_two,
    'Has Online delivery': has_online_delivery,
    'Is delivering now': is_delivering_now,
    'Price range': float(price_range),
    'Votes': votes,
    'City_encoded': float(city_encoded),
    'Cuisines_encoded': float(cuisines_encoded),
    'Currency_encoded': float(currency_encoded),
    'Has Table booking_encoded': float(has_table_booking_encoded),
    'City_grouped_encoded': float(city_grouped_encoded)
}

# Convert to DataFrame, ensuring correct order
input_df = pd.DataFrame([input_data], columns=feature_order)

# Prediction button
if st.button('Predict Aggregate Rating'):
    try:
        # Scale the input features
        scaled_input = scaler.transform(input_df)
        scaled_input_df = pd.DataFrame(scaled_input, columns=feature_order)

        # Make prediction
        prediction = rf_tuned_model.predict(scaled_input_df)[0]

        st.success(f'Predicted Aggregate Rating: {prediction:.2f}')
    except Exception as e:
        st.error(f'An error occurred during prediction: {e}')

st.markdown('---')
st.subheader('How to run this app locally:')
st.code('1. Save the model (tuned_random_forest_model.pkl) and scaler (scaler.pkl) files in the same directory as this app.py file.')
st.code('2. Open your terminal or command prompt.')
st.code('3. Navigate to the directory where you saved app.py.')
st.code('4. Run the command: streamlit run app.py')

st.subheader('Deployment to Streamlit Cloud Guidelines:')
st.write('1. **Repository**: Push your `app.py` file, `tuned_random_forest_model.pkl`, `scaler.pkl`, and a `requirements.txt` file (containing `streamlit`, `pandas`, `scikit-learn`, `joblib`) to a GitHub repository.')
st.write('2. **Streamlit Cloud**: Log in to Streamlit Cloud, connect your GitHub repository, and deploy the app.')
st.write('3. **Requirements.txt**: Make sure your `requirements.txt` includes all libraries used:')
st.code('streamlit\npandas\nscikit-learn\njoblib')
