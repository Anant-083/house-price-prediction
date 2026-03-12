import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 House Price Predictor")
st.markdown("### Predict California House Prices using Machine Learning")
st.markdown("---")

# Train model
@st.cache_resource
def train_model():
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

# Input section
st.markdown("## 📊 Enter House Details")

col1, col2 = st.columns(2)

with col1:
    med_inc = st.slider("Median Income (in $10,000)", 0.5, 15.0, 3.0)
    house_age = st.slider("House Age (years)", 1, 52, 20)
    ave_rooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
    ave_bedrms = st.slider("Average Bedrooms", 1.0, 5.0, 1.0)

with col2:
    population = st.slider("Population", 3.0, 35000.0, 1000.0)
    ave_occup = st.slider("Average Occupants", 1.0, 10.0, 3.0)
    latitude = st.slider("Latitude", 32.0, 42.0, 37.0)
    longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

st.markdown("---")

# Predict button
if st.button("🔮 Predict Price", use_container_width=True):
    features = np.array([[med_inc, house_age, ave_rooms, ave_bedrms,
                          population, ave_occup, latitude, longitude]])
    prediction = model.predict(features)[0]
    price_usd = prediction * 100000

    st.success(f"## 💰 Predicted House Price: ${price_usd:,.0f}")
    st.info(f"estimated value: ₹{price_usd * 84:,.0f} (approx)")

st.markdown("---")
st.markdown("Built with ❤️ using Python & Streamlit by Anant-083")
