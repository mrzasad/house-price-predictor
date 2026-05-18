
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Predictor", layout="wide")

st.title("🏡 House Price Predictor")
st.markdown(
    '''
    Predict house prices using Multiple Linear Regression (MLR) based on:
    - Bedrooms
    - Bathrooms
    - Square footage
    - Floors
    - Location
    - Year built
    and more.
    '''
)

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Remove unnecessary columns
drop_cols = ["date", "street", "country"]
X = df.drop(columns=["price"] + drop_cols)
y = df["price"]

# Detect categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)

col1.metric("Mean Absolute Error", f"${mae:,.2f}")
col2.metric("R² Score", f"{r2:.2f}")

st.markdown("---")
st.subheader("🔮 Predict House Price")

# User input form
with st.form("prediction_form"):
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0)
    sqft_living = st.number_input("Living Area (sqft)", min_value=300, max_value=15000, value=1800)
    sqft_lot = st.number_input("Lot Area (sqft)", min_value=500, max_value=1000000, value=5000)
    floors = st.number_input("Floors", min_value=1.0, max_value=5.0, value=1.0)
    waterfront = st.selectbox("Waterfront", [0, 1])
    view = st.slider("View Rating", 0, 4, 0)
    condition = st.slider("Condition", 1, 5, 3)
    sqft_above = st.number_input("Above Ground Sqft", min_value=300, max_value=10000, value=1500)
    sqft_basement = st.number_input("Basement Sqft", min_value=0, max_value=5000, value=0)
    yr_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2000)
    yr_renovated = st.number_input("Year Renovated", min_value=0, max_value=2025, value=0)

    city = st.selectbox("City", sorted(df["city"].unique()))
    statezip = st.selectbox("State Zip", sorted(df["statezip"].unique()))

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = pd.DataFrame({
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "sqft_living": [sqft_living],
        "sqft_lot": [sqft_lot],
        "floors": [floors],
        "waterfront": [waterfront],
        "view": [view],
        "condition": [condition],
        "sqft_above": [sqft_above],
        "sqft_basement": [sqft_basement],
        "yr_built": [yr_built],
        "yr_renovated": [yr_renovated],
        "city": [city],
        "statezip": [statezip]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated House Price: ${prediction:,.2f}")

st.markdown("---")
st.markdown(
    '''
    ### 🚀 How This Project Helps Users
    - Enables buyers to estimate fair property prices
    - Assists sellers in setting realistic listing prices
    - Helps agents provide data-driven recommendations
    - Saves time with instant predictions
    - Improves transparency using machine learning
    '''
)
