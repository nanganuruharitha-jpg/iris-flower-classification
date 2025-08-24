import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------- Load Dataset -----------------
@st.cache_data
def load_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y, iris.target_names

X, y, target_names = load_data()

# ----------------- Train Model -----------------
@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# ----------------- Streamlit UI -----------------
st.title("🌸 Iris Flower Classification")
st.write("Enter flower measurements and predict the Iris species.")

# Sidebar input sliders
sepal_length = st.slider("Sepal length (cm)", float(X["sepal length (cm)"].min()), float(X["sepal length (cm)"].max()))
sepal_width = st.slider("Sepal width (cm)", float(X["sepal width (cm)"].min()), float(X["sepal width (cm)"].max()))
petal_length = st.slider("Petal length (cm)", float(X["petal length (cm)"].min()), float(X["petal length (cm)"].max()))
petal_width = st.slider("Petal width (cm)", float(X["petal width (cm)"].min()), float(X["petal width (cm)"].max()))

# Make prediction
if st.button("🔍 Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)
    st.success(f"Predicted Species: {target_names[prediction[0]]}")
