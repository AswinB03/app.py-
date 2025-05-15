
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Air Quality Analysis", layout="wide")

st.title("Air Quality Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your air_quality_data.csv file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Drop rows without city
    df = df.dropna(subset=['City'])

    # Show missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Correlation heatmap for numeric data
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="hot", ax=ax)
    st.pyplot(fig)

    # AQI distribution
    if 'AQI' in df.columns:
        st.subheader(" AQI Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df['AQI'], kde=True, color='orange', ax=ax)
        ax.set_xlabel("AQI")
        st.pyplot(fig)

    # Model Training
    if 'AQI' in df.columns:
        X = df.drop('AQI', axis=1)
        y = df['AQI']
        X_encoded = pd.get_dummies(X)
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_preds = lr.predict(X_test)

        dt = DecisionTreeRegressor(random_state=42)
        dt.fit(X_train, y_train)
        dt_preds = dt.predict(X_test)

        st.subheader("Model Evaluation")

        st.write("**Linear Regression**")
        st.write(f"MAE: {mean_absolute_error(y_test, lr_preds):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_preds)):.2f}")
        st.write(f"R² Score: {r2_score(y_test, lr_preds):.2f}")

        st.write("**Decision Tree Regressor**")
        st.write(f"MAE: {mean_absolute_error(y_test, dt_preds):.2f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, dt_preds)):.2f}")
        st.write(f"R² Score: {r2_score(y_test, dt_preds):.2f}")

    # AQI by city bar chart
    st.subheader(" AQI by City")
    df_sorted = df.sort_values(by='AQI', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_sorted['City'], df_sorted['AQI'], color='skyblue')
    ax.set_xlabel("AQI Value")
    ax.set_ylabel("City")
    ax.set_title("Air Quality Index (AQI) by City")
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # Gas levels by city
    non_gas_columns = ['City', 'AQI', 'Date', 'Time']
    gas_columns = [col for col in df.columns if col not in non_gas_columns]

    for gas in gas_columns:
        df[gas] = pd.to_numeric(df[gas], errors='coerce')

    city_avg = df.groupby("City")[gas_columns].mean().reset_index()
    melted_df = city_avg.melt(id_vars='City', var_name='Gas', value_name='Level')

    st.subheader(" Pollution Gas Levels by City")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted_df, x='Level', y='City', hue='Gas', ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', label_type='edge', padding=3)
    ax.set_title("Average Pollution Gas Levels by City")
    ax.set_xlabel("Pollution Level")
    ax.set_ylabel("City")
    st.pyplot(fig)
      
