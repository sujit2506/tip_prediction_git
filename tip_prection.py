import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

import warnings
warnings.filterwarnings('ignore')

df = sns.load_dataset('tips')
st.title("Tips Prediction Web Application")
st.write("This app predicts the tip amount based on user input.")
menu = st.sidebar.radio("Menu",["Home","Tip Prediction"])
if menu == "Home":
    if st.checkbox("Show Data"):
        st.write(df.head())
    st.title("Data visualization")

    st.header("Pair Plot")
    st.pyplot(sns.pairplot(data=df, hue='sex', palette='Set2'))

    st.header("Plot between Tips and Days")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='day',y='tip', ax=ax,palette='Set2')
    st.pyplot(fig)

    st.header("Plot between Days and Total Observations")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='day', ax=ax,palette='Set2',hue = 'sex')
    st.pyplot(fig)

    st.header("Plot between Tip and Time")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='time',y='tip', ax=ax,palette='Set2')
    st.pyplot(fig)

    st.header("Plot between Tip and Gender")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='sex',y='tip', ax=ax,palette='Set2',hue='smoker')
    st.pyplot(fig)



if menu == "Tip Prediction":
    oe = OrdinalEncoder()
    def encode_params(df):
        for col in df:
            if df[col].dtype not in ['int32', 'int64', 'float64']:
                df[col] = oe.fit_transform(df[[col]])

    encode_params(df)
    df.info() 

    X = df.drop(columns=['tip'])
    y = df['tip']

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=2)

    L_reg = LinearRegression()
    L_reg.fit(X_train,y_train)

    y_pred = L_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Enter the values:")
    total_bill = st.number_input("Total Bill", min_value=0.0)
    sex = st.selectbox("Sex", options=['Female', 'Male'])
    smoker = st.selectbox("Smoker", options=['Yes', 'No'])
    day = st.selectbox("Day", options=['Thur', 'Fri', 'Sat', 'Sun'])
    time = st.selectbox("Time", options=['Lunch', 'Dinner'])
    size = st.number_input("Party Size", min_value=1)


    input_data = pd.DataFrame([[total_bill, sex, smoker, day, time, size]], 
                          columns=['total_bill', 'sex', 'smoker', 'day', 'time', 'size'])
    encode_params(input_data)


    input_poly = poly.transform(input_data)


    if st.button("Predict Tip"):
        tip_prediction = L_reg.predict(input_poly)
        st.write(f"Predicted Tip: ${tip_prediction[0]:.2f}")
    st.write(f"Mean Squared Error on Test Data: {mse*100:.2f}%")
    st.write(f"R² Score on Test Data: {r2*100:.2f}%")
    
    