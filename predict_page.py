import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_EmployedTrans = data["le_EmployedTrans"]
le_education = data["le_education"]




def show_predict_page():
    st.title("Softwar Developer Salary Prediction")
    st.write(""" ### we need some information to predict the SALARY !!! """)
    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    employed = (
        "Employed full-time",
        "Independent contractor, freelancer, or self-employed",
        "Employed part-time",
    )




    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    employed = st.selectbox("employed Level", employed)
    expericence = st.slider("Years of Experience", 0, 50, 3)
    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence, employed ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X[:, 3] = le_EmployedTrans.transform(X[:,3])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
