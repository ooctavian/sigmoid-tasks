import streamlit as st

st.title("Pima Indians Diabetes Database")

# Expander for Dataset Overview
with st.expander("Dataset Overview", expanded=True):
    st.text("""
    This dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. The goal is to predict whether or not a patient has diabetes, using diagnostic measurements available in the dataset. The data focuses on females who are at least 21 years old and of Pima Indian heritage.

    Several constraints were placed on the selection of these instances from a larger database, ensuring that the dataset only includes individuals who meet the aforementioned criteria.
    """)

# Expander for Dataset Features
with st.expander("Dataset Features", expanded=True):
    st.markdown("""
    - **Pregnancies**: Number of pregnancies the patient has had
    - **Glucose**: Plasma glucose concentration (2-hour oral glucose tolerance test)
    - **BloodPressure**: Diastolic blood pressure (mm Hg)
    - **SkinThickness**: Triceps skin fold thickness (mm)
    - **Insulin**: 2-Hour serum insulin""")
    st.latex(r"""
    \mu \, \text{U/ml} 
    """)
    st.markdown("""
    - **BMI**: Body mass index
    """)
    st.latex(r'''
    \\text{BMI} = \frac{\text{weight (kg)}}{\left(\text{height (m)}\right)^2}
    ''')
    st.markdown("""
    - **DiabetesPedigreeFunction**: Diabetes pedigree function
    - **Age**: Age of the patient (in years)
    """)

# Expander for Target Column
with st.expander("Target Column", expanded=True):
    st.markdown("""
    The **Outcome** column is the target variable to predict, with the following interpretation:
    - **0**: No diabetes
    - **1**: Diabetes
    """)
