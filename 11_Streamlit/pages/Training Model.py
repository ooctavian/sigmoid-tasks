import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier


@st.cache_data
def get_dataframe():
    return pd.read_csv('11_Streamlit/diabetes.csv')

df = get_dataframe()

st.title("Training on Dataset 🚀")

left, middle, right = st.columns(3)

with left.popover("⚙️ Model Settings"):
    criterion = st.selectbox(label="Criterion", options=["gini", "entropy", "log_loss"], index=0)
    max_features = st.segmented_control(label="Max Features", options=["sqrt", "log2"])
    class_weight = st.segmented_control(label="Class Weight", options=["balanced", "balanced_subsample"])
    col1, col2 = st.columns(2)
    bootstrap = col1.toggle("Bootstrap", value=True)
    warm_start = col2.toggle("Warm Start", value=True)
    no_estimators = col1.number_input(label="Number of Estimators", min_value=0, max_value=100, step=5, value=40)
    random_state = col2.number_input(label="Random State", min_value=0, max_value=100, step=1, value=42)
    min_samples_split = col1.slider(label="Min Samples Split", min_value=0, max_value=100, step=5, value=40)
    min_samples_leaf = col2.slider(label="Min Samples Leaf", min_value=0, max_value=100, step=5, value=40)

if 'model' not in st.session_state:
    st.session_state['model'] = RandomForestClassifier(
        criterion=criterion,
        bootstrap=bootstrap,
        warm_start=warm_start,
        max_features=max_features,
        class_weight=class_weight,
        n_estimators=no_estimators,
        random_state=random_state,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf
    )
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

if right.button("🔬 Train Model", use_container_width=True):
    st.toast("Training started...", icon="🏋️")
    st.session_state['model'].fit(X, y)
    st.toast("Training finished...", icon="🎉")

tab1, tab2, tab3 = st.tabs(["Predicting diabetes", "Dataset", "Data Visualization"])

with tab1:
    with st.form(key='diabetes_form'):
        st.header("Enter Patient Information:")
        st.caption(
            "Please provide the patient's BMI, blood pressure, and other relevant details to help assess the risk of diabetes.")
        # Form fields for each feature in the dataset
        # Create columns
        col1, col2 = st.columns(2)

        # Group input fields in two columns
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
            bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=70.0, value=25.0)
            glucose = st.slider("Glucose Level", min_value=0, max_value=200, value=90)
            skin_thickness = st.slider("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

        with col2:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=30)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
            insulin = st.slider("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=50)

        if st.form_submit_button("Predict", icon="🔮"):
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'Glucose': [glucose],
                'BloodPressure': [blood_pressure],
                'SkinThickness': [skin_thickness],
                'Insulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigreeFunction': [diabetes_pedigree],
                'Age': [age]
            })
            if not hasattr(st.session_state['model'], 'estimators_'):
                st.session_state['model'].fit(X, y)

            predictions = st.session_state['model'].predict(input_data)
            has_diabetes = bool(predictions[0])
            if has_diabetes:
                st.subheader("Sadly, you have diabetes 😔💔")
                st.text("""
                _____________________________________¶¶___________
________________________________¶1¶1111111¶_______
________¶¶111¶_______________¶¶¶¶111111111¶¶¶1____
_____¶1¶¶¶¶¶111111¶_________¶¶¶1¶¶¶11111111¶1¶¶___
___¶¶¶1¶1111111111¶¶1______¶¶1¶¶¶1111111111111¶¶__
__¶¶1¶¶1111111111111¶¶_____¶¶¶1¶¶¶¶1111111111111¶_
__¶¶_¶1111111111111111¶¶___¶¶¶¶¶¶11¶111111111111¶_
_11_¶11111111111111111¶¶_____¶¶¶¶__¶111111111111¶¶
¶¶¶¶1111111111111111¶¶¶¶_____1¶¶__11111111111111¶¶
¶¶¶¶11111111111¶¶¶¶¶¶¶______1¶1¶¶1111111111111111¶
¶¶1¶1111111111111¶¶¶¶¶¶_____¶¶¶¶¶¶11111111111111¶¶
¶¶11111111111111111111111¶¶___¶¶¶¶¶¶1111111111¶¶¶_
_1¶111111111111111111¶¶¶¶¶¶____¶¶¶¶11111111111¶1__
__¶¶11111111111111111¶¶¶_____¶¶¶1111111111111¶1___
___¶¶¶111111111111¶1¶¶¶____1¶¶111¶1111111¶11¶1____
____1¶¶¶11111111111¶¶¶¶111¶¶¶¶111111111¶11¶¶¶_____
______¶¶¶¶1111111111111¶¶¶¶1¶¶¶¶¶¶¶¶11¶11¶¶_______
_______¶¶¶¶¶11111111111¶111¶___¶¶¶111¶1¶¶¶________
_________¶¶¶¶¶¶111111111111¶__¶¶¶111¶¶¶1__________
____________1¶¶¶¶¶11111111¶¶_¶¶¶¶111¶¶____________
______________¶¶¶¶¶¶¶1111111_¶¶¶11¶¶1_____________
_________________1¶¶¶¶¶¶1111¶¶¶1¶¶¶¶______________
____________________¶¶¶¶¶¶1¶¶¶¶¶1¶________________
_______________________¶1¶¶¶1¶¶¶__________________
___________________________11¶____________________
                """)
            else:
                st.subheader("Congratulations!!! You don't have diabetes. 🎉😊🙌")


with tab2:
    st.subheader("Dataset head")
    st.dataframe(df.head(n=10), use_container_width=True)
    st.divider()
    st.subheader("Dataset description")
    st.dataframe(df.describe(), use_container_width=True)

with tab3:
    st.subheader("Number of positive diabetes per age:")
    st.bar_chart(data=df, x='Age', y='Outcome')
    st.divider()

    st.subheader("BMI Distribution")
    st.bar_chart(df['BMI'].value_counts().sort_index())
    st.divider()

    pregnancy_data = df.groupby('Pregnancies')['Outcome'].value_counts().unstack().fillna(0)
    pregnancy_data.columns = ['Non-diabetic', 'Diabetic']
    pregnancy_data = pregnancy_data.sort_index()

    st.subheader("Pregnancies vs Outcome")
    st.line_chart(pregnancy_data)
    st.divider()

    st.subheader("BMI vs Glucose Levels")
    st.scatter_chart(data=df, y='BMI', x='Glucose')
    st.divider()

    corr_matrix = df.corr()

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, ax=ax)
    st.pyplot(fig)

container = st.sidebar.container()
with container:
    st.subheader("Model Parameters:")
    st.metric(label="🧰 Criterion", value=criterion, border=True)
    st.metric(label="⚖️ Class Weight", value=class_weight, border=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="🔑 Bootstrap", value=str(bootstrap), border=True)
        st.metric(label="🔥 Warm Start", value=str(warm_start), border=True)
        st.metric(label="🔧 Max Features", value=max_features, border=True)
        st.metric(label="📉 Min Samples Leaf", value=str(min_samples_leaf), border=True)

    with col2:
        st.metric(label="📊 Number of Estimators", value=str(no_estimators), border=True)
        st.metric(label="🔢 Random State", value=str(random_state), border=True)
        st.metric(label="🔀 Min Samples Split", value=str(min_samples_split), border=True)
