import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

from my_module.model import predict


# لازم تكون أول دالة Streamlit في الملف
st.set_page_config(
    page_title="Iris Project",
    page_icon="🦈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# عنوان وشرح المشروع
st.header(":blue[Welcome to my Iris application]")
st.title("Iris Project")

st.write(
    "This is a simple web application built using Streamlit "
    "to demonstrate Iris dataset analysis, visualization, and prediction."
)

st.write(
    "The Iris dataset contains measurements of iris flowers "
    "from three different species."
)

st.write("**Sepal length:** The length of the sepal in centimeters.")
st.write("**Sepal width:** The width of the sepal in centimeters.")
st.write("**Petal length:** The length of the petal in centimeters.")
st.write("**Petal width:** The width of the petal in centimeters.")

st.caption("Use the buttons and sliders below to explore the Iris dataset.")


# تحميل البيانات
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# إضافة اسم نوع الزهرة للجدول
df["species"] = [
    iris.target_names[target]
    for target in iris.target
]


# عرض البيانات
st.header("Iris Dataset")

if st.button("Show Iris Dataset"):
    st.write("Here is the Iris dataset:")
    st.dataframe(df, use_container_width=True)


if st.button("Show Iris Dataset Summary"):
    st.write("Here is the summary of the Iris dataset:")
    st.dataframe(df.describe(), use_container_width=True)


# الرسومات
st.header("Iris Dataset Visualization")

tab1, tab2, tab3 = st.tabs(
    ["Line Chart", "Scatter Chart", "Bar Chart"]
)

with tab1:
    st.line_chart(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        use_container_width=True
    )

with tab2:
    st.scatter_chart(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        use_container_width=True
    )

with tab3:
    st.bar_chart(
        df,
        x="sepal length (cm)",
        y="sepal width (cm)",
        use_container_width=True
    )


# التوقع
st.header("Iris Species Prediction")

sepal_length = st.slider(
    "SEPAL LENGTH",
    min_value=4.0,
    max_value=8.0,
    value=5.1,
    step=0.1
)

sepal_width = st.slider(
    "SEPAL WIDTH",
    min_value=2.0,
    max_value=4.5,
    value=3.5,
    step=0.1
)

petal_length = st.slider(
    "PETAL LENGTH",
    min_value=1.0,
    max_value=7.0,
    value=1.4,
    step=0.1
)

petal_width = st.slider(
    "PETAL WIDTH",
    min_value=0.1,
    max_value=2.5,
    value=0.2,
    step=0.1
)


if st.button("Predict", type="primary"):

    result = predict(
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    )

    st.success(f"The predicted Iris species is: {result}")