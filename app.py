
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load dataset
DATA_PATH = Path(__file__).parent / "data" / "student_data.csv"
data = pd.read_csv(DATA_PATH)

# Prepare data
X = data[["study_hours"]]
y = data["marks"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Website heading
st.title("🎓 Student Marks Predictor")
st.write(
    "Predict student marks using Machine Learning "
    "based on study hours."
)

st.divider()

# User input
st.subheader("Enter Your Study Hours")

hours = st.slider(
    "Hours studied per day",
    min_value=1,
    max_value=10,
    value=7
)

# Prediction button
if st.button("Predict Marks", type="primary"):

    prediction = model.predict(
        pd.DataFrame({"study_hours": [hours]})
    )[0]

    st.success("Prediction completed!")

    st.metric(
        label="Predicted Student Marks",
        value=f"{prediction:.2f} / 100"
    )

    # Create graph
    fig, ax = plt.subplots()

    ax.scatter(
        data["study_hours"],
        data["marks"],
        label="Actual Marks"
    )

    ax.plot(
        data["study_hours"],
        model.predict(X),
        color="red",
        label="Regression Line"
    )

    ax.scatter(
        [hours],
        [prediction],
        color="green",
        s=100,
        label="Your Prediction"
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Marks")
    ax.set_title("Student Marks Prediction")
    ax.legend()

    st.pyplot(fig)

st.divider()

st.subheader("Dataset Preview")
st.dataframe(data, use_container_width=True)

st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")
