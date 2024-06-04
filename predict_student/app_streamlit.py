import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Title of the app
st.title("Student Exam Performance Predicting")

# Description
st.write("""
This app predicts student performance based on various factors such as gender, ethnicity, parental level of education, lunch type, test preparation course, and exam scores.
""")

# Form to input student data
with st.form(key='predict_form'):
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Race or Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", ["Associate's Degree", "Bachelor's Degree", "High School", "Master's Degree", "Some College", "Some High School"])
    lunch = st.selectbox("Lunch Type", ["Free/Reduced", "Standard"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["None", "Completed"])
    reading_score = st.number_input("Reading Score out of 100", min_value=0, max_value=100, step=1)
    writing_score = st.number_input("Writing Score out of 100", min_value=0, max_value=100, step=1)

    # Submit button
    submit_button = st.form_submit_button(label='Predict your Maths Score')

# When the form is submitted
if submit_button:
    # Create a data instance
    data = CustomData(
        gender=gender.lower(),
        race_ethnicity=ethnicity.lower(),
        parental_level_of_education=parental_level_of_education.lower(),
        lunch=lunch.lower(),
        test_preparation_course=test_preparation_course.lower(),
        reading_score=reading_score,
        writing_score=writing_score
    )

    # Convert data to dataframe
    pred_df = data.get_data_as_data_frame()

    # Predict
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)

    # Display prediction
    st.write(f"The predicted Maths Score is: {results[0]}")

# CSS for styling
st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f0f0;
            color: #333;
        }

        .container {
            width: 80%;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            margin-top: 50px;
        }

        h1, h2 {
            color: #444;
        }

        label {
            display: inline-block;
            width: 200px;
        }

        input[type="text"], input[type="number"], select {
            width: calc(100% - 220px);
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        input[type="submit"] {
            padding: 10px 20px;
            background-color: #5cb85c;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }

        input[type="submit"]:hover {
            background-color: #4cae4c;
        }

        .button-container {
            text-align: center;
            margin-top: 20px;
        }

        .btn {
            text-decoration: none;
            color: white;
            background-color: #5cb85c;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }

        .btn:hover {
            background-color: #4cae4c;
        }
    </style>
""", unsafe_allow_html=True)
