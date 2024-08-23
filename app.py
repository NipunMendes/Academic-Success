import streamlit as st
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('submission.pkl', 'rb'))

# Function to display input fields with options in Streamlit
def input_field(label, options=None):
    if options:
        return st.selectbox(label, list(options.keys()), format_func=lambda x: options[x])
    else:
        return st.slider(label, min_value=1, max_value=100)

st.title("Student Academic Status Prediction")

# Collecting user input using sliders and select boxes
marital_status = input_field("Marital Status", {1: 'Married', 2: 'Single', 3: 'Divorced'})
application_mode = input_field("Application Mode", {1: 'Mode A', 2: 'Mode B', 3: 'Mode C'})
application_order = st.slider("Application Order", 1, 5, 1)
course = st.slider("Course Code", 1000, 9999, 9001)
daytime_evening_attendance = input_field("Time of Study", {1: 'Daytime', 2: 'Evening'})
previous_qualification = input_field("Previous Qualification", {1: 'Qualification A', 2: 'Qualification B'})
previous_qualification_grade = st.slider("Previous Qualification Grade", 0.0, 200.0, 150.0, step=0.1)
mothers_qualification = input_field("Mother's Qualification", {1: 'Qualification A', 2: 'Qualification B'})
fathers_qualification = input_field("Father's Qualification", {1: 'Qualification A', 2: 'Qualification B'})
fathers_occupation = input_field("Father's Occupation", {1: 'Occupation A', 2: 'Occupation B'})
admission_grade = st.slider("Admission Grade", 0.0, 20.0, 15.0, step=0.1)
displaced = input_field("Are you displaced?", {0: 'No', 1: 'Yes'})
educational_special_needs = input_field("Do you have special educational needs?", {0: 'No', 1: 'Yes'})
debtor = input_field("Are you a debtor?", {0: 'No', 1: 'Yes'})
tuition_fees_up_to_date = input_field("Tuition Fees Status", {1: 'Up to date', 2: 'Not up to date'})
gender = input_field("Gender", {1: 'Male', 2: 'Female'})
scholarship_holder = input_field("Are you a scholarship holder?", {0: 'No', 1: 'Yes'})
age_at_enrollment = st.slider("Age at Enrollment", 15, 100, 18)
international = input_field("Are you an international student?", {0: 'No', 1: 'Yes'})
curricular_units_1st_sem_evaluations = st.slider("1st Semester Evaluations", 0, 10, 5)
curricular_units_1st_sem_without_evaluations = st.slider("1st Semester Subjects Without Evaluations", 0, 10, 1)
curricular_units_2nd_sem_credited = st.slider("2nd Semester Credits", 0, 10, 6)
curricular_units_2nd_sem_enrolled = st.slider("2nd Semester Enrolled Subjects", 0, 10, 6)
curricular_units_2nd_sem_evaluations = st.slider("2nd Semester Evaluations", 0, 10, 5)
curricular_units_2nd_sem_grade = st.slider("2nd Semester Grade", 0.0, 20.0, 14.0, step=0.1)
curricular_units_2nd_sem_without_evaluations = st.slider("2nd Semester Subjects Without Evaluations", 0, 10, 1)
unemployment_rate = st.slider("Unemployment Rate", 0.0, 100.0, 7.5, step=0.1)
inflation_rate = st.slider("Inflation Rate", 0.0, 100.0, 2.5, step=0.1)
gdp = st.slider("GDP", 0.0, 20.0, 3.0, step=0.1)

# Collect the input features into an array
features = np.array([
    marital_status, application_mode, application_order, course,
    daytime_evening_attendance, previous_qualification, previous_qualification_grade,
    mothers_qualification, fathers_qualification, fathers_occupation, admission_grade,
    displaced, educational_special_needs, debtor, tuition_fees_up_to_date, gender,
    scholarship_holder, age_at_enrollment, international,
    curricular_units_1st_sem_evaluations, curricular_units_1st_sem_without_evaluations,
    curricular_units_2nd_sem_credited, curricular_units_2nd_sem_enrolled,
    curricular_units_2nd_sem_evaluations, curricular_units_2nd_sem_grade,
    curricular_units_2nd_sem_without_evaluations, unemployment_rate, inflation_rate, gdp
]).reshape(1, -1)

# Predict when the user clicks the button
if st.button("Predict Academic Status"):
    prediction = model.predict(features)

    # Mapping the prediction to a label
    labels = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
    result = labels.get(prediction[0], "Unknown")

    # Display the result
    st.subheader(f"The predicted academic status is: {result}")
