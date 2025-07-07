import streamlit as st
import pickle
import numpy as np
import os 


st.title('Bank Marketing Prediction App')


# load the job encoder
encoder_path = os.path.join(os.path.dirname(__file__), 'job_encoder.pickle') 
if os.path.exists(encoder_path):
    with open(encoder_path, 'rb') as f:
        job_encoder = pickle.load(f)
else:
    st.error(f"Error: 'job_encoder.pickle' not found at {encoder_path}. Please ensure it's in the same directory as main.py.")
    st.stop()


# feature inputs
age = st.slider('Age:', min_value=18, max_value=95, value=50, step=1)

job_options = [
    'blue-collar',
    'management',
    'technician',
    'admin.',
    'services',
    'retired',
    'entrepreneur',
    'self-employed',
    'student',
    'unemployed',
    'housemaid',
    'unknown',
]
job = st.selectbox('Job:', job_options, index=job_options.index('blue-collar'))

marital_options = [
    ('married', 2),
    ('single', 1),
    ('divorced', 0),
]
marital = st.selectbox('Marital:', marital_options, format_func=lambda x: x[0], index=marital_options.index(('married', 2)))

education_options = [
    ('tertiary', 3),
    ('secondary', 2),
    ('elementary', 1),
    ('unknown', 0),
]
education = st.selectbox('Education:', education_options, format_func=lambda x: x[0], index=education_options.index(('tertiary', 3))) # Corrected spelling

boolean_options = [
    ('yes', 1),
    ('no', 0),
]
default = st.selectbox('Default:', boolean_options, format_func=lambda x: x[0], index=boolean_options.index(('no', 0)))

balance = st.slider('Balance:', min_value=-10000, max_value=120000, value=0, step=100)

housing = st.selectbox('Housing:', boolean_options, format_func=lambda x: x[0], index=boolean_options.index(('no', 0)))

loan = st.selectbox('Loan:', boolean_options, format_func=lambda x: x[0], index=boolean_options.index(('no', 0)))

contact_options = [
    ('cellular', 2),
    ('telephone', 1),
    ('unknown', 0),
]
contact = st.selectbox('Contact:', contact_options, format_func=lambda x: x[0], index=contact_options.index(('cellular', 2)))

day = st.slider('Day of Month:', min_value=1, max_value=31, value=1, step=1)

month_options = [
    ('jan', 1),
    ('feb', 2),
    ('mar', 3),
    ('apr', 4),
    ('may', 5),
    ('jun', 6),
    ('jul', 7),
    ('aug', 8),
    ('sep', 9),
    ('oct', 10),
    ('nov', 11),
    ('dec', 12),
]
month = st.selectbox('Month:', month_options, format_func=lambda x: x[0], index=month_options.index(('jan', 1)))

duration = st.slider('Duration (seconds):', min_value=0, max_value=5000, value=200, step=1)

campaign = st.slider('Campaign:', min_value=1, max_value=65, value=1, step=1)

pdays = st.slider('Pdays:', min_value=-1, max_value=1000, value=-1, step=1)  # -1 means client was not previously contacted

previous = st.slider('Previous:', min_value=0, max_value=300, value=0, step=1)

poutcome_options = [
    ('success', 3),
    ('other', 2),
    ('failure', 1),
    ('unknown', 0),
]
poutcome = st.selectbox('Poutcome:', poutcome_options, format_func=lambda x: x[0], index=poutcome_options.index(('unknown', 0)))


# Load all models
models = {}
models_dir = os.path.join(os.path.dirname(__file__), 'models')
if os.path.exists(models_dir):
    for filename in os.listdir(models_dir):
        if filename.endswith('.pickle'):
            model_name = os.path.splitext(filename)[0].replace('_', ' ').title() # Format model name nicely
            with open(os.path.join(models_dir, filename), 'rb') as f:
                models[model_name] = pickle.load(f)
else:
    st.error("The 'models' directory was not found. Please ensure your models are in a folder named 'models' in the same directory as main.py.")


# Load StandardScaler
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pickle')
if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
else:
    scaler = None
    st.warning("'scaler.pickle' not found. Input data will not be scaled.")


# Display model selection if models are available
if models:

    model_selection = st.selectbox('Select Model:', list(models.keys()))
    selected_model = models[model_selection]

    # Prediction logic
    try:
        encoded_job = job_encoder.transform([job])[0]  # SimpleEncoder returns a single value

        input_data = [
            age,
            encoded_job,
            marital[1],
            education[1],
            default[1],
            balance,
            housing[1],
            loan[1],
            contact[1],
            day,
            month[1],
            duration,
            campaign,
            pdays,
            previous,
            poutcome[1],
        ]

        combined_input = np.array(input_data).reshape(1, -1)

        # Scale input if scaler is available
        if scaler is not None:
            combined_input = scaler.transform(combined_input)

        prediction = selected_model.predict(combined_input)
        prediction_proba = selected_model.predict_proba(combined_input)

        st.subheader('\n')
        if prediction[0] == 1:
            st.success(f'The client is likely to subscribe to the term deposit (Yes)!')
        else:
            st.info(f'The client is not likely to subscribe to the term deposit (No).')
        st.subheader(f'Subscribing Probability: {prediction_proba[0][1] * 100 :.2f} %')

    except Exception as e:
        st.error(f"An error occurred during prediction. Please check your input data and model expectations: {e}")

else:
    st.warning("No models found in the 'models' directory. Please upload your .pickle model files to the 'models' folder.")