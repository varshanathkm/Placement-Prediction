import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import base64

# Load the model and preprocessing files
image_path = "/Users/varshanathkm/Desktop/final/charles-forerunner-3fPXt37X6UQ-unsplash.jpg"
model_path = '/Users/varshanathkm/Desktop/final/model.sav'
scaler_path = '/Users/varshanathkm/Desktop/final/scaler.sav'
encoding_files = {'ECA': '/Users/varshanathkm/Desktop/final/ECA.sav', 'PT': '/Users/varshanathkm/Desktop/final/PT.sav'}

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
encoders = {key: pickle.load(open(path, 'rb')) for key, path in encoding_files.items()}

def get_unique_values_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=['PlacementStatus', 'StudentID'], errors='ignore') 
    return {col: df[col].dropna().unique().tolist() for col in df.columns}, df.dtypes.to_dict()

data_path = '/Users/varshanathkm/Desktop/Dataset project/placementdata copy.csv'
data_columns, data_types = get_unique_values_from_csv(data_path)

st.set_page_config(page_title="Placement Eligibility Prediction", layout="wide")

# Function to set a local background image
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_background_local(image_path):
    if os.path.exists(image_path):  # Check if the file exists
        base64_str = get_base64_image(image_path)
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{base64_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stSidebar {{
            background: linear-gradient(135deg, #001F3F, #003366);

        }}
        label {{
            color: red !important;
            font-weight: bold;
        }}
        .stTextInput, .stNumberInput, .stSelectbox {{
            background: rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            color: red !important;
        }}
        input {{
            color: white  !important;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    else:
        st.warning("⚠ Background image not found. Please check the file path.")

# Call with a local image path (ensure this image exists)
set_background_local(image_path)

st.sidebar.title("Main Menu:")

if "page" not in st.session_state:
    st.session_state.page = "Placement Eligibility Prediction"

def go_to_placement_page():
    st.session_state.page = "Placement Eligibility Prediction"

def go_to_company_page():
    st.session_state.page = "Company Suggestions"

st.sidebar.button("Placement Eligibility Prediction", on_click=go_to_placement_page)
st.sidebar.button("Company Suggestions", on_click=go_to_company_page)

if st.session_state.page == "Placement Eligibility Prediction":
    st.title("Placement Eligibility Prediction")
    
    input_data = {}
    
    # Text input fields with appropriate validation
    for col, dtype in data_types.items():
        if col not in ['ExtracurricularActivities', 'PlacementTraining']:
            if np.issubdtype(dtype, np.integer):
                input_value = st.number_input(f"**{col}**", step=1, format="%d")
                input_data[col] = input_value
            elif np.issubdtype(dtype, np.floating):
                input_value = st.number_input(f"**{col}**", format="%.2f")
                input_data[col] = input_value
            else:
                input_value = st.text_input(f"**{col}**")
                if not input_value.isalpha():
                    st.error(f"{col} must be a valid text input.")
                input_data[col] = input_value
    
    # Dropdowns for categorical variables
    input_data['ExtracurricularActivities'] = st.selectbox("**Extracurricular Activities**", options=data_columns.get('ExtracurricularActivities', []))
    input_data['PlacementTraining'] = st.selectbox("**Placement Training**", options=data_columns.get('PlacementTraining', []))
    
    if st.button("Predict"):
        if None in input_data.values() or "" in input_data.values():
            st.error("Please fill in all required fields correctly.")
        else:
            try:
                # Convert categorical data using one-hot encoding
                df_input = pd.DataFrame([input_data])
                for key, encoder in encoders.items():
                    if key in df_input:
                        transformed = encoder.transform(df_input[[key]])
                        encoded_cols = encoder.get_feature_names_out([key])
                        df_encoded = pd.DataFrame(transformed, columns=encoded_cols)
                        df_input = df_input.drop(columns=[key]).join(df_encoded)
                
                # Load the feature names from training data
                expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
                
                if expected_features is None:
                    expected_features = list(df_input.columns)
                
                # Ensure input has the same columns as training data
                for col in expected_features:
                    if col not in df_input.columns:
                        df_input[col] = 0
                df_input = df_input[expected_features]
                
                # Scale the input data
                df_input_scaled = scaler.transform(df_input)
                
                # Predict
                prediction = model.predict(df_input_scaled)
                
                if prediction[0] == 1:
                    st.success("🎉 Congratulations! You are eligible for placement.")
                    st.info("Here are some top companies you can apply to:")
                    st.write(["Google", "Microsoft", "Amazon", "Facebook", "Apple"])
                else:
                    st.error(" Unfortunately, you are not eligible for placement.")
                    st.info("Here are some small companies you can apply to:")
                    st.write(["Lonex tech", "Light vision", "honor", "onex technologies"])
            except Exception as e:
                st.error(f"⚠ Error in prediction: {str(e)}")

elif st.session_state.page == "Company Suggestions":
    st.title("Additional Companies to Apply")
    st.markdown("<h2 style='color: #008080;'> If you didn't qualify, consider applying to these small companies:</h2>", unsafe_allow_html=True)
    
    companies = ["techo magix", "Tech Innovators", "IT Solutions", "GrowthCorp"]
    
    for company in companies:
        st.markdown(f"<h3 style='color: white; font-weight:bold;'>{company}</h3>", unsafe_allow_html=True)
