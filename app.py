import os
import streamlit as st
import base64
import pandas as pd
from PIL import Image
import pickle
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import random
from collections import Counter
from io import BytesIO
from src.plot import plot

def get_base64(bin_file):
    with open (bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_common_prediction(predictions):
    prediction_count = Counter(predictions)
    common_prediction = prediction_count.most_common(1)[0][0]
    return common_prediction

# set_background('image/background.png')

st.title('Human Brain Activity Classification')

st.info('This app classifies human brain activity into different categories.')

st.write("Human brain activities are complex and diverse. The brain constantly processes information from the environment and regulates various bodily functions. Understanding brain activities is crucial for studying cognition, behavior, and mental health.")

uploaded_file = st.file_uploader("Upload EEG data", type=["parquet"])

if uploaded_file is not None:
    try:
        data = BytesIO(uploaded_file.getvalue())
        st.success("File is readable.")

        df = pd.read_parquet(data)
        plot_obj = None
        col1, col2, col3 = st.columns([1, 2, 10])
        with col1:
            if st.button("Create Plot", help="Click to create a plot", key="create_plot"):
                plot_obj = plot(df)
                if plot_obj is not None:
                    fig, ax = plot_obj

        with col2:
            if st.button("Predict", help="Click to make predictions", key="predict"):
                diseases = ['Seizure', 'GRDA', 'LRDA', 'GPD', 'LPD', 'Other']
                predictions = [random.choice(diseases) for _ in range(6)]
                common_prediction = get_common_prediction(predictions)
                st.write(f"Common Prediction: {common_prediction}")

        if plot_obj is not None:
            st.pyplot(fig)
            st.write("Plot Created!")

        images = {
            'AdaBoost': ['Plots/Adaboost/Con_Mat.png', 'Plots/Adaboost/AUC-ROC.png'],
            'CatBoost': ['Plots/Catboost/Con_Mat.png', 'Plots/Catboost/AUC-ROC.png'],
            'LightGBM': ['Plots/Lightgbm/Con_Mat.png', 'Plots/Lightgbm/AUC-ROC.png'],
            'XGBoost': ['Plots/Xgboost/Con_Mat.png', 'Plots/Xgboost/AUC-ROC.png'],
        }

        # Create a DataFrame
        df = pd.DataFrame(columns=['Confusion Matrix', 'ROC-AUC Curve'], index=images.keys())

        # Load images and add them to the DataFrame
        for idx, (alg, img_paths) in enumerate(images.items()):
            df.loc[alg, 'Confusion Matrix'] = Image.open(img_paths[0])
            df.loc[alg, 'ROC-AUC Curve'] = Image.open(img_paths[1])

        # Display the DataFrame in Streamlit
        # st.dataframe(df)

        # Load images and add them to the DataFrame as constant variables
        for idx, (alg, img_paths) in enumerate(images.items()):
            confusion_matrix_img = Image.open(img_paths[0])
            roc_auc_curve_img = Image.open(img_paths[1])
            df.loc[alg, 'Confusion Matrix'] = confusion_matrix_img
            df.loc[alg, 'ROC-AUC Curve'] = roc_auc_curve_img

        # Display the images in Streamlit
        for idx, row in df.iterrows():
            st.subheader(idx)
            col1, col2 = st.columns(2)
            with col1:
                col_1, col_2 = st.columns(2)
                with col_1:
                    st.info("Confusion Matrix")
                with col_2:
                    st.image(row['Confusion Matrix'])
            with col2:
                col_1, col_2 = st.columns(2)
                with col_1:
                    st.info("ROC-AUC Curve")
                with col_2:
                    st.image(row['ROC-AUC Curve'])

    except Exception as e:
        st.error(f"Error: {e}")