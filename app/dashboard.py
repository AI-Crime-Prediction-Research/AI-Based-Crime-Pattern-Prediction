import streamlit as st
import numpy as np

st.title("AI Crime Prediction System")

st.write("Spatio-Temporal Crime Prediction using Deep Learning")

# -------------------------------
# LOAD CNN FEATURES
# -------------------------------
try:
    cnn_features = np.load("data/video_data/cnn_features.npy")
    st.success("CNN features loaded successfully")
except:
    st.error("Run CNN model first")
    cnn_features = None

# -------------------------------
# SHOW DATA
# -------------------------------
if cnn_features is not None:
    st.write("Feature Shape:", cnn_features.shape)

# -------------------------------
# SIMPLE PREDICTION DEMO
# -------------------------------
if st.button("Run Prediction"):

    if cnn_features is not None:
        prediction = np.mean(cnn_features)
        st.write("Predicted Crime Risk Score:", round(float(prediction), 4))

        if prediction > 0.5:
            st.error("High Crime Risk Area 🚨")
        else:
            st.success("Low Crime Risk Area ✅")
uploaded_file = st.file_uploader("Upload a video", type=["mp4"])

if uploaded_file:
    st.video(uploaded_file)
import matplotlib.pyplot as plt

if st.button("Show Prediction Graph"):
    plt.figure()
    plt.plot(cnn_features[:100])
    st.pyplot(plt)
import seaborn as sns

if st.button("Show Heatmap"):
    plt.figure()
    sns.heatmap(cnn_features[:50])
    st.pyplot(plt)