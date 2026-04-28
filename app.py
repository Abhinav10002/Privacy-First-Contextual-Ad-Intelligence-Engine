import streamlit as st
import pandas as pd  
import pickle
import os 
from PIL import Image 

st.set_page_config(page_title="Privacy-First Ad Engine", layout="wide")
st.title("🎯 Privacy-First Contextual Ad Engine")

st.markdown("""
This engine predicts the probability of an ad click based **only on context** (Time, Device, App) without using any personal tracking data.
""")

@st.cache_resource
def load_assets():
    with open('models/ad_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open('models/model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    return model, encoders, model_columns

try:
    model, encoders, model_columns = load_assets()
except FileNotFoundError:
    st.error("❌ Models not found. Please run train.py first!")
    st.stop()

st.sidebar.header("🌐 Live User Context")
hour = st.sidebar.slider("Hour of Day", 0, 23, 12)
day = st.sidebar.selectbox("Day of Week", 
                           options=[0,1,2,3,4,5,6], 
                           format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
is_4g = st.sidebar.radio("Connection Type", options=[0, 1], format_func=lambda x: "4G" if x==1 else "3G/Other")

os_options = encoders['os_version'].classes_.tolist()
os_ver = st.sidebar.selectbox("OS Version", options=os_options)

st.subheader("🚀 Candidate Ad Ranking")
st.write("Below are 5 potential ads (App Codes) currently in the inventory.")

candidate_apps = [123, 456, 789, 101, 202]

if st.button("Rank Ads Now"):
    results = []

    for app in candidate_apps:
        input_data = {
            'os_version': encoders['os_version'].transform([str(os_ver)])[0],
            'is_4G': is_4g,
            'hour': hour,
            'day_of_week': day,
            'app_code': encoders['app_code'].transform([str(app)])[0] if str(app) in encoders['app_code'].classes_ else 0
        }

        input_df = pd.DataFrame([input_data])[model_columns]

        prob = model.predict_proba(input_df)[0][1]
        results.append({"Ad (App Code)": app, "Click Probability": f"{prob*100:.2f}%", "Score": prob})

    res_df = pd.DataFrame(results).sort_values(by="Score", ascending=False).drop(columns=["Score"])
    st.table(res_df)
    st.success(f"Best Ad to show: App {res_df.iloc[0]['Ad (App Code)']}")


st.divider()
st.subheader("🧠 Why these results?")
col1, col2 = st.columns([1, 1])

with col1:
    st.write("""
    The chart on the right shows the **Global Feature Importance**. 
    It reveals which contextual factors generally drive clicks across your entire dataset.
    """)

with col2:
    if os.path.exists('reports/shap_summary.png'):
        image = Image.open('reports/shap_summary.png')
        st.image(image, caption="SHAP Global Explainer", use_container_width=True)
    else:
        st.warning("Run src/explain.py to see the SHAP report here!")