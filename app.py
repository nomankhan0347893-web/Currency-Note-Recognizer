import streamlit as st
from PIL import Image
import os
import json
from utils.predict import load_model, predict


# PAGE CONFIG

st.set_page_config(
    page_title="Currency Note Recognition",
    layout="wide"
)


# CUSTOM CSS

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.main-title {
    text-align: center;
    font-size: 3rem;
    font-weight: bold;
    color: #00D4FF;
}

.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #A0A0A0;
    margin-bottom: 20px;
}

.info-card {
    padding: 15px;
    border-radius: 12px;
    background-color: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 25px;
    border-top: 1px solid #444;
}

.footer a {
    text-decoration: none;
    margin: 0 10px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# SIDEBAR

with st.sidebar:

    st.title(" How To Use")

    st.markdown("""
    ### Steps

    1️⃣ Upload a currency note image

    2️⃣ Wait for AI analysis

    3️⃣ View prediction result

    4️⃣ Check confidence score

    5️⃣ Review all class probabilities

    ---

    ### Supported Formats

    JPG

    JPEG

    PNG

    ---

    ### Best Results

    ✔ Good Lighting

    ✔ Clear Image

    ✔ Single Currency Note

    ✔ Front Side Visible

    ---

    ### AI Model

    - ResNet50
    - TensorFlow
    - Transfer Learning
    """)


# HEADER

st.markdown(
    """
    <div class="main-title">
         Currency Note Recognition
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-Powered Pakistani 🇵🇰 & Afghani 🇦🇫 Currency Recognition
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# LOAD MODEL

@st.cache_resource
def get_model():

    if not os.path.exists("models/model.keras"):
        st.error(" Model file not found!")
        return None, None

    return load_model("models/model.keras")


model, class_names = get_model()

if model is None:
    st.error("No model found! Run train_model.py first.")
    st.stop()


# MODEL STATUS

if os.path.exists("models/history.json"):

    history = json.load(open("models/history.json"))

    best_acc = max(history["val_accuracy"])

    st.success(
        f" Model Loaded Successfully | Best Validation Accuracy: {best_acc*100:.2f}%"
    )

else:
    st.success(" Model Loaded Successfully")

st.divider()


# UPLOAD IMAGE

uploaded = st.file_uploader(
    "Upload Currency Note Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1])

    
    # IMAGE PREVIEW
    
    with col1:

        st.subheader(" Uploaded Image")

        st.image(
            image,
            caption="Currency Note",
            width="stretch"
        )

   
    # PREDICTION
    
    with col2:

        st.subheader(" AI Prediction")

        with st.spinner("Analyzing Currency Note..."):

            pred_class, confidence, all_probs = predict(
                image,
                model,
                class_names
            )

        flag = "🇦🇫" if pred_class == "Afghani" else "🇵🇰"

        if confidence >= 80:

            st.success(
                f"{flag} Prediction: {pred_class}"
            )

        elif confidence >= 60:

            st.warning(
                f"{flag} Prediction: {pred_class} (Moderate Confidence)"
            )

        else:

            st.error(
                f"{flag} Prediction: {pred_class} (Low Confidence)"
            )

        st.metric(
            "Confidence Score",
            f"{confidence:.2f}%"
        )

    st.divider()

    
    # PROBABILITIES

    st.subheader(" Class Probabilities")

    for cls, prob in all_probs.items():

        flag = "🇦🇫" if cls == "Afghani" else "🇵🇰"

        st.write(
            f"{flag} **{cls}** : {prob:.2f}%"
        )

        st.progress(float(prob) / 100)

    st.divider()

    st.info(
        f" **{pred_class}** detected with **{confidence:.2f}%** confidence."
    )

else:

    st.info(
        " Upload a currency note image to begin recognition."
    )


# ABOUT PROJECT

st.divider()

st.subheader("ℹ About This Project")

st.markdown("""
This project uses **ResNet50 Transfer Learning** to classify currency notes.

### Features

- Deep Learning Based Recognition
- Transfer Learning using ResNet50
- Real-Time Prediction
- Confidence Scoring
- Streamlit Web Interface

### Technologies

- Python
- TensorFlow
- ResNet50
- Streamlit
- Pillow
""")

# =====================================
# FOOTER
# =====================================
st.markdown("""
<div class="footer">

<h3>AI Developer</h3>

<p>
<strong>Noman khan</strong>
</p>

<a href="https://www.linkedin.com/in/noman-khan-95787139b/?lipi=urn%3Ali%3Apage%3Ad_flagship3_feed%3BqKO3u0D%2FTE%2Bfs6F8KG%2BVwQ%3D%3D" target="_blank">
🔗 LinkedIn
</a>

<a href="https://github.com/YOUR_GITHUB" target="_blank">
💻 GitHub
</a>

</a>

<br><br>

<small>
Currency Note Recognition System <br>
Built using ResNet50 • TensorFlow • Streamlit
</small>

</div>
""", unsafe_allow_html=True)