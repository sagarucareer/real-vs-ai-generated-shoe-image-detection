import streamlit as st

from PIL import Image

from utils import predict_image

#Configure the Page
st.set_page_config(
    page_title="Shoe Authenticity Detector",
    page_icon="👟",
    layout="centered"
)

#Title & Description
st.title("👟 Real vs AI-Generated Shoe Detection")
st.write(
    """
    Upload a shoe image and the model will predict whether it is
    **Real** or **AI-generated**.
    """
)

#Upload an Image
uploaded_file = st.file_uploader(
    "Choose a shoe image...",
    type=["jpg", "jpeg", "png", "webp"]
)

#Check if an Image was Uploaded
if uploaded_file is not None:
    
    #Open & Display the Image
    img = Image.open(uploaded_file)
    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    #Make the Prediction
    with st.spinner("🔍 Analyzing image..."):
        label, confidence = predict_image(img)

    #Display the Prediction
    st.markdown("---")
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2%}")
    st.progress(float(confidence))

    if label == "Real":
        st.success("✅ This appears to be a real shoe.")
    else:
        st.error("⚠️ This appears to be an AI-generated shoe.")