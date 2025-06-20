import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

st.set_page_config(page_title="Fashion GAN Generator", layout="wide")

# Sidebar controls
st.sidebar.title("GAN Controls")
num_images = st.sidebar.slider("Number of images", min_value=4, max_value=16, value=8, step=4)
generate = st.sidebar.button("Generate Images")

# Load the pre-trained GAN model
@st.cache_resource
def load_gan_model(model_path):
    return load_model(model_path)

generator = load_gan_model('generatormodel.h5')

st.title("🧥 Fashion GAN Image Generator")
st.markdown(
    """
    Generate new Fashion-MNIST-like images using your trained GAN model.<br>
    Use the sidebar to select the number of images, then click **Generate Images**.
    """,
    unsafe_allow_html=True,
)

def generate_images(generator, num_images):
    noise = np.random.normal(0, 1, (num_images, 128))  # latent_dim fixed at 128
    generated_images = generator.predict(noise)
    generated_images = (generated_images - generated_images.min()) / (generated_images.max() - generated_images.min() + 1e-8)  # Normalize to [0,1]
    return generated_images

if generate:
    images = generate_images(generator, num_images)
    st.subheader("Generated Images")
    cols = st.columns(4)
    for idx, img in enumerate(images):
        col = cols[idx % 4]
        col.image(np.squeeze(img), use_container_width=True, channels="GRAY", caption=f"Image {idx+1}")

    st.success("Images generated! Adjust the controls and generate again for new results.")

st.markdown("---")
st.markdown(
    "<small>Made with ❤️ using Streamlit and Keras | [GitHub](https://github.com/)</small>",
    unsafe_allow_html=True,
)