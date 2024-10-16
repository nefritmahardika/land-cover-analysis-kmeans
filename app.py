import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image

# Title of the app
st.title('Land Cover Analysis Clustering with K-Means')
# Add a subtitle
st.subheader('Project Data Mining oleh Muhammad Nefrit Mahardika, Rafa Agustant, dan Farhan Karisma')


# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Select number of clusters (from 2 to 5) using a dropdown menu
clusters = st.selectbox('Select number of clusters (2-5):', options=[2, 3, 4, 5])

# Function to perform K-Means clustering
def kmeans_clustering(image, n_clusters):
    # Convert image to RGB if it's not already in RGB
    if len(image.shape) == 2:  # If the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    elif image.shape[2] == 4:  # If the image has an alpha channel, remove it
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Reshape the image to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)
    
    # Replace each pixel with its cluster center
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = clustered.reshape(image.shape).astype(np.uint8)
    
    return clustered_image

# If an image is uploaded
if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    # Perform K-Means clustering
    clustered_image = kmeans_clustering(image, clusters)
    
    # Display the images side by side in a 1x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.image(clustered_image, caption=f"Clustered Image with {clusters} clusters", use_column_width=True)
