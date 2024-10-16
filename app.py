import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image

st.title('Land Cover Analysis Clustering with K-Means')
st.subheader('Project Data Mining oleh Muhammad Nefrit Mahardika, Rafa Agustant, dan Farhan Karisma')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

clusters = st.selectbox('Select number of clusters (2-5):', options=[2, 3, 4, 5])

def kmeans_clustering(image, n_clusters):

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    pixels = image.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(pixels)
    
    clustered = kmeans.cluster_centers_[kmeans.labels_]
    clustered_image = clustered.reshape(image.shape).astype(np.uint8)
    
    return clustered_image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    clustered_image = kmeans_clustering(image, clusters)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.image(clustered_image, caption=f"Clustered Image with {clusters} clusters", use_column_width=True)
