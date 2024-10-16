import streamlit as st
import numpy as np
import cv2
from PIL import Image

st.title('Land Cover Analysis Clustering with K-Means')
st.subheader('Project Data Mining oleh Muhammad Nefrit Mahardika, Rafa Agustant, dan Farhan Karisma')

uploaded_file = st.file_uploader("Upload gambar...", type=["jpg", "png", "jpeg"])

clusters = st.selectbox('Pilih berapa cluster yang ingin dibuat (2-5):', options=[2, 3, 4, 5])

def initialize_centroids(pixels, n_clusters):
    """Inisialisasi centroid secara acak dari data piksel."""
    return pixels[np.random.choice(pixels.shape[0], n_clusters, replace=False)]

def assign_clusters(pixels, centroids):
    """Menetapkan setiap piksel ke centroid terdekat."""
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(pixels, labels, n_clusters):
    """Menghitung centroid baru berdasarkan piksel yang terdaftar."""
    return np.array([pixels[labels == i].mean(axis=0) for i in range(n_clusters)])

def kmeans_clustering(image, n_clusters, max_iter=100):
    """Implementasi algoritma K-Means secara manual."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    pixels = image.reshape(-1, 3)
    centroids = initialize_centroids(pixels, n_clusters)
    
    for _ in range(max_iter):
        labels = assign_clusters(pixels, centroids)
        new_centroids = update_centroids(pixels, labels, n_clusters)
        
        # Hentikan jika centroid tidak berubah
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    clustered = centroids[labels]
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
