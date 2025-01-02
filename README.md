# Land Cover Analysis Clustering with K-Means

This application implements **K-Means Clustering** for land cover image analysis. It is designed to visualize the clustering results of aerial images and evaluate them using the **Silhouette Coefficient**.

## Group Members
| Name                     | NPM          |
|--------------------------|--------------|
| Muhammad Nefrit Mahardika| 140810220006 |
| Rafa Agustant            | 140810220016 |
| Farhan Karisma           | 140810220042 |

## Application Features
1. **Image Uploading**: Users can upload images in `.jpg`, `.png`, or `.jpeg` formats.
2. **Clustering with K-Means**: The images are processed using the K-Means algorithm, with the number of clusters selectable (2-5 clusters).
3. **Silhouette Coefficient**: Evaluates clustering quality by calculating the average **Silhouette Coefficient**.
4. **Result Visualization**: Displays the original and clustered images side-by-side.

## Technologies Used
- **Streamlit**: Framework for building interactive web interfaces.
- **NumPy**: For numerical data manipulation.
- **OpenCV**: For image processing.
- **Pillow**: For loading images in various formats.

## How to Use the Application
1. **Environment Setup**:
   - Ensure Python 3.x is installed.
   - Install all dependencies with the following command:
     ```bash
     pip install -r requirements.txt
     ```
2. **Running the Application**:
   - Launch the application using the following command:
     ```bash
     streamlit run app.py
     ```
3. **Interacting with the Application**:
   - Upload an image through the interface.
   - Select the number of clusters (2-5).
   - View the clustering results and the average **Silhouette Coefficient**.

## Directory Structure
|-- app.py # Main application file |-- requirements.txt # List of dependencies |-- README.md 

# Project documentation

## Algorithm Explanation
### K-Means Clustering
1. **Centroid Initialization**: Centroids are randomly chosen from the image pixels.
2. **Assign Clusters**: Each pixel is assigned to the nearest centroid.
3. **Update Centroids**: Calculate the mean of the pixels in each cluster to form new centroids.
4. **Iteration**: Steps 2-3 are repeated until the centroids no longer change or the maximum iteration limit is reached.

### Silhouette Coefficient
The **Silhouette Coefficient** is a metric to evaluate clustering quality:
- **+1**: Excellent clustering.
- **0**: Random clustering.
- **-1**: Poor clustering.
