import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_and_preprocess_data(file):
    # Try different delimiters, including space
 
    df = pd.read_csv(file, delimiter=';', engine='python')
  
    st.write("Data Preview:")
    st.write(df.head())
    
    st.write("Column Data Types:")
    st.write(df.dtypes)
    
    # Convert columns to numeric where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except:
            pass  # If conversion fails, leave the column as is
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        st.error("The uploaded file does not contain any numeric columns. Please upload a file with numeric data for clustering.")
        return None, None, None
    
    X = df[numeric_columns]
    X = X.fillna(X.mean())
    
    if X.empty:
        st.error("After preprocessing, no valid numeric data remains. Please check your data and try again.")
        return None, None, None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df, numeric_columns

# New function to perform elbow method for K-means
def elbow_method(X):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), wcss)
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('WCSS')
    return fig

# Modified perform_clustering function
def perform_clustering(X, algorithm, n_clusters=5):
    if algorithm == 'K-means':
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == 'Hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == 'DBSCAN':
        model = DBSCAN(eps=0.5, min_samples=5)
    elif algorithm == 'GMM':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    elif algorithm == 'Mean Shift':
        model = MeanShift()
    
    labels = model.fit_predict(X)
    return labels, model

# Function to plot clusters in 2D
def plot_clusters_2d(X, labels, title):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    plt.colorbar(scatter)
    return fig

# Function to plot feature importance
def plot_feature_importance(X, labels, feature_names, title):
    importances = np.abs(np.mean(X[labels == 0], axis=0) - np.mean(X, axis=0))
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(X.shape[1]), importances[indices])
    ax.set_title(title)
    ax.set_ylabel('Feature Importance')
    ax.set_xticks(range(X.shape[1]))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90)
    
    return fig

def main():
    st.title('Advanced Clustering Analysis')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        X, df, numeric_columns = load_and_preprocess_data(uploaded_file)
        
        if X is None:
            return  # Exit the function if data loading failed
        
        st.write("Data Preview:")
        st.write(df.head())
        
        algorithm = st.selectbox(
            "Select Clustering Algorithm",
            ('K-means', 'Hierarchical', 'DBSCAN', 'GMM', 'Mean Shift')
        )
        
        if algorithm == 'K-means':
            st.write("Elbow Method for K-means:")
            elbow_fig = elbow_method(X)
            st.pyplot(elbow_fig)
            st.write("Use the Elbow Method graph to choose the optimal number of clusters where the 'elbow' occurs.")
        
        if algorithm in ['K-means', 'Hierarchical', 'GMM']:
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        else:
            n_clusters = None
        
        if st.button('Run Clustering Analysis'):
            labels, model = perform_clustering(X, algorithm, n_clusters)
            
            st.write(f"\n{algorithm} Clustering Results:")
            
            # 2D visualization of clusters
            fig_clusters = plot_clusters_2d(X, labels, f'{algorithm} Clustering')
            st.pyplot(fig_clusters)
            st.write("This plot shows how the data points are grouped into clusters. " 
                     "Each color represents a different cluster.")
            
            # Feature importance plot
            fig_importance = plot_feature_importance(X, labels, numeric_columns, f'Feature Importance for {algorithm}')
            st.pyplot(fig_importance)
            st.write("This plot shows which features are most important in determining the clusters. " 
                     "Taller bars indicate more important features.")
            
            # Cluster statistics
            st.write("\nCluster Statistics:")
            for cluster in np.unique(labels):
                st.write(f"Cluster {cluster}:")
                cluster_data = df[labels == cluster]
                st.write(cluster_data.describe())
                st.write("---")

if __name__ == "__main__":
    main()