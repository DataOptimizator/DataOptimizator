import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px

@st.cache_data
def load_and_preprocess_data(file):
    df = pd.read_csv(file, delimiter=';', engine='python')
    
    st.write("Data Preview:")
    st.write(df.head())
    
    st.write("Column Data Types:")
    st.write(df.dtypes)
    
    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if len(numeric_columns) == 0 and len(categorical_columns) == 0:
        st.error("The uploaded file does not contain any valid columns for analysis. Please check your data.")
        return None, None, None
    
    return df, numeric_columns, categorical_columns


def preprocess_data(df, selected_numeric, selected_categorical):
    # Combine selected columns
    selected_columns = selected_numeric + selected_categorical
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, selected_numeric),
            ('cat', categorical_transformer, selected_categorical)
        ])
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(df[selected_columns])
    
    # Get feature names after preprocessing
    numeric_features = selected_numeric
    categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(selected_categorical).tolist() if selected_categorical else []
    feature_names = numeric_features + categorical_features
    
    return X_preprocessed, feature_names, preprocessor

def perform_kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def get_feature_importance(X, labels, feature_names):
    importances = []
    for feature_idx in range(X.shape[1]):
        f_importances = []
        for cluster in np.unique(labels):
            f_imp = np.abs(np.mean(X[labels == cluster, feature_idx]) - np.mean(X[:, feature_idx]))
            f_importances.append(f_imp)
        importances.append(np.mean(f_importances))
    
    sorted_idx = np.argsort(importances)[::-1]
    return [feature_names[i] for i in sorted_idx], np.array(importances)[sorted_idx]

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

def plot_clusters_2d(X, labels, title):
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_2d = svd.fit_transform(X)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('First SVD Component')
    ax.set_ylabel('Second SVD Component')
    plt.colorbar(scatter)
    return fig


def plot_feature_importance(X, labels, feature_names, title):
    if isinstance(X, np.ndarray):
        importances = np.abs(np.mean(X[labels == 0], axis=0) - np.mean(X, axis=0))
    else:  # Sparse matrix
        importances = np.abs(X[labels == 0].mean(axis=0) - X.mean(axis=0)).A1
    
    # Sort features by importance
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    # Create an interactive bar plot using Plotly
    fig = go.Figure(go.Bar(
        x=sorted_importances,
        y=sorted_features,
        orientation='h'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Feature Importance',
        yaxis_title='Features',
        height=max(600, len(feature_names) * 20),  # Adjust height based on number of features
        width=800
    )
    
    return fig, sorted_features, sorted_importances


def plot_top_features(df, top_features, cluster_column, agg_func, title):
    df_agg = df.groupby(cluster_column)[top_features].agg(agg_func).reset_index()
    df_melted = df_agg.melt(id_vars=[cluster_column], var_name='Feature', value_name='Value')
    
    fig = px.bar(df_melted, x='Feature', y='Value', color=cluster_column, barmode='group',
                 title=f'{title} of Top 5 Features Across Clusters')
    fig.update_layout(xaxis_title='Features', yaxis_title=f'{agg_func.capitalize()} Value', legend_title='Cluster')
    
    return fig

def plot_top_features_by_cluster(X, labels, feature_names, top_n=5):
    if isinstance(X, np.ndarray):
        importances = np.abs(np.mean(X[labels == 0], axis=0) - np.mean(X, axis=0))
    else:  # Sparse matrix
        importances = np.abs(X[labels == 0].mean(axis=0) - X.mean(axis=0)).A1
    
    # Get top N features
    top_indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]
    
    # Calculate mean values for each cluster
    cluster_means = []
    for cluster in np.unique(labels):
        if isinstance(X, np.ndarray):
            cluster_mean = np.mean(X[labels == cluster][:, top_indices], axis=0)
        else:  # Sparse matrix
            cluster_mean = X[labels == cluster][:, top_indices].mean(axis=0).A1
        cluster_means.append(cluster_mean)
    
    # Create a DataFrame for plotting
    df_plot = pd.DataFrame(cluster_means, columns=top_features)
    df_plot['Cluster'] = [f'Cluster {i}' for i in range(len(cluster_means))]
    df_melted = df_plot.melt(id_vars=['Cluster'], var_name='Feature', value_name='Value')
    
    # Create the bar plot
    fig = px.bar(df_melted, x='Feature', y='Value', color='Cluster', barmode='group',
                 title=f'Top {top_n} Features Comparison Across Clusters (Scaled)')
    fig.update_layout(xaxis_title='Features', yaxis_title='Scaled Feature Value', legend_title='Cluster')
    
    return fig, top_indices, top_features

def plot_top_features_by_cluster_original_scale(X, labels, feature_names, top_indices, preprocessor, df, selected_numeric, selected_date):
    # Get the scaler for numeric features
    numeric_scaler = preprocessor.named_transformers_['num'].named_steps['scaler'] if 'num' in preprocessor.named_transformers_ else None
    
    # Calculate mean values for each cluster
    cluster_means = []
    for cluster in np.unique(labels):
        if isinstance(X, np.ndarray):
            cluster_mean = np.mean(X[labels == cluster][:, top_indices], axis=0)
        else:  # Sparse matrix
            cluster_mean = X[labels == cluster][:, top_indices].mean(axis=0).A1
        cluster_means.append(cluster_mean)
    
    # Create a DataFrame for plotting
    df_plot = pd.DataFrame(cluster_means, columns=[feature_names[i] for i in top_indices])
    
    # Inverse transform the scaled values for numeric features
    numeric_features = [f for f in df_plot.columns if f in selected_numeric]
    if numeric_features and numeric_scaler:
        numeric_indices = [list(feature_names).index(f) for f in numeric_features]
        df_plot[numeric_features] = numeric_scaler.inverse_transform(df_plot[numeric_features])
    
    # Convert date features back to datetime
    date_features = [f for f in df_plot.columns if f in selected_date]
    for date_feature in date_features:
        df_plot[date_feature] = pd.to_datetime(df_plot[date_feature] * 10**9)
    
    df_plot['Cluster'] = [f'Cluster {i}' for i in range(len(cluster_means))]
    df_melted = df_plot.melt(id_vars=['Cluster'], var_name='Feature', value_name='Value')
    
    # Create the bar plot
    fig = px.bar(df_melted, x='Feature', y='Value', color='Cluster', barmode='group',
                 title=f'Top Features Comparison Across Clusters (Original Scale)')
    fig.update_layout(xaxis_title='Features', yaxis_title='Original Feature Value', legend_title='Cluster')
    
    return fig


def plot_feature_across_clusters(df, feature, cluster_column):
    median_values = df.groupby(cluster_column)[feature].median().sort_values(ascending=False)
    mean_values = df.groupby(cluster_column)[feature].mean().sort_values(ascending=False)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=median_values.index,
        y=median_values.values,
        name='Median',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        x=mean_values.index,
        y=mean_values.values,
        name='Mean',
        marker_color='red'
    ))
    
    fig.update_layout(
        title=f'{feature} Across Clusters',
        xaxis_title='Cluster',
        yaxis_title='Value',
        barmode='group'
    )
    
    return fig

def plot_2d_kmeans(X, labels, n_clusters):
    # Perform PCA to reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a DataFrame for plotting
    df_plot = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': labels
    })
    
    # Calculate cluster centers
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X_pca)
    centers = kmeans.cluster_centers_
    
    # Create the scatter plot
    fig = px.scatter(df_plot, x='PC1', y='PC2', color='Cluster', 
                     title='2D Visualization of K-means Clustering')
    
    # Add cluster centers to the plot
    fig.add_trace(go.Scatter(x=centers[:, 0], y=centers[:, 1],
                             mode='markers',
                             marker=dict(color='black', size=10, symbol='x'),
                             name='Cluster Centers'))
    
    fig.update_layout(legend_title_text='Cluster')
    return fig

def main():
    st.title('Advanced Clustering Analysis for Buyer Personas')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df, numeric_columns, categorical_columns = load_and_preprocess_data(uploaded_file)
        
        if df is None:
            return  # Exit the function if data loading failed
        
        # Allow user to select columns for clustering
        selected_numeric = st.multiselect("Select numeric columns for clustering", options=numeric_columns, default=numeric_columns)
        selected_categorical = st.multiselect("Select categorical columns for clustering", options=categorical_columns, default=categorical_columns)
        
        if not (selected_numeric or selected_categorical):
            st.error("Please select at least one column for clustering.")
            return
        
        X_preprocessed, feature_names, preprocessor = preprocess_data(df, selected_numeric, selected_categorical)
        
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        
        if st.button('Run Clustering Analysis'):
            labels, kmeans = perform_kmeans(X_preprocessed, n_clusters)
            
            # Add cluster labels to the original dataframe
            df['k_means_cluster'] = labels
            
            st.write("\nK-means Clustering Results:")
            
            # Plot 2D visualization of K-means clustering
            fig_2d_kmeans = plot_2d_kmeans(X_preprocessed, labels, n_clusters)
            st.plotly_chart(fig_2d_kmeans)
            st.write("This plot shows a 2D representation of the clustering results. " 
                     "Each point represents a data point, colored by its assigned cluster. "
                     "The black X markers represent the cluster centers.")
            
            # Get feature importance
            top_features, importances = get_feature_importance(X_preprocessed, labels, feature_names)
            
            # Plot top 5 feature importances
            fig_importance = px.bar(x=top_features[:5], y=importances[:5], 
                                    labels={'x': 'Features', 'y': 'Importance'},
                                    title='Top 5 Feature Importances')
            st.plotly_chart(fig_importance)
            
            # Plot individual charts for top 5 features
            st.write("### Individual Feature Analysis")
            for feature in top_features[:5]:
                if feature in df.columns:  # Check if the feature is in the original dataframe
                    fig = plot_feature_across_clusters(df, feature, 'k_means_cluster')
                    st.plotly_chart(fig)
                else:
                    st.write(f"Feature '{feature}' not found in the original dataset. It might be an encoded categorical feature.")
            
            # Buyer Persona Analysis
            st.write("\n### Buyer Persona Analysis:")
            for cluster in range(n_clusters):
                st.write(f"Cluster {cluster}:")
                cluster_data = df[df['k_means_cluster'] == cluster]
                
                for col in selected_numeric + selected_categorical:
                    if df[col].dtype in ['int64', 'float64']:
                        st.write(f"{col}: Mean = {cluster_data[col].mean():.2f}, Median = {cluster_data[col].median():.2f}")
                    else:
                        st.write(f"{col}: Most common = {cluster_data[col].mode().values[0]}")
                
                st.write("---")

if __name__ == "__main__":
    main()