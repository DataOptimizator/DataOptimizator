import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from scipy import stats

def load_data(file):
    df = pd.read_csv(file, delimiter=';', engine='python')
    return df

def prepare_date_features(df):
    date_columns = df.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        df[col] = pd.to_datetime(df[col]).astype(np.int64) // 10**9
    return df, date_columns

def preprocess_data(df):
    # Identify column types
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    
    # Handle dates
    df, date_columns = prepare_date_features(df)
    
    # StandardScaler for numeric columns
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # OneHotEncoder for categorical columns
    if len(categorical_columns) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_cats = encoder.fit_transform(df[categorical_columns])
        encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
        
        # Create DataFrame with encoded categories
        encoded_df = pd.DataFrame(encoded_cats, columns=encoded_feature_names)
        
        # Combine with scaled numeric data
        df_preprocessed = pd.concat([df_scaled[numeric_columns], encoded_df], axis=1)
    else:
        df_preprocessed = df_scaled[numeric_columns]
    
    return df_preprocessed, scaler, numeric_columns, categorical_columns

def plot_elbow_method(X):
    inertias = []
    K = range(1, 11)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig = go.Figure(data=go.Scatter(x=list(K), y=inertias, mode='lines+markers'))
    fig.update_layout(
        title='Elbow Method for Optimal k',
        xaxis_title='k',
        yaxis_title='Inertia',
        showlegend=False
    )
    return fig

def plot_2d_clusters(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    fig = px.scatter(
        x=X_pca[:, 0], 
        y=X_pca[:, 1], 
        color=labels,
        title='2D Visualization of Clusters',
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
    )
    return fig

def get_feature_importance(X, kmeans, feature_names):
    centroids = kmeans.cluster_centers_
    overall_mean = np.mean(X, axis=0)
    importance = np.mean([np.abs(centroid - overall_mean) for centroid in centroids], axis=0)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    return importance_df

def plot_top_features_by_cluster(df_preprocessed, labels, feature_names, scaler, numeric_columns, top_n=5):
    """Plot top features by cluster with correct inverse transformation"""
    # Get top features
    importance_df = get_feature_importance(df_preprocessed.values, 
                                         KMeans(n_clusters=len(np.unique(labels))).fit(df_preprocessed), 
                                         feature_names)
    top_features = importance_df['feature'].head(top_n).tolist()
    
    # Create cluster summary
    cluster_summary = []
    for cluster in np.unique(labels):
        cluster_data = df_preprocessed[labels == cluster]
        cluster_means = cluster_data[top_features].mean()
        
        # Create a dictionary for this cluster
        cluster_dict = {'Cluster': f'Cluster {cluster}'}
        
        # Handle inverse transform for numeric features
        for feature in top_features:
            if feature in numeric_columns:
                # Create a full-sized array for inverse transform
                transform_array = np.zeros((1, len(numeric_columns)))
                feature_idx = list(numeric_columns).index(feature)
                transform_array[0, feature_idx] = cluster_means[feature]
                
                # Inverse transform and get the specific feature value
                original_value = scaler.inverse_transform(transform_array)[0, feature_idx]
                cluster_dict[feature] = original_value
            else:
                cluster_dict[feature] = cluster_means[feature]
        
        cluster_summary.append(cluster_dict)
    
    cluster_df = pd.DataFrame(cluster_summary)
    
    # Create visualization
    fig = go.Figure()
    for feature in top_features:
        fig.add_trace(go.Bar(
            name=feature,
            x=cluster_df['Cluster'],
            y=cluster_df[feature],
            text=cluster_df[feature].round(2)
        ))
    
    fig.update_layout(
        title=f'Top {top_n} Features Across Clusters',
        barmode='group',
        xaxis_title='Cluster',
        yaxis_title='Value'
    )
    
    return fig, cluster_df


def analyze_nan_values(df):
    """Analyze and handle NaN values in the dataset"""
    # Calculate percentage of NaN values for each column
    nan_percentages = (df.isna().sum() / len(df) * 100).round(2)
    
    # Create a DataFrame with NaN analysis
    nan_analysis = pd.DataFrame({
        'Column': nan_percentages.index,
        'NaN %': nan_percentages.values
    })
    
    # Sort by NaN percentage
    nan_analysis = nan_analysis.sort_values('NaN %', ascending=False)
    
    # Create bar plot
    fig = px.bar(
        nan_analysis,
        x='Column',
        y='NaN %',
        title='Percentage of NaN Values by Column'
    )
    fig.update_layout(xaxis_tickangle=45)
    
    return fig, nan_analysis

def impute_nan_values(df):
    """Impute NaN values for columns with less than 5% missing values"""
    df_imputed = df.copy()
    nan_percentages = df.isna().sum() / len(df) * 100
    
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Columns to impute (less than 5% NaN)
    numeric_to_impute = [col for col in numeric_cols if nan_percentages[col] < 5 and nan_percentages[col] > 0]
    categorical_to_impute = [col for col in categorical_cols if nan_percentages[col] < 5 and nan_percentages[col] > 0]
    
    # Impute numeric columns with median
    if numeric_to_impute:
        imputer = SimpleImputer(strategy='median')
        df_imputed[numeric_to_impute] = imputer.fit_transform(df[numeric_to_impute])
    
    # Impute categorical columns with mode
    if categorical_to_impute:
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed[categorical_to_impute] = imputer.fit_transform(df[categorical_to_impute])
    
    return df_imputed, numeric_to_impute + categorical_to_impute

def plot_correlation_matrix(df):
    """Create correlation matrix plot for numeric columns"""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        title="Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_tickangle=45,
        yaxis_tickangle=0
    )
    
    return fig, corr_matrix

def analyze_outliers(df):
    """Analyze and handle outliers in numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_stats = {}
    df_no_outliers = df.copy()
    
    # Create subplots for boxplots
    fig = make_subplots(
        rows=len(numeric_cols), 
        cols=1,
        subplot_titles=numeric_cols
    )
    
    for i, col in enumerate(numeric_cols, 1):
        # Calculate quartiles and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        
        # Calculate outliers
        outliers = df[df[col] > upper_limit][col]
        outlier_percentage = (len(outliers) / len(df)) * 100
        
        # Store statistics
        outlier_stats[col] = {
            'outlier_percentage': outlier_percentage,
            'upper_limit': upper_limit
        }
        
        # Create boxplot
        fig.add_trace(
            go.Box(y=df[col], name=col),
            row=i, col=1
        )
        
        # Handle outliers by capping at upper limit
        df_no_outliers.loc[df_no_outliers[col] > upper_limit, col] = upper_limit
    
    # Update layout
    fig.update_layout(
        height=300*len(numeric_cols),
        showlegend=False,
        title_text="Box Plots with Outliers"
    )
    
    return fig, outlier_stats, df_no_outliers

def perform_eda(df):
    """Perform complete EDA with separated categorical and numeric analysis"""
    st.header("Exploratory Data Analysis")
    
    # 1. Analyze NaN values
    st.subheader("1. Missing Values Analysis")
    nan_fig, nan_analysis = analyze_nan_values(df)
    st.plotly_chart(nan_fig)
    
    # Display NaN percentages
    st.write("NaN Percentages by Column:")
    st.dataframe(nan_analysis)
    
    # Impute values if necessary
    df_imputed, imputed_cols = impute_nan_values(df)
    if imputed_cols:
        st.write(f"Columns imputed (< 5% NaN): {', '.join(imputed_cols)}")
    
    # 2. Categorical Data Analysis
    st.subheader("2. Categorical Data Analysis")
    cat_figs = analyze_categorical_data(df_imputed)
    if cat_figs:
        for fig in cat_figs:
            st.plotly_chart(fig)
    else:
        st.write("No categorical columns found in the dataset.")
    
    # 3. Numeric Data Analysis
    st.subheader("3. Numeric Data Analysis")
    num_fig, outlier_stats, df_clean = analyze_numeric_data(df_imputed)
    if num_fig:
        st.plotly_chart(num_fig)
        
        # Display outlier statistics
        st.write("Outlier Statistics:")
        outlier_summary = pd.DataFrame.from_dict(outlier_stats, orient='index')
        st.dataframe(outlier_summary)
    else:
        st.write("No numeric columns found in the dataset.")
        df_clean = df_imputed
    
    # 4. Correlation Analysis
    st.subheader("4. Correlation Analysis")
    corr_fig, corr_matrix = plot_correlation_matrix(df_clean)
    st.plotly_chart(corr_fig)
    
    # Display strongest correlations
    st.write("Top 10 Strongest Correlations:")
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    strong_corr = (upper_tri.stack()
                  .reset_index()
                  .rename(columns={'level_0': 'Feature 1', 'level_1': 'Feature 2', 0: 'Correlation'})
                  .sort_values('Correlation', key=abs, ascending=False)
                  .head(10))
    st.dataframe(strong_corr)
    
    return df_clean
def analyze_categorical_data(df):
    """Create bar charts for categorical columns"""
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    
    if len(categorical_cols) == 0:
        return None
    
    figs = []
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f'Distribution of {col}',
            labels={'x': col, 'y': 'Count'}
        )
        figs.append(fig)

def analyze_numeric_data(df):
    """Create box plots for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return None, None, df
    
    # Create box plots
    fig = make_subplots(
        rows=len(numeric_cols),
        cols=1,
        subplot_titles=numeric_cols
    )
    
    outlier_stats = {}
    df_no_outliers = df.copy()
    
    for i, col in enumerate(numeric_cols, 1):
        # Calculate quartiles and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 1.5 * IQR
        lower_limit = Q1 - 1.5 * IQR
        
        # Calculate outliers
        outliers = df[(df[col] > upper_limit) | (df[col] < lower_limit)][col]
        outlier_percentage = (len(outliers) / len(df)) * 100
        
        # Store statistics
        outlier_stats[col] = {
            'outlier_percentage': outlier_percentage,
            'upper_limit': upper_limit,
            'lower_limit': lower_limit
        }
        
        # Create boxplot
        fig.add_trace(
            go.Box(y=df[col], name=col, boxpoints='outliers'),
            row=i, col=1
        )
        
        # Handle outliers by capping
        df_no_outliers.loc[df_no_outliers[col] > upper_limit, col] = upper_limit
        df_no_outliers.loc[df_no_outliers[col] < lower_limit, col] = lower_limit
    
    fig.update_layout(
        height=300*len(numeric_cols),
        showlegend=False,
        title_text="Box Plots of Numeric Features"
    )
    
    return fig, outlier_stats, df_no_outliers

def create_custom_cluster_summary(df, labels, selected_columns):
    """Create a custom summary comparing clusters based on selected columns"""
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    
    summary_stats = []
    
    for cluster in range(len(np.unique(labels))):
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
        
        # Calculate statistics for each column
        stats_dict = {'Cluster': f'Cluster {cluster}'}
        
        for col in selected_columns:
            if df[col].dtype in [np.number]:
                # Numeric columns
                stats_dict.update({
                    f'{col} (mean)': cluster_data[col].mean(),
                    f'{col} (median)': cluster_data[col].median(),
                    f'{col} (std)': cluster_data[col].std()
                })
            else:
                # Categorical columns
                mode_value = cluster_data[col].mode().iloc[0] if not cluster_data[col].empty else 'N/A'
                mode_percent = (cluster_data[col] == mode_value).mean() * 100
                stats_dict.update({
                    f'{col} (most common)': mode_value,
                    f'{col} (% most common)': mode_percent
                })
        
        summary_stats.append(stats_dict)
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create visualizations for each selected column
    figs = []
    
    for col in selected_columns:
        if df[col].dtype in [np.number]:
            # Box plot for numeric columns
            fig = go.Figure()
            for cluster in range(len(np.unique(labels))):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
                fig.add_trace(go.Box(
                    y=cluster_data[col],
                    name=f'Cluster {cluster}',
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title=f'Distribution of {col} across Clusters',
                yaxis_title=col,
                showlegend=True
            )
            figs.append(fig)
        else:
            # Bar plot for categorical columns
            cluster_cats = []
            for cluster in range(len(np.unique(labels))):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
                value_counts = cluster_data[col].value_counts().reset_index()
                value_counts['Cluster'] = f'Cluster {cluster}'
                value_counts.columns = ['Value', 'Count', 'Cluster']
                cluster_cats.append(value_counts)
            
            cat_df = pd.concat(cluster_cats)
            
            fig = px.bar(
                cat_df,
                x='Value',
                y='Count',
                color='Cluster',
                title=f'Distribution of {col} across Clusters',
                barmode='group'
            )
            figs.append(fig)
    
    return summary_df, figs

def main():
    st.title('Enhanced Clustering Analysis with EDA')
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        # Perform EDA
        df_clean = perform_eda(df)
        
        # Continue with clustering analysis using clean data
        df_preprocessed, scaler, numeric_columns, categorical_columns = preprocess_data(df_clean)
        
        # Show elbow method
        st.subheader("Elbow Method for Optimal k Selection")
        elbow_fig = plot_elbow_method(df_preprocessed)
        st.plotly_chart(elbow_fig)
        
        # Let user select number of clusters
        n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=4)
        
        if st.button("Perform Clustering"):
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(df_preprocessed)
            
            # 2D visualization
            st.subheader("2D Cluster Visualization")
            cluster_fig = plot_2d_clusters(df_preprocessed, labels)
            st.plotly_chart(cluster_fig)
            
            # Feature importance and top features visualization
            st.subheader("Feature Importance Analysis")
            feature_fig, cluster_summary = plot_top_features_by_cluster(
                df_preprocessed, 
                labels, 
                df_preprocessed.columns, 
                scaler,
                numeric_columns
            )
            st.plotly_chart(feature_fig)
            
            # Custom Cluster Summary
            st.subheader("Custom Cluster Summary")
            # Let user select columns for custom summary
            all_columns = list(df_clean.columns)
            selected_columns = st.multiselect(
                "Select columns for cluster comparison",
                options=all_columns,
                default=all_columns[:5]  # Default to first 5 columns
            )
            
            if selected_columns:
                custom_summary, custom_figs = create_custom_cluster_summary(
                    df_clean, 
                    labels, 
                    selected_columns
                )
                
                # Display summary table
                st.write("Summary Statistics by Cluster:")
                st.dataframe(custom_summary)
                
                # Display visualizations
                st.write("Detailed Visualizations for Selected Features:")
                for fig in custom_figs:
                    st.plotly_chart(fig)
                
                # Export option
                if st.button("Download Summary as CSV"):
                    csv = custom_summary.to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name="cluster_summary.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    main()