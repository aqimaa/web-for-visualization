import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Set up Streamlit layout
st.title("Clustering Data Analysis")

# Sidebar for navigation with buttons
st.sidebar.title("Menu Options")

# CSS for uniform button size
st.markdown("""
    <style>
    .css-1aumxhk {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .stButton > button {
        width: 100%;
        font-size: 1.1em;
        padding: 0.6em;
        margin-bottom: 0.5em;
        background-color: #444;
        color: #FFF;
        border: 2px solid #999;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #333;
        color: #FFF;
    }
    .stButton > button:active {
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Define pages
pages = {
    "Upload Dataset": "upload",
    "Data Preprocessing": "preprocess",
    "Data Analysis": "analyze",
    "Data Visualization": "visualize"
}

# Default page
if 'page' not in st.session_state:
    st.session_state.page = "upload"

# Sidebar buttons for navigation
if st.sidebar.button("Upload Dataset"):
    st.session_state.page = "upload"
if st.sidebar.button("Data Preprocessing"):
    st.session_state.page = "preprocess"
if st.sidebar.button("Data Analysis"):
    st.session_state.page = "analyze"
if st.sidebar.button("Data Visualization"):
    st.session_state.page = "visualize"


# Global variable to hold dataset
if 'dataset' not in st.session_state:
    st.session_state['dataset'] = None

# Step 1: Upload Dataset
if st.session_state.page == "upload":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state['dataset'] = df
            st.write("Dataset loaded successfully!")
            st.write(df.head(10))
        except Exception as e:
            st.error(f"Error loading file: {e}")

# Step 2: Data Preprocessing
elif st.session_state.page == "preprocess":
    st.header("Data Preprocessing")
    if st.session_state['dataset'] is None:
        st.warning("Please upload a dataset first.")
    else:
        df = st.session_state['dataset']

        # Show basic dataset info
        st.subheader("Basic Information")
        st.write(df.describe())

        # Missing values handling
        st.subheader("Missing Values Check")
        if st.checkbox("Show Missing Values"):
            st.write(df.isnull().sum())
        
        # Option to fill or drop missing values
        preprocess_option = st.selectbox("Handle Missing Values:", ["None", "Fill with Mean", "Drop Rows"])
        if preprocess_option == "Fill with Mean":
            numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
            df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
            st.success("Missing values in numerical columns filled with mean.")
        elif preprocess_option == "Drop Rows":
            df = df.dropna()
            st.success("Rows with missing values dropped.")
        
        # Duplicate values handling
        st.subheader("Duplicate Data Check")
        
        # Option to check for duplicates
        if st.checkbox("Show Duplicate Rows"):
            duplicates = df[df.duplicated()]
            st.write(f"Number of duplicate rows: {len(duplicates)}")
            if len(duplicates) > 0:
                st.write(duplicates)  # Display duplicate rows if any

        # Option to remove duplicates
        remove_duplicates = st.checkbox("Remove Duplicate Rows")
        if remove_duplicates:
            df = df.drop_duplicates()
            st.success("Duplicate rows removed.")

        # Save changes back to session state
        st.session_state['dataset'] = df
        st.subheader("Updated Dataset")
        st.write(df.head(100))

# Step 3: Data Analysis
elif st.session_state.page == "analyze":
    st.header("Data Analysis")
    if st.session_state['dataset'] is None:
        st.warning("Please upload and preprocess a dataset first.")
    else:
        df = st.session_state['dataset']
        
        # Select clustering method
        clustering_method = st.selectbox("Select clustering method:", ["KMeans", "DBSCAN", "Hierarchical Clustering"])
            
        # Select only numerical columns for clustering
        numerical_df = df.select_dtypes(include=['float64', 'int64'])
            
        if numerical_df.empty:
            st.warning("No numerical data available for clustering.")
        else:
            if clustering_method == "KMeans":
                n_clusters = st.slider("Select number of clusters:", 2, 10, 3)
                    
                try:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(numerical_df)
                    df['Cluster'] = clusters
                    st.write("KMeans clustering completed. Cluster labels added to dataset.")
                except Exception as e:
                    st.error(f"Error with KMeans clustering: {e}")

            elif clustering_method == "DBSCAN":
                eps = st.slider("Select epsilon (eps):", 0.1, 10.0, 0.5)
                min_samples = st.slider("Select minimum samples:", 1, 20, 5)
                    
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(numerical_df)
                    df['Cluster'] = clusters
                    st.write("DBSCAN clustering completed. Cluster labels added to dataset.")
                except Exception as e:
                    st.error(f"Error with DBSCAN clustering: {e}")

            elif clustering_method == "Hierarchical Clustering":
                n_clusters = st.slider("Select number of clusters:", 2, 10, 3)
                    
                try:
                    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                    clusters = hierarchical.fit_predict(numerical_df)
                    df['Cluster'] = clusters
                    st.write("Hierarchical clustering completed. Cluster labels added to dataset.")
                except Exception as e:
                    st.error(f"Error with Hierarchical clustering: {e}")

            # Display updated dataset with clusters
            st.write(df.head(100))
                
            # Save changes back to session state
            st.session_state['dataset'] = df

# Step 4: Data Visualization
elif st.session_state.page == "visualize":
    st.header("Data Visualization")
    if st.session_state['dataset'] is None:
        st.warning("Please upload and preprocess a dataset first.")
    else:
        df = st.session_state['dataset']

        # Visualize clusters if they exist
        if 'Cluster' in df.columns:
            
            # Select only numerical columns and drop the 'Cluster' column
            numerical_df = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Cluster'], errors='ignore')
            
            if numerical_df.empty:
                st.warning("No numerical data available for visualization.")
            else:
                # Standardize data before PCA (skip if already standardized)
                scaler = StandardScaler()
                standardized_data = scaler.fit_transform(numerical_df)
                
                # Apply PCA
                pca = PCA(n_components=2)
                pca_components = pca.fit_transform(standardized_data)
                
                # Add PCA components and Cluster label back to the original dataframe
                df['PCA1'] = pca_components[:, 0]
                df['PCA2'] = pca_components[:, 1]

                # Clear the plot
                plt.clf()
                
                # Plotting PCA results with cluster labels
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette="viridis", s=100)
                plt.title("Cluster Visualization")
                plt.xlabel("PCA Component 1")
                plt.ylabel("PCA Component 2")
                plt.legend(title='Cluster')
                
                # Display plot in Streamlit
                st.pyplot(plt)
        else:
            st.write("No clustering information available. Please complete the analysis step first.")