
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="EDA Analysis", page_icon="📊", layout="wide")

def main():
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Comprehensive statistical analysis and data exploration")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("⚠️ Please load a dataset first from the main page.")
        return
    
    df = st.session_state.data
    
    # Sidebar controls
    st.sidebar.header("📈 Analysis Options")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["Statistical Summary", "Correlation Analysis", "Distribution Analysis", "Outlier Detection"]
    )
    
    # Main content based on analysis type
    if analysis_type == "Statistical Summary":
        statistical_summary(df)
    elif analysis_type == "Correlation Analysis":
        correlation_analysis(df)
    elif analysis_type == "Distribution Analysis":
        distribution_analysis(df)
    elif analysis_type == "Outlier Detection":
        outlier_detection(df)

def statistical_summary(df):
    st.header("📊 Statistical Summary")
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Tabs for different summaries
    tab1, tab2, tab3 = st.tabs(["🔢 Numeric Summary", "📝 Categorical Summary", "📋 Overall Summary"])
    
    with tab1:
        if numeric_cols:
            st.subheader("Numeric Columns Analysis")
            
            # Basic statistics
            st.write("**Descriptive Statistics:**")
            desc_stats = df[numeric_cols].describe()
            st.dataframe(desc_stats)
            
            # Additional statistics
            st.write("**Additional Statistics:**")
            additional_stats = pd.DataFrame({
                'Column': numeric_cols,
                'Skewness': [df[col].skew() for col in numeric_cols],
                'Kurtosis': [df[col].kurtosis() for col in numeric_cols],
                'Variance': [df[col].var() for col in numeric_cols],
                'CV (%)': [abs(df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else 0 for col in numeric_cols]
            })
            st.dataframe(additional_stats)
            
        else:
            st.info("No numeric columns found in the dataset.")
    
    with tab2:
        if categorical_cols:
            st.subheader("Categorical Columns Analysis")
            
            for col in categorical_cols:
                with st.expander(f"Analysis for: {col}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Basic Info:**")
                        st.write(f"Unique values: {df[col].nunique()}")
                        st.write(f"Most frequent: {df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A'}")
                        st.write(f"Missing values: {df[col].isnull().sum()}")
                    
                    with col2:
                        st.write("**Value Counts (Top 10):**")
                        value_counts = df[col].value_counts().head(10)
                        st.dataframe(value_counts)
        else:
            st.info("No categorical columns found in the dataset.")
    
    with tab3:
        st.subheader("Overall Dataset Summary")
        
        # Dataset info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
            st.metric("Total Columns", df.shape[1])
        
        with col2:
            st.metric("Numeric Columns", len(numeric_cols))
            st.metric("Categorical Columns", len(categorical_cols))
        
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
            st.metric("Memory Usage (KB)", f"{df.memory_usage(deep=True).sum() / 1024:.1f}")

def correlation_analysis(df):
    st.header("🔗 Correlation Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return
    
    # Correlation matrix
    st.subheader("📊 Correlation Matrix")
    
    # Calculate correlation
    correlation_method = st.selectbox("Select correlation method:", ["pearson", "spearman", "kendall"])
    corr_matrix = df[numeric_cols].corr(method=correlation_method)
    
    # Display correlation heatmap
    fig = px.imshow(
        corr_matrix,
        title=f"{correlation_method.title()} Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Strong correlations
    st.subheader("🔍 Strong Correlations")
    
    # Find strong correlations (> 0.7 or < -0.7)
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                strong_corr.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Correlation': round(corr_val, 3),
                    'Strength': 'Strong Positive' if corr_val > 0 else 'Strong Negative'
                })
    
    if strong_corr:
        strong_corr_df = pd.DataFrame(strong_corr)
        st.dataframe(strong_corr_df)
    else:
        st.info("No strong correlations (|r| > 0.7) found.")
    
    # Pairwise analysis
    st.subheader("🎯 Pairwise Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("Select first variable:", numeric_cols)
    with col2:
        var2 = st.selectbox("Select second variable:", [col for col in numeric_cols if col != var1])
    
    if var1 and var2:
        # Scatter plot
        fig = px.scatter(
            df, x=var1, y=var2,
            title=f"Scatter Plot: {var1} vs {var2}",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation statistics
        correlation = df[var1].corr(df[var2])
        st.write(f"**Correlation coefficient:** {correlation:.4f}")
        
        # Statistical significance
        from scipy.stats import pearsonr
        stat, p_value = pearsonr(df[var1].dropna(), df[var2].dropna())
        st.write(f"**P-value:** {p_value:.4f}")
        st.write(f"**Significance:** {'Significant' if p_value < 0.05 else 'Not significant'} at α = 0.05")

def distribution_analysis(df):
    st.header("📈 Distribution Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Tabs for numeric and categorical
    tab1, tab2 = st.tabs(["🔢 Numeric Distributions", "📝 Categorical Distributions"])
    
    with tab1:
        if numeric_cols:
            selected_col = st.selectbox("Select numeric column:", numeric_cols)
            
            if selected_col:
                data = df[selected_col].dropna()
                
                # Distribution plots
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        df, x=selected_col,
                        title=f"Distribution of {selected_col}",
                        nbins=30
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(
                        df, y=selected_col,
                        title=f"Box Plot of {selected_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical tests
                st.subheader("📊 Distribution Statistics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write("**Basic Statistics:**")
                    st.write(f"Mean: {data.mean():.4f}")
                    st.write(f"Median: {data.median():.4f}")
                    st.write(f"Mode: {data.mode().iloc[0]:.4f}" if not data.mode().empty else "Mode: N/A")
                    st.write(f"Std Dev: {data.std():.4f}")
                
                with col2:
                    st.write("**Shape Statistics:**")
                    st.write(f"Skewness: {data.skew():.4f}")
                    st.write(f"Kurtosis: {data.kurtosis():.4f}")
                    st.write(f"Range: {data.max() - data.min():.4f}")
                    st.write(f"IQR: {data.quantile(0.75) - data.quantile(0.25):.4f}")
                
                with col3:
                    st.write("**Normality Test:**")
                    if len(data) >= 3 and len(data) <= 5000:
                        stat, p_value = stats.shapiro(data)
                        st.write(f"Shapiro-Wilk test:")
                        st.write(f"Statistic: {stat:.4f}")
                        st.write(f"P-value: {p_value:.4f}")
                        st.write(f"Normal: {'No' if p_value < 0.05 else 'Yes'}")
                    else:
                        st.write("Sample size not suitable for Shapiro-Wilk test")
        else:
            st.info("No numeric columns available for distribution analysis.")
    
    with tab2:
        if categorical_cols:
            selected_col = st.selectbox("Select categorical column:", categorical_cols)
            
            if selected_col:
                value_counts = df[selected_col].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar plot
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {selected_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Pie chart
                    fig = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"Proportion of {selected_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.subheader("📊 Categorical Statistics")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Info:**")
                    st.write(f"Unique categories: {df[selected_col].nunique()}")
                    st.write(f"Most frequent: {value_counts.index[0]}")
                    st.write(f"Frequency of most common: {value_counts.iloc[0]}")
                    st.write(f"Missing values: {df[selected_col].isnull().sum()}")
                
                with col2:
                    st.write("**Distribution Info:**")
                    entropy = stats.entropy(value_counts.values)
                    st.write(f"Entropy: {entropy:.4f}")
                    st.write(f"Concentration ratio (top 3): {value_counts.head(3).sum() / value_counts.sum():.4f}")
        else:
            st.info("No categorical columns available for distribution analysis.")

def outlier_detection(df):
    st.header("🎯 Outlier Detection")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("No numeric columns available for outlier detection.")
        return
    
    selected_col = st.selectbox("Select column for outlier detection:", numeric_cols)
    
    if selected_col:
        data = df[selected_col].dropna()
        
        # Different outlier detection methods
        st.subheader("🔍 Outlier Detection Methods")
        
        tab1, tab2, tab3 = st.tabs(["📊 IQR Method", "📈 Z-Score Method", "📋 Summary"])
        
        with tab1:
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**IQR Method Results:**")
                st.write(f"Q1 (25th percentile): {Q1:.4f}")
                st.write(f"Q3 (75th percentile): {Q3:.4f}")
                st.write(f"IQR: {IQR:.4f}")
                st.write(f"Lower bound: {lower_bound:.4f}")
                st.write(f"Upper bound: {upper_bound:.4f}")
                st.write(f"Number of outliers: {len(iqr_outliers)}")
                st.write(f"Percentage of outliers: {len(iqr_outliers)/len(data)*100:.2f}%")
            
            with col2:
                # Box plot with outliers highlighted
                fig = px.box(df, y=selected_col, title=f"Box Plot - {selected_col} (IQR Outliers)")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_threshold = st.slider("Z-score threshold:", 1.0, 4.0, 3.0, 0.1)
            z_outliers = data[z_scores > z_threshold]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Z-Score Method Results:**")
                st.write(f"Z-score threshold: {z_threshold}")
                st.write(f"Mean: {data.mean():.4f}")
                st.write(f"Standard deviation: {data.std():.4f}")
                st.write(f"Number of outliers: {len(z_outliers)}")
                st.write(f"Percentage of outliers: {len(z_outliers)/len(data)*100:.2f}%")
            
            with col2:
                # Histogram with outliers highlighted
                fig = px.histogram(df, x=selected_col, title=f"Distribution - {selected_col} (Z-score threshold: {z_threshold})")
                
                # Add vertical lines for outlier boundaries
                mean_val = data.mean()
                std_val = data.std()
                fig.add_vline(x=mean_val - z_threshold*std_val, line_dash="dash", line_color="red")
                fig.add_vline(x=mean_val + z_threshold*std_val, line_dash="dash", line_color="red")
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.write("**Outlier Summary:**")
            
            # Combine both methods
            iqr_outlier_indices = df[df[selected_col].isin(iqr_outliers)].index
            z_outlier_indices = df[df[selected_col].isin(z_outliers)].index
            combined_outliers = set(iqr_outlier_indices) | set(z_outlier_indices)
            
            summary_data = {
                'Method': ['IQR Method', 'Z-Score Method', 'Combined (Union)'],
                'Number of Outliers': [len(iqr_outliers), len(z_outliers), len(combined_outliers)],
                'Percentage': [
                    f"{len(iqr_outliers)/len(data)*100:.2f}%",
                    f"{len(z_outliers)/len(data)*100:.2f}%",
                    f"{len(combined_outliers)/len(data)*100:.2f}%"
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)
            
            if combined_outliers:
                st.write("**Sample Outliers:**")
                outlier_sample = df.loc[list(combined_outliers)[:10]]  # Show first 10 outliers
                st.dataframe(outlier_sample)

if __name__ == "__main__":
    main()
