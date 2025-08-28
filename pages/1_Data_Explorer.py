
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")

def main():
    st.title("🔍 Data Explorer")
    st.markdown("Comprehensive data profiling and exploration tools")
    
    # Check if data is loaded
    if st.session_state.data is None:
        st.warning("⚠️ Please load a dataset first from the main page.")
        return
    
    df = st.session_state.data
    
    # Sidebar options
    st.sidebar.header("🔧 Explorer Options")
    
    # Data cleaning options
    st.sidebar.subheader("Data Cleaning")
    
    # Handle missing values
    if df.isnull().sum().sum() > 0:
        handle_missing = st.sidebar.checkbox("Handle Missing Values")
        if handle_missing:
            missing_strategy = st.sidebar.selectbox(
                "Missing Value Strategy:",
                ["Drop rows with missing values", "Fill with mean (numeric)", "Fill with mode (categorical)", "Fill with forward fill"]
            )
            
            if st.sidebar.button("Apply Missing Value Strategy"):
                df_cleaned = df.copy()  # Initialize with copy
                
                if missing_strategy == "Drop rows with missing values":
                    df_cleaned = df.dropna()
                elif missing_strategy == "Fill with mean (numeric)":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(df_cleaned[numeric_cols].mean())
                elif missing_strategy == "Fill with mode (categorical)":
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    for col in categorical_cols:
                        mode_value = df_cleaned[col].mode()
                        if not mode_value.empty:
                            df_cleaned[col] = df_cleaned[col].fillna(mode_value[0])
                elif missing_strategy == "Fill with forward fill":
                    df_cleaned = df.fillna(method='ffill')
                
                st.session_state.data = df_cleaned
                st.success("✅ Missing values handled successfully!")
                st.rerun()
    
    # Remove duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.sidebar.write(f"**Duplicate rows found**: {duplicate_count}")
        if st.sidebar.button("Remove Duplicates"):
            df_cleaned = df.drop_duplicates()
            st.session_state.data = df_cleaned
            st.success(f"✅ Removed {duplicate_count} duplicate rows!")
            st.rerun()
    
    # Main content
    tabs = st.tabs(["📊 Dataset Overview", "🔍 Column Analysis", "📈 Data Quality", "🧹 Data Cleaning"])
    
    with tabs[0]:
        dataset_overview(df)
    
    with tabs[1]:
        column_analysis(df)
    
    with tabs[2]:
        data_quality_analysis(df)
    
    with tabs[3]:
        data_cleaning_interface(df)

def dataset_overview(df):
    st.header("📊 Dataset Overview")
    
    # Basic metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col4:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col5:
        st.metric("Duplicates", f"{df.duplicated().sum():,}")
    
    # Data types summary
    st.subheader("📋 Data Types Summary")
    dtype_summary = df.dtypes.value_counts()
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Column Types Count:**")
        st.dataframe(dtype_summary.to_frame('Count'))
    
    with col2:
        st.write("**Numeric vs Categorical:**")
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        summary_df = pd.DataFrame({
            'Type': ['Numeric', 'Categorical'],
            'Count': [numeric_count, categorical_count]
        })
        st.dataframe(summary_df)
    
    # Sample data
    st.subheader("🔍 Sample Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First 5 rows:**")
        st.dataframe(df.head())
    
    with col2:
        st.write("**Last 5 rows:**")
        st.dataframe(df.tail())

def column_analysis(df):
    st.header("🔍 Detailed Column Analysis")
    
    # Column information table
    col_info = []
    for col in df.columns:
        info = {
            'Column': col,
            'Data Type': str(df[col].dtype),
            'Non-Null Count': df[col].count(),
            'Null Count': df[col].isnull().sum(),
            'Null %': round((df[col].isnull().sum() / len(df)) * 100, 2),
            'Unique Values': df[col].nunique(),
            'Unique %': round((df[col].nunique() / len(df)) * 100, 2)
        }
        
        if df[col].dtype in ['int64', 'float64']:
            info['Min'] = str(df[col].min())
            info['Max'] = str(df[col].max())
            info['Mean'] = str(round(df[col].mean(), 2))
            info['Std'] = str(round(df[col].std(), 2))
        else:
            info['Min'] = 'N/A'
            info['Max'] = 'N/A'
            info['Mean'] = 'N/A'
            info['Std'] = 'N/A'
            
        col_info.append(info)
    
    col_info_df = pd.DataFrame(col_info)
    st.dataframe(col_info_df, width='stretch')
    
    # Individual column analysis
    st.subheader("🎯 Individual Column Analysis")
    selected_col = st.selectbox("Select a column to analyze:", df.columns)
    
    if selected_col:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Analysis for: {selected_col}**")
            st.write(f"Data Type: {df[selected_col].dtype}")
            st.write(f"Non-null values: {df[selected_col].count():,}")
            st.write(f"Missing values: {df[selected_col].isnull().sum():,}")
            st.write(f"Unique values: {df[selected_col].nunique():,}")
            
            if df[selected_col].dtype in ['int64', 'float64']:
                st.write("**Statistical Summary:**")
                st.write(df[selected_col].describe())
        
        with col2:
            st.write("**Value Counts (Top 10):**")
            value_counts = df[selected_col].value_counts().head(10)
            st.dataframe(value_counts.to_frame('Count'))

def data_quality_analysis(df):
    st.header("📈 Data Quality Analysis")
    
    # Missing values analysis
    st.subheader("🔍 Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if not missing_data.empty:
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df)
    else:
        st.success("✅ No missing values found!")
    
    # Duplicate analysis
    st.subheader("🔄 Duplicate Rows Analysis")
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        st.warning(f"⚠️ Found {duplicate_count} duplicate rows ({duplicate_count/len(df)*100:.2f}% of data)")
        
        # Show sample duplicates
        if st.checkbox("Show sample duplicate rows"):
            duplicates = df[df.duplicated(keep=False)].head(10)
            st.dataframe(duplicates)
    else:
        st.success("✅ No duplicate rows found!")
    
    # Data type consistency
    st.subheader("🔧 Data Type Consistency")
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if numeric strings exist
            sample_values = df[col].dropna().head(100)
            numeric_like = 0
            for val in sample_values:
                try:
                    float(val)
                    numeric_like += 1
                except:
                    pass
            
            if numeric_like > len(sample_values) * 0.8:
                st.warning(f"⚠️ Column '{col}' might contain numeric data stored as text")

def data_cleaning_interface(df):
    st.header("🧹 Interactive Data Cleaning")
    
    st.subheader("🔧 Column Operations")
    
    # Column selection
    selected_col = st.selectbox("Select column for operations:", df.columns)
    
    if selected_col:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current column info:**")
            st.write(f"Data type: {df[selected_col].dtype}")
            st.write(f"Missing values: {df[selected_col].isnull().sum()}")
            st.write(f"Unique values: {df[selected_col].nunique()}")
        
        with col2:
            st.write("**Available operations:**")
            
            # Convert data type
            if st.button(f"Convert {selected_col} to numeric"):
                try:
                    df_temp = df.copy()
                    df_temp[selected_col] = pd.to_numeric(df_temp[selected_col], errors='coerce')
                    st.session_state.data = df_temp
                    st.success("✅ Column converted to numeric!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error converting column: {str(e)}")
            
            # Fill missing values
            if df[selected_col].isnull().sum() > 0:
                fill_value = st.text_input(f"Fill missing values in {selected_col} with:")
                if st.button("Fill Missing") and fill_value:
                    df_temp = df.copy()
                    df_temp[selected_col] = df_temp[selected_col].fillna(fill_value)
                    st.session_state.data = df_temp
                    st.success("✅ Missing values filled!")
                    st.rerun()
    
    # Global operations
    st.subheader("🌐 Global Operations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Remove All Missing Rows"):
            df_temp = df.dropna()
            st.session_state.data = df_temp
            st.success("✅ All rows with missing values removed!")
            st.rerun()
    
    with col2:
        if st.button("Remove Duplicate Rows"):
            df_temp = df.drop_duplicates()
            st.session_state.data = df_temp
            st.success("✅ Duplicate rows removed!")
            st.rerun()
    
    with col3:
        if st.button("Reset to Original"):
            st.warning("This will reset all changes!")
            if st.button("Confirm Reset"):
                # Note: This would need to be implemented with session state backup
                st.info("Reset functionality would restore original data")

if __name__ == "__main__":
    main()
