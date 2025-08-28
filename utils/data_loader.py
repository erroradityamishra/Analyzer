
"""
Data Loading utility for local Data Science project.
Simplified version without Replit-specific dependencies.
"""

import pandas as pd
import numpy as np
from typing import Optional
import requests
from io import StringIO

class DataLoader:
    """
    A utility class for loading data from various sources in local environment.
    """
    
    def __init__(self):
        self.supported_formats = ['.csv']
    
    def load_csv(self, file_path_or_buffer, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file from file path or file buffer.
        
        Args:
            file_path_or_buffer: File path string or file-like object
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            # Default parameters for robust CSV loading
            default_params = {
                'encoding': 'utf-8',
                'low_memory': False,
                'na_values': ['', 'NA', 'N/A', 'null', 'NULL', 'None', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']
            }
            
            # Merge with user-provided parameters
            params = {**default_params, **kwargs}
            
            df = pd.read_csv(file_path_or_buffer, **params)
            
            # Basic data cleaning
            df = self._basic_data_cleaning(df)
            
            return df
            
        except UnicodeDecodeError:
            # Try different encoding if UTF-8 fails
            try:
                params['encoding'] = 'latin-1'
                df = pd.read_csv(file_path_or_buffer, **params)
                df = self._basic_data_cleaning(df)
                return df
            except Exception as e:
                raise Exception(f"Error loading CSV with different encodings: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def load_url(self, url: str, **kwargs) -> pd.DataFrame:
        """
        Load a CSV file from a URL.
        
        Args:
            url: URL string pointing to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            # Download the file
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Convert to StringIO for pandas
            csv_data = StringIO(response.text)
            
            return self.load_csv(csv_data, **kwargs)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error downloading file from URL: {str(e)}")
        except Exception as e:
            raise Exception(f"Error loading CSV from URL: {str(e)}")
    
    def _basic_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Args:
            df: Input dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all')  # Remove rows where all values are NaN
            df = df.dropna(axis=1, how='all')  # Remove columns where all values are NaN
            
            # Strip whitespace from string columns
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()
                    # Convert 'nan' strings back to actual NaN
                    df[col] = df[col].replace('nan', np.nan)
            
            # Reset index after dropping rows
            df = df.reset_index(drop=True)
            
            return df
            
        except Exception as e:
            # If cleaning fails, return original dataframe
            print(f"Warning: Basic data cleaning failed: {str(e)}")
            return df
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Dictionary containing data information
        """
        try:
            info = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum(),
                'missing_values': df.isnull().sum().to_dict(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'duplicate_rows': df.duplicated().sum(),
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
                'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
            }
            
            # Basic statistics for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                info['numeric_stats'] = numeric_df.describe().to_dict()
            
            return info
            
        except Exception as e:
            return {'error': f"Error getting data info: {str(e)}"}
    
    def validate_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Validate the dataframe and identify potential issues.
        
        Args:
            df: Input dataframe
            
        Returns:
            dict: Validation results
        """
        try:
            validation = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'recommendations': []
            }
            
            # Check if dataframe is empty
            if df.empty:
                validation['errors'].append("Dataframe is empty")
                validation['is_valid'] = False
                return validation
            
            # Check for high percentage of missing values
            missing_pct = (df.isnull().sum() / len(df) * 100)
            high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
            if high_missing_cols:
                validation['warnings'].append(f"Columns with >50% missing values: {high_missing_cols}")
            
            # Check for duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                validation['warnings'].append(f"Found {duplicate_count} duplicate rows")
                validation['recommendations'].append("Consider removing duplicate rows")
            
            # Check for columns with single unique value
            single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
            if single_value_cols:
                validation['warnings'].append(f"Columns with single unique value: {single_value_cols}")
                validation['recommendations'].append("Consider removing constant columns")
            
            # Check for very high cardinality in categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            high_cardinality_cols = []
            for col in categorical_cols:
                if df[col].nunique() > len(df) * 0.5:  # More than 50% unique values
                    high_cardinality_cols.append(col)
            
            if high_cardinality_cols:
                validation['warnings'].append(f"High cardinality categorical columns: {high_cardinality_cols}")
                validation['recommendations'].append("Consider feature engineering for high cardinality categorical columns")
            
            return validation
            
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [f"Error during validation: {str(e)}"],
                'warnings': [],
                'recommendations': []
            }

# Utility functions
def load_sample_datasets():
    """
    Get a dictionary of sample datasets URLs.
    
    Returns:
        dict: Dictionary of dataset names and URLs
    """
    return {
        "Iris Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        "Tips Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv", 
        "Titanic Dataset": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
        "Boston Housing": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        "Wine Quality": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        "Car Evaluation": "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    }

def get_file_info(file_path: str) -> dict:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        dict: File information
    """
    try:
        import os
        
        if not os.path.exists(file_path):
            return {'error': 'File does not exist'}
        
        file_size = os.path.getsize(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        return {
            'file_name': os.path.basename(file_path),
            'file_size': file_size,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'file_extension': file_extension,
            'is_supported': file_extension in ['.csv'],
            'full_path': os.path.abspath(file_path)
        }
        
    except Exception as e:
        return {'error': f"Error getting file info: {str(e)}"}
