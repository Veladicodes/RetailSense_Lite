
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
import streamlit as st
from typing import Dict, Optional, Tuple, Any
import difflib

warnings.filterwarnings('ignore')


def load_csv_bytes(file_obj) -> pd.DataFrame:
    """
    Read CSV from file object, handle duplicate headers, and normalize column names.
    
    Args:
        file_obj: File-like object (e.g., from st.file_uploader)
    
    Returns:
        pd.DataFrame: DataFrame with normalized and deduplicated column names
    """
    try:
        # Reset file pointer
        file_obj.seek(0)
        
        # Read CSV with parse_dates=False initially (we'll parse after mapping)
        df = pd.read_csv(file_obj, parse_dates=False, low_memory=False, encoding='utf-8')
        
        # Handle duplicate headers
        df = dedupe_columns(df)
        
        # Normalize column names
        df = normalize_columns(df)
        
        # Store in session state
        st.session_state['raw_df'] = df.copy()
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names: lowercase, strip whitespace, replace spaces with underscores.
    
    Args:
        df: Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with normalized column names
    """
    df = df.copy()
    
    # Normalize: lowercase, strip, replace spaces with underscores
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('/', '_')
    )
    
    # Ensure unique column names after normalization
    df.columns = _make_unique_columns(df.columns)
    
    return df


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename duplicate column names by appending suffix _dup1, _dup2, etc.
    
    Args:
        df: Input DataFrame (may have duplicate column names)
    
    Returns:
        pd.DataFrame: DataFrame with unique column names
    """
    df = df.copy()
    
    if df.columns.duplicated().any():
        seen = {}
        new_columns = []
        
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_col = f"{col}_dup{seen[col]}"
                new_columns.append(new_col)
            else:
                seen[col] = 0
                new_columns.append(col)
        
        df.columns = new_columns
    
    return df


def _make_unique_columns(columns: pd.Index) -> pd.Index:
    """
    Ensure all column names are unique by appending incremental suffix if needed.
    
    Args:
        columns: pandas Index of column names
    
    Returns:
        pd.Index: Index with unique column names
    """
    seen = {}
    new_columns = []
    
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    
    return pd.Index(new_columns)


@st.cache_data
def get_quick_summary(df: pd.DataFrame, date_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate quick summary of dataset.
    
    Args:
        df: Input DataFrame
        date_col: Optional column name for date (to compute date range)
    
    Returns:
        dict: Summary with keys: rows, cols, date_range, n_products, top_categories
    """
    summary = {
        'rows': len(df),
        'cols': len(df.columns),
        'date_range': None,
        'n_products': 0,
        'top_categories': []
    }
    
    # Try to detect date column if not provided
    if date_col is None:
        date_candidates = [c for c in df.columns if any(
            term in c.lower() for term in ['date', 'time', 'week', 'day', 'ds']
        )]
        if date_candidates:
            date_col = date_candidates[0]
    
    # Parse date range if date column available
    if date_col and date_col in df.columns:
        try:
            dates = pd.to_datetime(df[date_col], errors='coerce')
            valid_dates = dates.dropna()
            if len(valid_dates) > 0:
                min_date = valid_dates.min()
                max_date = valid_dates.max()
                summary['date_range'] = (
                    min_date.strftime('%Y-%m-%d'),
                    max_date.strftime('%Y-%m-%d')
                )
        except Exception:
            pass
    
    # Detect product column
    product_candidates = [c for c in df.columns if any(
        term in c.lower() for term in ['product', 'item', 'sku', 'name']
    )]
    if product_candidates:
        product_col = product_candidates[0]
        try:
            summary['n_products'] = df[product_col].nunique()
        except Exception:
            pass
    
    # Detect category column
    category_candidates = [c for c in df.columns if any(
        term in c.lower() for term in ['category', 'department', 'aisle', 'class']
    )]
    if category_candidates:
        category_col = category_candidates[0]
        try:
            top_cats = df[category_col].value_counts().head(5).index.tolist()
            summary['top_categories'] = [str(c) for c in top_cats]
        except Exception:
            pass
    
    return summary


def validate_mapping(mapping: Dict[str, str], df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate that required columns (date, product, sales_qty) are mapped.
    
    Args:
        mapping: Dictionary mapping canonical names to actual column names
        df: DataFrame to validate against
    
    Returns:
        tuple: (is_valid, missing_columns_list)
    """
    required = ['date', 'product', 'sales_qty']
    missing = []
    
    for req_col in required:
        mapped_col = mapping.get(req_col)
        if not mapped_col or mapped_col not in df.columns:
            missing.append(req_col)
    
    return len(missing) == 0, missing


# Legacy functions for backward compatibility
class RetailDataLoader:
    """
    Retail Data Loader for Market Dataset.
    
    Features:
    - Load raw CSV data
    - Handle missing values & date parsing
    - Validate dataset quality
    - Create time-based, expiry, and pricing features
    - Extract product-level or store-level time series
    - Save processed datasets for later modeling
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Load and perform initial data cleaning"""
        print("📥 Loading retail data...")
        try:
            self.df = pd.read_csv(self.file_path)
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False


def load_dataset(file_path):
    """Load dataset from file path with error handling"""
    try:
        if not os.path.exists(file_path):
            return None
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        return None


def preprocess_data(df):
    """Preprocess data for analysis and forecasting"""
    if df is None or df.empty:
        return None
        
    try:
        # Convert date columns
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'week' in col.lower()]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        # Handle missing values
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
            
        return df
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return df
