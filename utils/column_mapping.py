
import pandas as pd
import streamlit as st
from typing import Dict, Optional, List
import difflib


def fuzzy_match_column(target: str, candidates: List[str], cutoff: float = 0.6) -> Optional[str]:
    """
    Find closest matching column name using fuzzy matching.
    
    Args:
        target: Target column name to match
        candidates: List of candidate column names
        cutoff: Minimum similarity score (0-1)
    
    Returns:
        str or None: Best matching column name, or None if no match above cutoff
    """
    if not candidates:
        return None
    
    # Create lowercase versions for matching
    candidates_lower = [c.lower() for c in candidates]
    matches = difflib.get_close_matches(
        target.lower(), 
        candidates_lower, 
        n=1, 
        cutoff=cutoff
    )
    
    if matches:
        # Find the index of the matched lowercase string
        matched_lower = matches[0]
        try:
            idx = candidates_lower.index(matched_lower)
            return candidates[idx]
        except (ValueError, IndexError):
            return None
    
    return None


def get_column_suggestions(candidates: List[str]) -> Dict[str, Optional[str]]:
    """
    Generate fuzzy-matched suggestions for canonical column names.
    
    Args:
        candidates: List of available column names in dataset
    
    Returns:
        dict: Mapping of canonical names to suggested column names
    """
    suggestions = {}
    
    # Date column suggestions
    date_patterns = ['date', 'week_start', 'ds', 'timestamp', 'time', 'week']
    suggestions['date'] = _match_patterns(date_patterns, candidates)
    
    # Product column suggestions
    product_patterns = ['product', 'product_name', 'sku', 'item', 'item_name', 'name']
    suggestions['product'] = _match_patterns(product_patterns, candidates)
    
    # Sales quantity suggestions
    sales_patterns = ['sales_qty', 'quantity', 'units', 'order_quantity', 'orders', 'y', 'sales']
    suggestions['sales_qty'] = _match_patterns(sales_patterns, candidates)
    
    # Price suggestions
    price_patterns = ['price', 'unit_price', 'selling_price', 'mrp', 'cost']
    suggestions['price'] = _match_patterns(price_patterns, candidates)
    
    # Stock suggestions
    stock_patterns = ['stock_on_hand', 'stock', 'inventory', 'on_hand']
    suggestions['stock_on_hand'] = _match_patterns(stock_patterns, candidates)
    
    # Category suggestions
    category_patterns = ['category', 'department', 'aisle', 'dept']
    suggestions['category'] = _match_patterns(category_patterns, candidates)
    
    # Store suggestions
    store_patterns = ['store', 'store_id', 'location', 'region']
    suggestions['store'] = _match_patterns(store_patterns, candidates)
    
    return suggestions


def _match_patterns(patterns: List[str], candidates: List[str]) -> Optional[str]:
    """Match patterns against candidates using fuzzy matching."""
    for pattern in patterns:
        match = fuzzy_match_column(pattern, candidates)
        if match:
            return match
    return None


def render_column_mapping(normalized_columns: List[str], default_map: Optional[Dict[str, str]] = None) -> Dict[str, Optional[str]]:
    """
    Render column mapping UI component with fuzzy-matched suggestions.
    
    Args:
        normalized_columns: List of normalized column names from dataset
        default_map: Optional default mapping dictionary
    
    Returns:
        dict: Mapping of canonical names to selected column names
    """
    # Get fuzzy-matched suggestions
    suggestions = get_column_suggestions(normalized_columns)
    
    # Merge with default_map (default_map takes precedence)
    if default_map:
        for key in suggestions:
            if key in default_map and default_map[key] in normalized_columns:
                suggestions[key] = default_map[key]
    
    mapping = {}
    
    st.sidebar.subheader("🗂️ Column Mapping")
    st.sidebar.caption("Map your CSV columns to required fields")
    
    # Required fields
    st.sidebar.markdown("**Required Fields:**")
    
    # Date column
    date_default = suggestions.get('date') or normalized_columns[0] if normalized_columns else None
    date_index = normalized_columns.index(date_default) if date_default in normalized_columns else 0
    mapping['date'] = st.sidebar.selectbox(
        "📅 Date Column *",
        options=normalized_columns,
        index=date_index if date_index < len(normalized_columns) else 0,
        key="map_date",
        help="Column containing date/week information"
    )
    
    # Product column
    product_default = suggestions.get('product') or normalized_columns[0] if normalized_columns else None
    product_index = normalized_columns.index(product_default) if product_default in normalized_columns else 0
    mapping['product'] = st.sidebar.selectbox(
        "🏷️ Product Column *",
        options=normalized_columns,
        index=product_index if product_index < len(normalized_columns) else 0,
        key="map_product",
        help="Column containing product names/IDs"
    )
    
    # Sales quantity column
    sales_default = suggestions.get('sales_qty') or normalized_columns[0] if normalized_columns else None
    sales_index = normalized_columns.index(sales_default) if sales_default in normalized_columns else 0
    mapping['sales_qty'] = st.sidebar.selectbox(
        "📊 Sales Quantity Column *",
        options=normalized_columns,
        index=sales_index if sales_index < len(normalized_columns) else 0,
        key="map_sales_qty",
        help="Column containing sales quantities"
    )
    
    # Optional fields
    st.sidebar.markdown("**Optional Fields:**")
    
    # Price column (optional)
    price_options = ["<None>"] + normalized_columns
    price_default = suggestions.get('price')
    price_index = 1 + normalized_columns.index(price_default) if price_default and price_default in normalized_columns else 0
    price_sel = st.sidebar.selectbox(
        "💰 Price Column",
        options=price_options,
        index=price_index if price_index < len(price_options) else 0,
        key="map_price",
        help="Column containing product prices (optional)"
    )
    mapping['price'] = None if price_sel == "<None>" else price_sel
    
    # Stock column (optional)
    stock_options = ["<None>"] + normalized_columns
    stock_default = suggestions.get('stock_on_hand')
    stock_index = 1 + normalized_columns.index(stock_default) if stock_default and stock_default in normalized_columns else 0
    stock_sel = st.sidebar.selectbox(
        "📦 Stock Column",
        options=stock_options,
        index=stock_index if stock_index < len(stock_options) else 0,
        key="map_stock",
        help="Column containing stock levels (optional)"
    )
    mapping['stock_on_hand'] = None if stock_sel == "<None>" else stock_sel
    
    # Category column (optional)
    category_options = ["<None>"] + normalized_columns
    category_default = suggestions.get('category')
    category_index = 1 + normalized_columns.index(category_default) if category_default and category_default in normalized_columns else 0
    category_sel = st.sidebar.selectbox(
        "📁 Category Column",
        options=category_options,
        index=category_index if category_index < len(category_options) else 0,
        key="map_category",
        help="Column containing product categories (optional)"
    )
    mapping['category'] = None if category_sel == "<None>" else category_sel
    
    # Store column (optional)
    store_options = ["<None>"] + normalized_columns
    store_default = suggestions.get('store')
    store_index = 1 + normalized_columns.index(store_default) if store_default and store_default in normalized_columns else 0
    store_sel = st.sidebar.selectbox(
        "🏪 Store Column",
        options=store_options,
        index=store_index if store_index < len(store_options) else 0,
        key="map_store",
        help="Column containing store/location IDs (optional)"
    )
    mapping['store'] = None if store_sel == "<None>" else store_sel
    
    # Apply Mapping button
    if st.sidebar.button("✅ Apply Mapping", use_container_width=True, key="apply_mapping"):
        # Validate required fields
        from utils.data_loader import validate_mapping
        raw_df = st.session_state.get('raw_df')
        if raw_df is not None:
            is_valid, missing = validate_mapping(mapping, raw_df)
            if is_valid:
                st.session_state['column_mapping'] = mapping
                st.sidebar.success("✅ Mapping saved!")
                # Clear mapped DataFrame to trigger regeneration
                st.session_state.pop('df_mapped', None)
                st.session_state.pop('last_mapping_applied', None)
            else:
                st.sidebar.error(f"❌ Missing required columns: {', '.join(missing)}")
        else:
            st.sidebar.error("❌ No data loaded. Please upload a CSV first.")
    
    return mapping


def apply_mapping_to_df(df: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    """
    Apply column mapping to DataFrame, renaming to canonical names.
    
    Args:
        df: Input DataFrame with original column names
        mapping: Dictionary mapping canonical names to original column names
    
    Returns:
        pd.DataFrame: DataFrame with canonical column names (date, product, sales_qty, etc.)
    """
    df = df.copy()
    
    # Create reverse mapping: canonical_name -> original_name
    rename_dict = {}
    for canonical, original in mapping.items():
        if original and original in df.columns:
            rename_dict[original] = canonical
    
    # Rename columns
    df = df.rename(columns=rename_dict)
    
    # Convert date column
    if 'date' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=False)
            # Check if >50% of dates failed to parse
            if df['date'].isna().sum() > len(df) * 0.5:
                # Try with dayfirst=True
                df['date'] = pd.to_datetime(df['date'], errors='coerce', dayfirst=True)
            if df['date'].isna().sum() > len(df) * 0.5:
                st.warning(
                    f"⚠️ Warning: {df['date'].isna().sum()} rows ({df['date'].isna().sum()/len(df)*100:.1f}%) "
                    f"could not be parsed as dates. Please check date format."
                )
        except Exception as e:
            st.warning(f"⚠️ Error parsing date column: {e}")
    
    # Convert sales_qty to numeric
    if 'sales_qty' in df.columns:
        try:
            df['sales_qty'] = pd.to_numeric(df['sales_qty'], errors='coerce')
            if df['sales_qty'].isna().sum() > 0:
                na_count = df['sales_qty'].isna().sum()
                df['sales_qty'] = df['sales_qty'].fillna(0)
                st.warning(f"⚠️ Warning: {na_count} non-numeric values in sales_qty converted to 0.")
        except Exception as e:
            st.warning(f"⚠️ Error converting sales_qty to numeric: {e}")
    
    # Convert price to numeric
    if 'price' in df.columns:
        try:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            if df['price'].isna().sum() > 0:
                na_count = df['price'].isna().sum()
                df['price'] = df['price'].fillna(0)
                st.warning(f"⚠️ Warning: {na_count} non-numeric values in price converted to 0.")
        except Exception as e:
            st.warning(f"⚠️ Error converting price to numeric: {e}")
    
    # Convert stock_on_hand to numeric if present
    if 'stock_on_hand' in df.columns:
        try:
            df['stock_on_hand'] = pd.to_numeric(df['stock_on_hand'], errors='coerce')
            if df['stock_on_hand'].isna().sum() > 0:
                na_count = df['stock_on_hand'].isna().sum()
                df['stock_on_hand'] = df['stock_on_hand'].fillna(0)
                st.warning(f"⚠️ Warning: {na_count} non-numeric values in stock_on_hand converted to 0.")
        except Exception as e:
            st.warning(f"⚠️ Error converting stock_on_hand to numeric: {e}")
    
    return df

