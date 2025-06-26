# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:44:18 2025

@author: KurtGrove
"""

import streamlit as st
import pandas as pd
from src.plotting import (create_box_whisker_plot_v2, 
                         create_ratios_box_whisker_plot,
                         create_risk_return_scatter, 
                         create_market_cap_animation)

st.set_page_config(page_title="SWS Peer Analysis", layout="wide")

# Add title and description
st.title("SWS Peer Analysis Dashboard")
st.markdown("""
This dashboard provides a comparative analysis of SWS SMA strategies against peer funds.
""")

# Add filters in sidebar
st.sidebar.header("Filters")

# Target Fund Selection
target_fund = st.sidebar.selectbox(
    'Target Fund',
    options=['SWS Growth Equity', 'SWS Dividend Equity'],
    index=0
)

#category filter
category = st.sidebar.selectbox(
    'Category Filter',
    options=['Large Growth','Large Blend','Large Growth and Large Blend','Large Value','All']
)

# Active Share Benchmark Selection
active_share_benchmark = st.sidebar.selectbox(
    'Active Share Benchmark',
    options=['Active Share RLG','Active Share R1000','Active Share RLV'],
    index=0
)
#active share filter
active_share_threshold = st.sidebar.slider(
    'Active Share Threshold', 0, 100, 50,10)

holdings_filter = st.sidebar.slider(
    'Maximum Holdings', 0, 1000, 1500,50)

market_cap_filter = st.sidebar.slider(
    'Maximum Wtd. Market Cap ($Bs)', 0, 1500, 2000,100)


# File uploader
uploaded_file = st.file_uploader(
    "Upload peer analysis data (CSV)",
    type="csv",
    help="Upload the peer analysis data file in CSV format"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)

        # Convert numeric columns
        numeric_columns = ['1YR', '3YR', '5YR', '7YR', 'SI Growth', 'SI Dividend', 'Fund AUM', 'Market Cap ($B) 2024', 'Sharpe', 'Sortino', 'Treynor', 'Upside Capture']
        for col in numeric_columns:
            if col not in df.columns:
                continue
                
            if col == 'Fund AUM':
                # Remove $ and commas, and any trailing spaces
                df[col] = df[col].astype(str).str.strip().str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col in ['1YR', '3YR', '5YR', '7YR', 'SI Growth', 'SI Dividend']:
                # Handle percentage columns - check if they're strings first
                if df[col].dtype == 'object':
                    # If string data, remove % sign and convert
                    df[col] = df[col].replace('', pd.NA).astype(str).str.rstrip('%')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Convert from percentage to decimal
                    df[col] = df[col] / 100
                else:
                    # If already numeric, just convert to float
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            elif col == 'Upside Capture':
                # Handle Upside Capture - convert % to ratio but don't divide by 100
                if df[col].dtype == 'object':
                    # If string data, remove % sign but keep as ratio (113% becomes 1.13)
                    df[col] = df[col].replace('', pd.NA).astype(str).str.rstrip('%')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Convert from percentage to ratio (113 becomes 1.13)
                    df[col] = df[col] / 100
                else:
                    # If already numeric, just convert to float
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                # For other numeric columns (ratios like Sharpe, Sortino, Treynor)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Select appropriate SI column based on target fund
        if target_fund == 'SWS Growth Equity':
            df['SI'] = df['SI Growth']
        else:
            df['SI'] = df['SI Dividend']

        # Apply filters
        if category == 'All':
            # Keep all three categories without additional filtering
            df = df[df['Morningstar Category'].isin(['Large Growth', 'Large Blend', 'Large Value'])]
        elif category == 'Large Growth and Large Blend':
            df = df[df['Morningstar Category'].isin(['Large Growth', 'Large Blend'])]
        elif category in ['Large Growth', 'Large Blend', 'Large Value']:
            df = df[df['Morningstar Category'] == category]
        
        if active_share_threshold > 0:
            df = df[df[active_share_benchmark] >= active_share_threshold]
        
        if holdings_filter < 2000:
            df = df[df['Total Holdings'] < holdings_filter]
        
        if market_cap_filter < 2000:
            df = df[df['Market Cap ($B) 2024'] < market_cap_filter]

        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "Performance Distribution", 
            "Risk-Return Analysis", 
            "Market Cap Animation",
            "Risk-Adjusted Ratios"
        ])
        
        # Performance Distribution Tab
        with tab1:
            st.plotly_chart(
                create_box_whisker_plot_v2(df, target_fund),
                use_container_width=True
            )
            
        # Risk-Return Analysis Tab
        with tab2:
            st.plotly_chart(
                create_risk_return_scatter(df, target_fund),
                use_container_width=True
            )
        
        # Market Cap Animation Tab
        with tab3:
            st.plotly_chart(
                create_market_cap_animation(df, target_fund),
                use_container_width=True
            )
            
        # Risk-Adjusted Ratios Tab
        with tab4:
            st.plotly_chart(
                create_ratios_box_whisker_plot(df, target_fund),
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error("Please ensure your CSV file has the required columns and format.")
else:
    # Show example format when no file is uploaded
    st.info("Please upload a CSV file to view the dashboard.")
    st.markdown("""
    **Required CSV Format:**
    ```
    Fund,Fund AUM,Market Cap ($B),1YR,3YR,5YR,7YR,SI Growth,SI Dividend,Morningstar Category
    ...
    ```
    """)