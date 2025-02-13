# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:44:18 2025

@author: KurtGrove
"""

import streamlit as st
import pandas as pd
from src.plotting import (create_box_whisker_plot, 
                         create_risk_return_scatter, 
                         create_market_cap_bubble)

st.set_page_config(page_title="SWS Peer Analysis", layout="wide")

# Add title and description
st.title("SWS Peer Analysis Dashboard")
st.markdown("""
This dashboard provides a comparative analysis of SWS Growth Equity against peer funds.
""")

# Add filters in sidebar
st.sidebar.header("Filters")


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



       # Convert numeric columns to float
        numeric_columns = ['Fund AUM', 'Market Cap ($B)', '1YR', '3YR', '5YR', 'SI']
        for col in numeric_columns:
            if col == 'Fund AUM':
                # Remove $ and commas, and any trailing spaces
                df[col] = df[col].str.strip().str.replace('$', '').str.replace(',', '').astype(float)
            elif col in ['1YR', '3YR', '5YR', 'SI']:
                # Remove % sign and convert to decimal
                df[col] = df[col].str.rstrip('%').astype(float) 
            else:
                # For other numeric columns
                df[col] = pd.to_numeric(df[col].str.replace(',', ''), errors='coerce')

            #If these are percentages in the CSV, convert to decimals
            if col in ['1YR', '3YR', '5YR', 'SI']:
                df[col] = df[col] / 100



        #Apply filters
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
            df = df[df['Market Cap ($B)'] < market_cap_filter]

        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs([
            "Performance Distribution", 
            "Risk-Return Analysis", 
            "Market Cap Distribution"
        ])
        
        # Performance Distribution Tab
        with tab1:
            st.plotly_chart(
                create_box_whisker_plot(df),
                use_container_width=True
            )
            
        # AUM Distribution Tab
        with tab2:
            st.plotly_chart(
                create_risk_return_scatter(df),
                use_container_width=True
            )
            
        # Market Cap Distribution Tab
        with tab3:
            st.plotly_chart(
                create_market_cap_bubble(df),
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
    Fund,Fund AUM,Market Cap ($B),1YR,3YR,5YR,SI,Morningstar Category
    Fund Name 1,5.2,320,0.15,0.45,0.62,0.75,Large Growth
    Fund Name 2,3.1,280,0.12,0.38,0.55,0.68,Large Growth
    ...
    ```
    """)