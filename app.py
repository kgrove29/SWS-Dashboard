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
                         create_market_cap_animation,
                         create_clean_cumulative_returns_chart,
                         create_clean_rolling_returns_chart)

st.set_page_config(page_title="SWS Peer Analysis", layout="wide")

# Add title and description
st.title("SWS Peer Analysis Dashboard")
st.markdown("""
This dashboard provides a comparative analysis of SWS SMA strategies against peer funds.
""")

# Add logo to sidebar
st.sidebar.image("Logo Lockup.png", width=200)

# Add filters in sidebar
st.sidebar.header("Filters")

# Target Fund Selection
target_fund = st.sidebar.selectbox(
    'Target Fund',
    options=['SWS Growth Equity', 'SWS Dividend Equity'],
    index=0
)

# Category filter
category = st.sidebar.selectbox(
    'Category Filter',
    options=['Large Growth','Large Blend','Large Growth and Large Blend','Large Value','Large Blend and Large Value','All']
)

# Active Share Benchmark Selection
active_share_benchmark = st.sidebar.selectbox(
    'Active Share Benchmark',
    options=['Active Share RLG','Active Share R1000','Active Share RLV'],
    index=0
)

# Active share filter
active_share_threshold = st.sidebar.slider(
    'Active Share Threshold', 0, 100, 50, 10)

holdings_filter = st.sidebar.slider(
    'Maximum Holdings', 0, 1000, 1500, 50)

market_cap_filter = st.sidebar.slider(
    'Maximum Wtd. Market Cap ($Bs)', 0, 1500, 3000, 100)

# File uploader
uploaded_file = st.file_uploader(
    "Upload peer analysis data (Excel)",
    type=["xlsx", "xls"],
    help="Upload the peer analysis data file in Excel format"
)

if uploaded_file is not None:
    try:
        # Load the main data (first sheet by default)
        df = pd.read_excel(uploaded_file)

        # Convert numeric columns - dynamically detect market cap columns
        base_numeric_columns = ['1YR', '3YR', '5YR', '7YR', 'SI Growth', 'SI Dividend', 'Fund AUM', 'Sharpe', 'Sortino', 'Treynor', 'Upside Capture']
        
        # Find all market cap columns dynamically
        market_cap_columns = [col for col in df.columns if 'Market Cap ($B)' in str(col)]
        
        # Combine base columns with dynamically found market cap columns
        numeric_columns = base_numeric_columns + market_cap_columns
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

        # Store target fund data to ensure it's always included
        target_fund_data = df[df['Fund'] == target_fund].copy()
        
        # Store category selection for title generation
        category_for_title = category
        
        # Apply filters
        if category == 'All':
            # Keep all three categories without additional filtering
            df_filtered = df[df['Morningstar Category'].isin(['Large Growth', 'Large Blend', 'Large Value'])]
        elif category == 'Large Growth and Large Blend':
            df_filtered = df[df['Morningstar Category'].isin(['Large Growth', 'Large Blend'])]
        elif category == 'Large Blend and Large Value':
            df_filtered = df[df['Morningstar Category'].isin(['Large Blend', 'Large Value'])]
        elif category in ['Large Growth', 'Large Blend', 'Large Value']:
            df_filtered = df[df['Morningstar Category'] == category]
        
        # Always include target fund even if it doesn't match category filter
        if len(target_fund_data) > 0 and target_fund not in df_filtered['Fund'].values:
            df_filtered = pd.concat([df_filtered, target_fund_data], ignore_index=True)
        
        # Add category_for_title to dataframe for plotting functions to use
        df_filtered.attrs['category_for_title'] = category_for_title
        
        df = df_filtered
        
        if active_share_threshold > 0:
            df = df[df[active_share_benchmark] >= active_share_threshold]
        
        if holdings_filter < 2000:
            df = df[df['Total Holdings'] < holdings_filter]
        
        if market_cap_filter < 3000:
            df = df[df['Market Cap ($B) 2025'] < market_cap_filter]

        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Performance Distribution", 
            "Risk-Return Analysis", 
            "Market Cap Animation",
            "Risk-Adjusted Ratios",
            "Mountain Chart"
        ])
        
        # Performance Distribution Tab
        with tab1:
            st.plotly_chart(
                create_box_whisker_plot_v2(df, target_fund),
                use_container_width=True,
                key="performance_distribution_chart"
            )
            
        # Risk-Return Analysis Tab
        with tab2:
            # Add regression line toggle
            show_regression = st.checkbox(
                "Show Regression Line",
                value=False,
                help="Display a linear regression line showing the relationship between risk and return"
            )
            
            st.plotly_chart(
                create_risk_return_scatter(df, target_fund, show_regression=show_regression),
                use_container_width=True,
                key="risk_return_scatter_chart"
            )
        
        # Market Cap Animation Tab
        with tab3:
            st.plotly_chart(
                create_market_cap_animation(df, target_fund),
                use_container_width=True,
                key="market_cap_animation_chart"
            )
            
        # Risk-Adjusted Ratios Tab
        with tab4:
            st.plotly_chart(
                create_ratios_box_whisker_plot(df, target_fund),
                use_container_width=True,
                key="risk_adjusted_ratios_chart"
            )
            
        # Time Series Analysis Tab
        with tab5:
            st.subheader("Cumulative Returns")
            
            cumulative_chart = create_clean_cumulative_returns_chart(uploaded_file, target_fund)
            
            st.plotly_chart(
                cumulative_chart,
                use_container_width=True,
                key="cumulative_returns_chart"
            )
            
            # Rolling returns period selection
            col1, col2 = st.columns([1, 3])
            with col1:
                rolling_period_years = st.selectbox(
                    'Rolling Period',
                    options=[1, 2, 3],
                    index=0,
                    help="Select the rolling period duration"
                )
            
            st.subheader(f"Rolling {rolling_period_years}-Year Returns")
            
            # Convert years to months
            window_months = rolling_period_years * 12
            rolling_chart = create_clean_rolling_returns_chart(uploaded_file, target_fund, window_months=window_months)
            
            st.plotly_chart(
                rolling_chart,
                use_container_width=True,
                key="rolling_returns_chart"
            )
                        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error("Please ensure your Excel file has the required columns and format.")
else:
    # Show example format when no file is uploaded
    st.info("Please upload an Excel file to view the dashboard.")
    st.markdown("""
    **Required Excel Format:**
    The Excel file should contain:
    - **Main Sheet**: Fund performance data with columns like Fund, Fund AUM, Market Cap ($B), 1YR, 3YR, 5YR, 7YR, SI Growth, SI Dividend, Morningstar Category
    - **Monthly returns Sheet**: Dedicated worksheet with Fund names in column A and monthly returns data
    
    *Excel format preserves numeric precision better than CSV.*
    """)