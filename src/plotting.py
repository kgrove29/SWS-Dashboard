# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:47:08 2025

@author: KurtGrove
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
import base64
from datetime import datetime

def get_logo_base64():
    """Convert logo to base64 for embedding in Plotly charts."""
    try:
        with open("Logo Lockup.png", "rb") as f:
            img_bytes = f.read()
        encoded = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{encoded}"
    except:
        return None

def add_custom_title_and_logo(fig, title_text, subtitle_text):
    """Add logo to chart that will be included in downloads."""
    logo_base64 = get_logo_base64()
    
    # Add logo positioned next to the built-in title
    if logo_base64:
        fig.add_layout_image(
            dict(
                source=logo_base64,
                xref="paper", yref="paper",
                x=0.01, y=1.10,  # Position higher up with title
                sizex=0.12, sizey=0.12,
                xanchor="left", yanchor="bottom",
                opacity=1.0,
                layer="above"
            )
        )
    
    return fig

def create_box_whisker_plot_v2(df, target_fund):
    """Create box and whisker plot using Plotly."""
    # Define base periods
    base_periods = ['1YR', '3YR', '5YR']
    
    # Add 7YR period only for SWS Growth Equity
    if target_fund == 'SWS Growth Equity':
        periods = base_periods + ['7YR', 'SI']
    else:
        periods = base_periods + ['SI']
    
    fig = go.Figure()
    
    # Add box plots for each period
    for period in periods:
        # Skip if period doesn't exist in dataframe
        if period not in df.columns:
            continue
            
        # Calculate percentiles and extremes
        p_min = df[period].min()
        p05 = df[period].quantile(0.05)
        p25 = df[period].quantile(0.25)
        median = df[period].quantile(0.50)
        p75 = df[period].quantile(0.75)
        p95 = df[period].quantile(0.95)
        p_max = df[period].max()

        # Box plot trace - simplified with no hover
        fig.add_trace(go.Box(
            x=[period],
            q1=[p25],
            median=[median],
            q3=[p75],
            lowerfence=[p05],
            upperfence=[p95],
            boxpoints=False,
            marker_color='rgb(176, 224, 230)',
            line_color='black',
            whiskerwidth=0.2,
            width=0.6,
            fillcolor='rgb(176, 224, 230)',
            orientation='v',
            name="Box 25th to 75th Percentile",
            showlegend=True if period == '1YR' else False,
            hoverinfo='skip'
        ))
        
        # Add invisible scatter point for custom hover at median position
        fig.add_trace(go.Scatter(
            x=[period],
            y=[median],
            mode='markers',
            marker=dict(
                color='rgba(0,0,0,0)',  # Invisible
                size=60,  # Large hit area
                line=dict(width=0)
            ),
            hovertemplate="<b>%{x}</b><br>" +
                          f"Maximum: {p_max:.2%}<br>" +
                          f"Upper Fence (95%): {p95:.2%}<br>" +
                          f"Q3 (75%): {p75:.2%}<br>" +
                          f"Median (50%): {median:.2%}<br>" +
                          f"Q1 (25%): {p25:.2%}<br>" +
                          f"Lower Fence (5%): {p05:.2%}<br>" +
                          f"Minimum: {p_min:.2%}<br>" +
                          "<extra></extra>",
            showlegend=False,
            name=""
        ))
        
        # Add target fund callout box for each period
        target_data = df[df['Fund'] == target_fund]
        if len(target_data) > 0:
            target_return = target_data[period].iloc[0]
            rank = 100 - (df[period].le(target_return).mean() * 100)
            
            fig.add_annotation(
                x=period,
                y=target_return + 0.05,
                text=f"<b>{target_fund}</b><br>Return: {target_return:.1%}<br>Rank: {rank:.0f}%ile",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black',
                bgcolor='black',
                bordercolor='black',
                borderwidth=2,
                borderpad=4,
                font=dict(color='white', size=10)
            )

            # Add marker for target fund
            fig.add_trace(go.Scatter(
                x=[period],
                y=[target_return],
                mode='markers',
                name=target_fund,
                marker=dict(
                    color='red',
                    size=10,
                    symbol='diamond'
                ),
                showlegend=True if period == '1YR' else False
            ))

    # Add whiskers to legend
    fig.add_trace(go.Scatter(
        x=['1YR', '1YR'],
        y=[df['1YR'].quantile(0.05), df['1YR'].quantile(0.95)],
        mode='lines',
        line=dict(color='black', width=.2),
        name="Whiskers (5%-95%)", 
        showlegend=True
    ))
    
    # Add median to legend
    fig.add_trace(go.Scatter(
        x=['1YR', '1YR'],
        y=[df['1YR'].quantile(0.50), df['1YR'].quantile(0.50)],
        mode='lines',
        line=dict(color='black', width=2),
        name="Median",
        showlegend=True
    ))

    # Update layout - RESTORE built-in title
    total_funds = len(df['Fund'].unique())
    
    # Use original category selection for title
    category_for_title = df.attrs.get('category_for_title', 'All')
    if category_for_title == 'All':
        categories_display = "All"
    elif category_for_title == 'Large Growth and Large Blend':
        categories_display = "Large Growth, Large Blend"
    elif category_for_title == 'Large Blend and Large Value':
        categories_display = "Large Blend, Large Value"
    else:
        categories_display = category_for_title
    
    fig.update_layout(
        title={
            'text': f'Performance Distribution: {target_fund} vs {categories_display} Funds<br>' +
                   f'<sup>Analysis includes {total_funds} funds</sup>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis=dict(
            title="Time Period",
            showticklabels=True,
            tickmode='array',
            ticktext=periods,
            tickvals=periods,
            type='category'
        ),
        yaxis=dict(
            title="Return",
            tickformat='.0%',
            gridcolor='lightgrey',
            zerolinecolor='lightgrey',
            zeroline=True,
            autorange=True
        ),
        showlegend=True,
        height=600,
        template='plotly_white',
        legend=dict(
            x=1,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add horizontal grid lines only
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    
    # Add logo next to title
    title_text = f'Performance Distribution: {target_fund} vs {categories_display} Funds'
    subtitle_text = f'Analysis includes {total_funds} funds'
    fig = add_custom_title_and_logo(fig, title_text, subtitle_text)
    
    return fig


def create_ratios_box_whisker_plot(df, target_fund):
    """Create box and whisker plot for financial ratios using Plotly with separate subplots."""
    # Define ratios to compare
    ratios = ['Sharpe', 'Sortino', 'Treynor', 'Upside Capture']
    
    from plotly.subplots import make_subplots
    
    # Create subplots with separate y-axes
    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=ratios,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.08
    )
    
    # Add box plots for each ratio
    for i, ratio in enumerate(ratios, 1):
        # Skip if ratio doesn't exist in dataframe
        if ratio not in df.columns:
            continue
            
        # Calculate percentiles and extremes
        p_min = df[ratio].min()
        p05 = df[ratio].quantile(0.05)
        p25 = df[ratio].quantile(0.25)
        median = df[ratio].quantile(0.50)
        p75 = df[ratio].quantile(0.75)
        p95 = df[ratio].quantile(0.95)
        p_max = df[ratio].max()

        # Box plot trace - simplified with no hover
        fig.add_trace(go.Box(
            x=[ratio],
            q1=[p25],
            median=[median],
            q3=[p75],
            lowerfence=[p05],
            upperfence=[p95],
            boxpoints=False,
            marker_color='rgb(176, 224, 230)',
            line_color='black',
            whiskerwidth=0.2,
            width=0.6,
            fillcolor='rgb(176, 224, 230)',
            orientation='v',
            name="Box 25th to 75th Percentile",
            showlegend=True if ratio == 'Sharpe' else False,
            hoverinfo='skip'
        ), row=1, col=i)
        
        # All ratios should be formatted as decimal numbers
        # Upside Capture as 1.13 (meaning 113% capture), not as percentage
        format_str = ".2f"
        
        # Add invisible scatter point for custom hover at median position
        fig.add_trace(go.Scatter(
            x=[ratio],
            y=[median],
            mode='markers',
            marker=dict(
                color='rgba(0,0,0,0)',  # Invisible
                size=60,  # Large hit area
                line=dict(width=0)
            ),
            hovertemplate="<b>%{x}</b><br>" +
                          f"Maximum: {p_max:{format_str}}<br>" +
                          f"Upper Fence (95%): {p95:{format_str}}<br>" +
                          f"Q3 (75%): {p75:{format_str}}<br>" +
                          f"Median (50%): {median:{format_str}}<br>" +
                          f"Q1 (25%): {p25:{format_str}}<br>" +
                          f"Lower Fence (5%): {p05:{format_str}}<br>" +
                          f"Minimum: {p_min:{format_str}}<br>" +
                          "<extra></extra>",
            showlegend=False,
            name=""
        ), row=1, col=i)
        
        # Add target fund callout box for each ratio
        target_data = df[df['Fund'] == target_fund]
        if len(target_data) > 0:
            target_value = target_data[ratio].iloc[0]
            rank = 100 - (df[ratio].le(target_value).mean() * 100)
            
            fig.add_annotation(
                x=ratio,
                y=target_value + (p_max - p_min) * 0.05,  # Offset based on range
                text=f"<b>{target_fund}</b><br>{ratio}: {target_value:{format_str}}<br>Rank: {rank:.0f}%ile",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black',
                bgcolor='black',
                bordercolor='black',
                borderwidth=2,
                borderpad=4,
                font=dict(color='white', size=10),
                xref=f"x{i}",
                yref=f"y{i}"
            )

            # Add marker for target fund
            fig.add_trace(go.Scatter(
                x=[ratio],
                y=[target_value],
                mode='markers',
                name=target_fund,
                marker=dict(
                    color='red',
                    size=10,
                    symbol='diamond'
                ),
                showlegend=True if ratio == 'Sharpe' else False
            ), row=1, col=i)

    # Add whiskers to legend (on first subplot)
    fig.add_trace(go.Scatter(
        x=['Sharpe', 'Sharpe'],
        y=[df['Sharpe'].quantile(0.05), df['Sharpe'].quantile(0.95)],
        mode='lines',
        line=dict(color='black', width=.2),
        name="Whiskers (5%-95%)", 
        showlegend=True
    ), row=1, col=1)
    
    # Add median to legend (on first subplot)
    fig.add_trace(go.Scatter(
        x=['Sharpe', 'Sharpe'],
        y=[df['Sharpe'].quantile(0.50), df['Sharpe'].quantile(0.50)],
        mode='lines',
        line=dict(color='black', width=2),
        name="Median",
        showlegend=True
    ), row=1, col=1)

    # Update layout - RESTORE built-in title
    total_funds = len(df['Fund'].unique())
    
    # Use original category selection for title
    category_for_title = df.attrs.get('category_for_title', 'All')
    if category_for_title == 'All':
        categories_display = "All"
    elif category_for_title == 'Large Growth and Large Blend':
        categories_display = "Large Growth, Large Blend"
    elif category_for_title == 'Large Blend and Large Value':
        categories_display = "Large Blend, Large Value"
    else:
        categories_display = category_for_title
    
    fig.update_layout(
        title={
            'text': f'Risk-Adjusted Ratios: {target_fund} vs {categories_display} Funds<br>' +
                   f'<sup>Analysis includes {total_funds} funds since inception</sup>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        height=600,
        template='plotly_white',
        legend=dict(
            x=1,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Update individual subplot axes
    for i in range(1, 5):
        fig.update_xaxes(
            showgrid=False,
            showticklabels=False,  # Hide x-axis labels since we have subplot titles
            row=1, col=i
        )
        fig.update_yaxes(
            showgrid=True, 
            gridwidth=1, 
            gridcolor='lightgrey',
            title_text="Ratio Value" if i == 1 else "",  # Only show y-axis title on first subplot
            row=1, col=i
        )
    
    # Add logo next to title
    title_text = f'Risk-Adjusted Ratios: {target_fund} vs {categories_display} Funds'
    subtitle_text = f'Analysis includes {total_funds} funds'
    fig = add_custom_title_and_logo(fig, title_text, subtitle_text)
    
    return fig


def create_risk_return_scatter(df, target_fund, show_regression=False):
    """Create risk-return scatter plot using Plotly."""
    
    # Split data into target fund and peers
    target_data = df[df['Fund'] == target_fund]
    peer_data = df[df['Fund'] != target_fund]
    
    fig = go.Figure()
    
    # Add peer funds scatter
    fig.add_trace(go.Scatter(
        x=peer_data['Volatility'],
        y=peer_data['SI'],
        mode='markers',
        name='Peer Funds',
        marker=dict(
            color='lightgray',
            size=8,
            opacity=0.6
        ),
        hovertemplate="<b>%{customdata}</b><br>" +
                      "Return: %{y:.1%}<br>" +
                      "Volatility: %{x:.1%}<br>" +
                      "<extra></extra>",
        customdata=peer_data['Fund']
    ))
    
    # Add target fund marker
    if len(target_data) > 0:
        target_return = target_data['SI'].iloc[0]
        target_vol = target_data['Volatility'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=target_data['Volatility'],
            y=target_data['SI'],
            mode='markers',
            name=f"{target_fund} (Return: {target_return:.1%}, Vol: {target_vol:.1%})",
            marker=dict(
                color='red',
                size=12,
                symbol='diamond'
            ),
            hovertemplate="<b>" + target_fund + "</b><br>" +
                          "Return: %{y:.1%}<br>" +
                          "Volatility: %{x:.1%}<br>" +
                          "<extra></extra>"
        ))
    
    # Update layout - RESTORE built-in title
    total_funds = len(df['Fund'].unique())
    
    # Use original category selection for title
    category_for_title = df.attrs.get('category_for_title', 'All')
    if category_for_title == 'All':
        categories_display = "All"
    elif category_for_title == 'Large Growth and Large Blend':
        categories_display = "Large Growth, Large Blend"
    elif category_for_title == 'Large Blend and Large Value':
        categories_display = "Large Blend, Large Value"
    else:
        categories_display = category_for_title
    
    fig.update_layout(
        title={
            'text': f'Risk-Return Analysis: {target_fund} vs {categories_display} Funds<br>' +
                   f'<sup>Analysis includes {total_funds} funds</sup>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Risk (Volatility)",
        yaxis_title="Return (Since Inception)",
        xaxis=dict(
            tickformat='.0%',
            range=[.15, 0.40]  # Set volatility limit to 40%
        ),
        yaxis=dict(
            tickformat='.0%',
            range=[0.05, .25]  # Set return lower limit to 5%
        ),
        showlegend=True,
        height=600,
        hovermode='closest',
        template='plotly_white'
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    
    # Calculate and add regression line (if enabled)
    if show_regression:
        valid_data = df.dropna(subset=['Volatility', 'SI'])
        if len(valid_data) > 1:
            x_vals = valid_data['Volatility'].values
            y_vals = valid_data['SI'].values
            
            # Calculate regression coefficients
            coeffs = np.polyfit(x_vals, y_vals, 1)
            slope, intercept = coeffs
            
            # Create regression line points
            x_min, x_max = x_vals.min(), x_vals.max()
            x_regression = np.linspace(x_min, x_max, 100)
            y_regression = slope * x_regression + intercept
            
            # Add regression line
            fig.add_trace(go.Scatter(
                x=x_regression,
                y=y_regression,
                mode='lines',
                name=f"Regression Line (R² = {np.corrcoef(x_vals, y_vals)[0,1]**2:.3f})",
                line=dict(color='lightgrey', width=2, dash='dot'),
                showlegend=True
            ))
    
    # Add quadrant lines with legend entries
    median_volatility = df['Volatility'].median()
    median_return = df['SI'].median()
    
    fig.add_hline(y=median_return, line_dash="dash", line_color="darkgray", line_width=2, opacity=0.8)
    fig.add_vline(x=median_volatility, line_dash="dash", line_color="darkgray", line_width=2, opacity=0.8)
    
    # Add invisible trace for median lines legend entry
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color='darkgray', width=2, dash='dash'),
        name=f"Median Lines (Return: {median_return:.1%}, Vol: {median_volatility:.1%})",
        showlegend=True
    ))
    
    # Add logo next to title
    title_text = f'Risk-Return Analysis: {target_fund} vs {categories_display} Funds'
    subtitle_text = f'Analysis includes {total_funds} funds'
    fig = add_custom_title_and_logo(fig, title_text, subtitle_text)
    
    return fig

def create_market_cap_animation(df, target_fund):
    """Create animated market cap distribution chart using Plotly."""
    try:
        # First ensure we have the required columns
        required_cols = ['Fund', 'Fund AUM', 'Morningstar Category']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
            
        # Find market cap columns
        market_cap_cols = []
        for col in df.columns:
            if isinstance(col, str) and 'Market Cap ($B)' in col:
                try:
                    year = int(col.split()[-1])
                    market_cap_cols.append((col, year))
                except (ValueError, IndexError):
                    continue
        
        if not market_cap_cols:
            raise ValueError("No valid market cap columns found")
            
        # Sort by year
        market_cap_cols.sort(key=lambda x: x[1])
        years = [year for _, year in market_cap_cols]
        
        # Create figure
        fig = go.Figure()
        frames = []
        
        # Create market cap buckets
        market_cap_bins = [0, 250, 500, 1000, float('inf')]
        market_cap_labels = ['$0-250B', '$250-500B', '$500-1T', '>$1T']
        
        # Process each year
        for col, year in market_cap_cols:
            # Ensure numeric data
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            df['Fund AUM'] = pd.to_numeric(df['Fund AUM'].astype(str).str.replace('$', '').str.replace(',', ''), errors='coerce')
            
            # Create buckets
            df['temp_bucket'] = pd.cut(
                df[col],
                bins=market_cap_bins,
                labels=market_cap_labels,
                include_lowest=True
            )
            
            # Group data
            grouped = df.groupby('temp_bucket', observed=True).agg({
                'Fund AUM': 'sum',
                'Fund': 'count'
            }).reset_index()
            
            # Create frame data with annotations for each year
            frame_data = [go.Bar(
                x=market_cap_labels,
                y=[grouped[grouped['temp_bucket'] == label]['Fund AUM'].sum() / 1e9 
                   if label in grouped['temp_bucket'].values else 0 
                   for label in market_cap_labels],
                text=[f"Total Funds: {len(df[df['temp_bucket'] == label])}<br>${grouped[grouped['temp_bucket'] == label]['Fund AUM'].sum()/1e9:,.0f}B" 
                      if label in grouped['temp_bucket'].values else "n=0<br>$0B"
                      for label in market_cap_labels],
                textposition='auto',
                marker_color='lightblue',
                opacity=0.7,
                name='Total AUM'
            )]
            
            # Add target fund marker if present
            if target_fund in df['Fund'].values:
                target_data = df[df['Fund'] == target_fund]
                target_bucket = pd.cut(
                    [target_data[col].iloc[0]], 
                    bins=market_cap_bins, 
                    labels=market_cap_labels
                )[0]
                
                # Calculate marker height - fixed calculation
                bucket_data = grouped[grouped['temp_bucket'] == target_bucket]
                if not bucket_data.empty:
                    bar_height = float(bucket_data['Fund AUM'].iloc[0]) / 1e9
                    marker_height = float(bar_height * 1.1)
                    
                    # Create frame with annotation
                    frame = go.Frame(
                        data=frame_data,
                        name=str(year),
                        layout=dict(
                            annotations=[dict(
                                x=target_bucket,
                                y=marker_height,
                                text=f"<b>{target_fund}</b><br>Wtd. Market Cap: ${target_data[col].iloc[0]:.1f}B",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor='black',
                                bgcolor='black',
                                bordercolor='black',
                                borderwidth=2,
                                borderpad=4,
                                font=dict(color='white', size=10),
                                yshift=20
                            )]
                        )
                    )
            if 'frame' not in locals():
                frame = go.Frame(data=frame_data, name=str(year))
            
            frames.append(frame)
            
        # Add initial data
        if frames:
            for trace in frames[0].data:
                fig.add_trace(trace)
            
            # Add initial annotation if target fund exists
            if target_fund in df['Fund'].values:
                first_col = market_cap_cols[0][0]
                target_data = df[df['Fund'] == target_fund]
                target_bucket = pd.cut(
                    [target_data[first_col].iloc[0]], 
                    bins=market_cap_bins, 
                    labels=market_cap_labels
                )[0]
                
                # Calculate initial marker height
                first_grouped = df.groupby('temp_bucket', observed=True).agg({
                    'Fund AUM': 'sum',
                    'Fund': 'count'
                }).reset_index()
                
                bucket_data = first_grouped[first_grouped['temp_bucket'] == target_bucket]
                if not bucket_data.empty:
                    bar_height = float(bucket_data['Fund AUM'].iloc[0]) / 1e9
                    marker_height = float(bar_height * 1.1)
                    
                    fig.update_layout(
                        annotations=[dict(
                            x=target_bucket,
                            y=marker_height,
                            text=f"<b>{target_fund}</b><br>Market Cap: ${target_data[first_col].iloc[0]:.1f}B",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor='black',
                            bgcolor='black',
                            bordercolor='black',
                            borderwidth=2,
                            borderpad=4,
                            font=dict(color='white', size=10),
                            yshift=20
                        )]
                    )
        
        # Get categories and fund count for title
        # Use original category selection for title
        category_for_title = df.attrs.get('category_for_title', 'All')
        if category_for_title == 'All':
            categories_display = "All"
        elif category_for_title == 'Large Growth and Large Blend':
            categories_display = "Large Growth, Large Blend"
        elif category_for_title == 'Large Blend and Large Value':
            categories_display = "Large Blend, Large Value"
        else:
            categories_display = category_for_title
        
        total_funds = len(df['Fund'].unique())
        
        # Calculate max y value for range
        max_y = max(max(frame.data[0].y) for frame in frames)
        
        # Update layout - RESTORE built-in title
        fig.update_layout(
            title={
                'text': f'Market Cap Distribution: {categories_display} Funds<br>' +
                       f'<sup>Analysis includes {total_funds} funds</sup>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Market Cap Range",
            yaxis_title="Total AUM ($B)",
            yaxis=dict(
                tickformat="$,.0f",
                gridcolor='lightgrey',
                zerolinecolor='lightgrey',
                zeroline=True,
                range=[0, max_y * 1.2]
            ),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            showlegend=True,
            legend=dict(
                x=1,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            sliders=[{
                'currentvalue': {'prefix': 'Year: '},
                'steps': [{'args': [[str(year)], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }], 
                          'label': str(year),
                          'method': 'animate'} for year in years],
                'x': 0.0,
                'len': 0.9,
               
            }]
        )
        
        # Add horizontal grid lines only
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        
        # Add logo next to title
        title_text = f'Market Cap Distribution: {categories_display} Funds'
        subtitle_text = f'Analysis includes {total_funds} funds'
        fig = add_custom_title_and_logo(fig, title_text, subtitle_text)
        
        fig.frames = frames
        return fig

    except Exception as e:
        st.error(f"Error in market cap animation: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text="Could not create animation - check data format",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )
        return fig

def create_clean_cumulative_returns_chart(uploaded_file, target_fund):
    """
    Create cumulative returns chart using the dedicated Monthly returns worksheet.
    
    This is a clean implementation that expects:
    - Monthly returns in a separate worksheet named 'Monthly returns'
    - Fund names in column 0
    - Monthly returns in datetime-named columns (already in decimal format)
    """
    try:
        # Read the Monthly returns sheet directly
        monthly_df = pd.read_excel(uploaded_file, sheet_name='Monthly returns')
        
        # Find the target fund
        target_data = monthly_df[monthly_df['Fund'] == target_fund]
        if target_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Fund '{target_fund}' not found in Monthly returns sheet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        target_row = target_data.iloc[0]
        
        # Get all date columns (everything except 'Fund')
        date_columns = [col for col in monthly_df.columns if col != 'Fund']
        
        # Extract monthly returns and dates
        returns = []
        dates = []
        
        for date_col in date_columns:
            monthly_return = target_row[date_col]
            
            # Skip NaN values (fund hadn't started yet)
            if pd.notna(monthly_return):
                returns.append(float(monthly_return))
                dates.append(date_col)
        
        if not returns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No valid returns found for {target_fund}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate cumulative returns: (1 + r1) * (1 + r2) * ... - 1
        cumulative_values = [1.0]  # Start at $1.00
        running_value = 1.0
        
        for monthly_return in returns:
            running_value = running_value * (1.0 + monthly_return)
            cumulative_values.append(running_value)
        
        # Convert to percentage returns for display
        cumulative_returns = [(value - 1.0) for value in cumulative_values]
        
        # Create date series for chart - start at beginning of inception month, then use month-end dates
        inception_month_start = datetime(dates[0].year, dates[0].month, 1)
        chart_dates = [inception_month_start] + dates
        
        final_return = cumulative_values[-1] - 1.0
        
        # Create the chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=chart_dates,
            y=cumulative_returns,
            mode='lines+markers',
            name=target_fund,
            line=dict(width=3, color='red'),
            marker=dict(size=4, color='red'),
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Date: %{x}<br>" +
                         "Cumulative Return: %{y:.1%}<br>" +
                         "<extra></extra>"
        ))
        
        # Update layout
        inception_display = inception_month_start.strftime('%B %Y')
        
        fig.update_layout(
            title={
                'text': f'Cumulative Returns: {target_fund}<br>' +
                       f'<sup>Since {inception_display} | Total Return: {final_return:.1%}</sup>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            yaxis=dict(
                tickformat='.1%',
                gridcolor='lightgrey',
                zerolinecolor='lightgrey',
                zeroline=True
            ),
            xaxis=dict(
                gridcolor='lightgrey',
                showgrid=True
            ),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            showlegend=True
        )
        
        # Add logo next to title
        title_text = f'Cumulative Returns: {target_fund}'
        subtitle_text = f'Since {inception_display}'
        fig = add_custom_title_and_logo(fig, title_text, subtitle_text)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

def create_clean_rolling_returns_chart(uploaded_file, target_fund, window_months=12):
    """
    Create rolling returns chart using the dedicated Monthly returns worksheet.
    
    This is a clean implementation that calculates annualized rolling returns
    over the specified window (default 12 months).
    """
    try:
        # Read the Monthly returns sheet directly
        monthly_df = pd.read_excel(uploaded_file, sheet_name='Monthly returns')
        
        # Find the target fund
        target_data = monthly_df[monthly_df['Fund'] == target_fund]
        if target_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Fund '{target_fund}' not found in Monthly returns sheet",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        target_row = target_data.iloc[0]
        
        # Get all date columns (everything except 'Fund')
        date_columns = [col for col in monthly_df.columns if col != 'Fund']
        
        # Extract monthly returns and dates (skip NaN values)
        returns = []
        dates = []
        
        for date_col in date_columns:
            monthly_return = target_row[date_col]
            if pd.notna(monthly_return):
                returns.append(float(monthly_return))
                dates.append(date_col)
        
        if len(returns) < window_months:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Insufficient data for {window_months}-month rolling returns",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Calculate rolling annualized returns
        rolling_returns = []
        rolling_dates = []
        
        for i in range(window_months - 1, len(returns)):
            # Get window of returns
            window_returns = returns[i - window_months + 1:i + 1]
            
            # Calculate compound return for the window
            compound_return = 1.0
            for monthly_return in window_returns:
                compound_return = compound_return * (1.0 + monthly_return)
            
            # Annualize the return
            annualized_return = compound_return ** (12.0 / window_months) - 1.0
            
            rolling_returns.append(annualized_return)
            rolling_dates.append(dates[i])
        
        # Create the chart
        fig = go.Figure()
        
        # Add prominent zero line first
        fig.add_hline(
            y=0, 
            line_dash="solid", 
            line_color="black", 
            line_width=2,
            opacity=1.0
        )
        
        # Add invisible zero line trace for fill reference
        fig.add_trace(go.Scatter(
            x=rolling_dates,
            y=[0] * len(rolling_dates),
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Create arrays for positive and negative fills
        positive_returns = [max(0, ret) for ret in rolling_returns]
        negative_returns = [min(0, ret) for ret in rolling_returns]
        
        # Add positive fill (green)
        fig.add_trace(go.Scatter(
            x=rolling_dates,
            y=positive_returns,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(46, 204, 113, 0.3)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add negative fill (red) 
        fig.add_trace(go.Scatter(
            x=rolling_dates,
            y=negative_returns,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 182, 193, 0.3)',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add the main data line on top
        fig.add_trace(go.Scatter(
            x=rolling_dates,
            y=rolling_returns,
            mode='lines+markers',
            name=target_fund,
            line=dict(width=3, color='black'),
            marker=dict(size=4, color='black'),
            hovertemplate="<b>%{fullData.name}</b><br>" +
                         "Date: %{x}<br>" +
                         f"{window_months}-Month Annualized Return: %{{y:.1%}}<br>" +
                         "<extra></extra>"
        ))
        
        # Calculate percentage of time with positive returns
        positive_periods = sum(1 for ret in rolling_returns if ret > 0)
        total_periods = len(rolling_returns)
        positive_percentage = (positive_periods / total_periods) * 100 if total_periods > 0 else 0
        
        # Find min and max values for callouts
        if rolling_returns:
            min_return = min(rolling_returns)
            max_return = max(rolling_returns)
            min_index = rolling_returns.index(min_return)
            max_index = rolling_returns.index(max_return)
            min_date = rolling_dates[min_index]
            max_date = rolling_dates[max_index]
        
        # Update layout
        inception_display = dates[0].strftime('%B %Y')
        period_years = window_months / 12
        
        fig.update_layout(
            title={
                'text': f'{window_months}-Month Rolling Returns: {target_fund}<br>' +
                       f'<sup>Annualized rolling performance since {inception_display} | ' +
                       f'{positive_percentage:.1f}% of {period_years:.0f}-year rolling periods with positive returns</sup>',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Date",
            yaxis_title=f"{window_months}-Month Annualized Return",
            yaxis=dict(
                tickformat='.1%',
                gridcolor='lightgrey',
                zerolinecolor='lightgrey',
                zeroline=True
            ),
            xaxis=dict(
                gridcolor='lightgrey',
                showgrid=True
            ),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            showlegend=True
        )
        
        # Add min and max callouts
        if rolling_returns:
            # Add max callout (green)
            fig.add_annotation(
                x=max_date,
                y=max_return,
                text=f"<b>Max- {period_years:.0f} year annualized return</b><br>{max_return:.1%}<br>{max_date.strftime('%b %Y')}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='green',
                bgcolor='green',
                bordercolor='green',
                borderwidth=2,
                borderpad=4,
                font=dict(color='white', size=9),
                yshift=15
            )
            
            # Add min callout (red)
            fig.add_annotation(
                x=min_date,
                y=min_return,
                text=f"<b>Min- {period_years:.0f} year annualized return</b><br>{min_return:.1%}<br>{min_date.strftime('%b %Y')}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red',
                bgcolor='red',
                bordercolor='red',
                borderwidth=2,
                borderpad=4,
                font=dict(color='white', size=9),
                yshift=60
            )
        
        # Add logo next to title
        title_text = f'{window_months}-Month Rolling Returns: {target_fund}'
        subtitle_text = f'Annualized rolling performance since {inception_display}'
        fig = add_custom_title_and_logo(fig, title_text, subtitle_text)
        
        return fig
        
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig