# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:47:08 2025

@author: KurtGrove
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st

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

    # Update layout
    total_funds = len(df['Fund'].unique())
    categories = ', '.join(df['Morningstar Category'].unique())
    
    fig.update_layout(
        title={
            'text': f'Performance Distribution: {target_fund} vs {categories} Funds<br>' +
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

    # Update layout
    total_funds = len(df['Fund'].unique())
    categories = ', '.join(df['Morningstar Category'].unique())
    
    fig.update_layout(
        title={
            'text': f'Risk-Adjusted Ratios: {target_fund} vs {categories} Funds<br>' +
                   f'<sup>Analysis includes {total_funds} funds</sup>',
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
    
    return fig


def create_risk_return_scatter(df, target_fund):
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
        fig.add_trace(go.Scatter(
            x=target_data['Volatility'],
            y=target_data['SI'],
            mode='markers',
            name=target_fund,
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
    
    # Update layout
    total_funds = len(df['Fund'].unique())
    categories = ', '.join(df['Morningstar Category'].unique())
    
    fig.update_layout(
        title={
            'text': f'Risk-Return Analysis: {target_fund} vs {categories} Funds<br>' +
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
    
    # Add quadrant lines (optional)
    median_volatility = df['Volatility'].median()
    median_return = df['SI'].median()
    
    fig.add_hline(y=median_return, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=median_volatility, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_market_cap_bubble(df, target_fund):
    """Create market cap distribution chart using Plotly."""
    
    # Create Market Cap buckets
    df['Market Cap Bucket'] = pd.cut(
        df['Market Cap ($B) 2024'],
        bins=[0, 250, 500, 1000, float('inf')],
        labels=['$0-250B', '$250-500B', '$500-1T', '$1T+']
    )
    
    # Calculate metrics for each bucket
    grouped = df.groupby('Market Cap Bucket').agg({
        'Fund AUM': ['sum', 'count']
    }).reset_index()
    grouped.columns = ['Market Cap Bucket', 'Total AUM', 'Fund Count']
    
    fig = go.Figure()
    
    # Add bar chart for AUM
    fig.add_trace(go.Bar(
        x=grouped['Market Cap Bucket'],
        y=grouped['Total AUM'] / 1000000,  # Convert to billions
        name='Total AUM',
        marker_color='lightblue',
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>" +
                      "Total AUM: $%{y:,.0f}B<br>" +
                      "Fund Count: %{customdata}<br>" +
                      "<extra></extra>",
        customdata=grouped['Fund Count']
    ))
    
    # Add target fund marker
    target_data = df[df['Fund'] == target_fund].copy()
    if len(target_data) > 0:
        target_bucket = target_data['Market Cap Bucket'].iloc[0]
        target_market_cap = target_data['Market Cap ($B) 2024'].iloc[0]
        
        # Find the corresponding bar height
        bucket_data = grouped[grouped['Market Cap Bucket'] == target_bucket]
        if not bucket_data.empty:
            bar_height = bucket_data['Total AUM'].iloc[0]
            marker_height = bar_height * 0.2 / 1000000  # Convert to billions
            
            # Add marker
            fig.add_trace(go.Scatter(
                x=[target_bucket],
                y=[marker_height],
                mode='markers',
                name=target_fund,
                marker=dict(
                    color='red',
                    size=10,
                    symbol='diamond'
                ),
                showlegend=True
            ))
            
            # Add callout box
            fig.add_annotation(
                x=target_bucket,
                y=marker_height,
                text=f"<b>{target_fund}</b><br>Wtd. Avg. Market Cap: ${target_market_cap:.1f}B",
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
    
    # Update layout
    total_funds = len(df['Fund'].unique())
    categories = ', '.join(df['Morningstar Category'].unique())
    
    fig.update_layout(
        title={
            'text': f'Market Cap Distribution: {categories} Funds<br>' +
                   f'<sup>Analysis includes {total_funds} funds</sup>',
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Weighted Average Market Cap",
        yaxis=dict(
            title="Total AUM ($B)",
            tickformat=",.0f",
            gridcolor='lightgrey',
            zerolinecolor='lightgrey',
            zeroline=True
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
        categories = ', '.join(df['Morningstar Category'].unique())
        total_funds = len(df['Fund'].unique())
        
        # Calculate max y value for range
        max_y = max(max(frame.data[0].y) for frame in frames)
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Market Cap Distribution: {categories} Funds<br>' +
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