# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:47:08 2025

@author: KurtGrove
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np

def create_box_whisker_plot(df):
    """Create box and whisker plot using Plotly."""
    periods = ['1YR', '3YR', '5YR', 'SI']
    target_fund = 'SWS Growth Equity'

    # Debug print to check if the fund exists in the dataset
    print("available funds:", df['Fund'].unique())
    print("target fund:", target_fund in df['Fund'].values)
    
    fig = go.Figure()
    
    # Add box plots for each period
    for period in periods:
        # Calculate percentiles
        p05 = df[period].quantile(0.05)
        p25 = df[period].quantile(0.25)
        median = df[period].quantile(0.50)
        p75 = df[period].quantile(0.75)
        p95 = df[period].quantile(0.95)

        # Box plot trace
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
            showlegend=True if period == '1YR' else False
        ))
        
        # Add SWS Growth Equity callout box for each period
        target_data = df[df['Fund'] == target_fund]
        if not target_data.empty:
            target_return = target_data[period].iloc[0]
            rank = 100 - (df[period] <= target_return).mean() * 100
            
            fig.add_annotation(
                x=period,
                y=target_return,
                text=f"<b>SWS Growth Equity</b><br>Return: {target_return:.1%}<br>Rank: {rank:.0f}%ile",
                yshift=40,
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

            # Add marker for SWS Growth Equity
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



def create_risk_return_scatter(df):
    """Create risk-return scatter plot using Plotly."""
    target_fund = 'SWS Growth Equity'  # You might want to make this configurable
    
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
    if not target_data.empty:
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
# ... keep other existing functions ...

def create_market_cap_bubble(df):
    """Create market cap distribution chart using Plotly."""
    target_fund = 'SWS Growth Equity'
    
    # Create Market Cap buckets
    df['Market Cap Bucket'] = pd.cut(
        df['Market Cap ($B)'],
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
        y=grouped['Total AUM'] / 1000,  # Convert to trillions
        name='Total AUM',
        marker_color='lightblue',
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>" +
                      "Total AUM: $%{y:,.1f}T<br>" +
                      "Fund Count: %{customdata}<br>" +
                      "<extra></extra>",
        customdata=grouped['Fund Count']
    ))
    
    # Add target fund marker
    target_data = df[df['Fund'] == target_fund]
    if not target_data.empty:
        target_bucket = target_data['Market Cap Bucket'].iloc[0]
        target_market_cap = target_data['Market Cap ($B)'].iloc[0]
        bar_height = grouped[grouped['Market Cap Bucket'] == target_bucket]['Total AUM'].iloc[0]
        
        # Add marker at 20% of bar height
        marker_height = bar_height * 0.2 / 1000  # Convert to trillions
        
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
            text=f"<b>SWS Growth Equity</b><br>Market Cap: ${target_market_cap:.1f}B",
            yshift=40,
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
            title="Total AUM ($T)",
            tickformat=",.1f",
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