import os

# Change to Python Scripts directory
os.chdir(r'C:\Users\KurtGrove\Documents\Python Scripts')

# Create README content
readme_content = '''# Portfolio Optimizer for SWS Funds

This portfolio optimizer finds the optimal allocation between SWS Growth Equity and SWS Dividend Equity funds to maximize the Sharpe ratio.

## Features

- Uses actual monthly returns data for accurate optimization
- Fills missing SWS Dividend Equity data with r.1000v (Russell 1000 Value) returns
- Calculates optimal portfolio weights using scipy optimization
- Provides comprehensive performance analysis and reporting

## Files

- `portfolio_optimizer.py` - Main portfolio optimization script
- `requirements.txt` - Required Python packages
- `README.md` - This file

## Requirements

Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure the Excel file `streamlit example GE 6-30-2025.xlsx` is in the correct location:
   `C:\\Users\\KurtGrove\\Documents\\streamlit example GE 6-30-2025.xlsx`

2. Run the portfolio optimizer:
   ```bash
   python portfolio_optimizer.py
   ```

## Expected Output

The optimizer will:
1. Load fund data from the Excel file
2. Process monthly returns data
3. Fill missing SWS Dividend Equity data with r.1000v returns
4. Find optimal portfolio weights to maximize Sharpe ratio
5. Generate a comprehensive optimization report

## Data Requirements

The Excel file must contain:
- Main sheet with fund data including Sharpe ratios for SWS Growth and Dividend Equity
- 'Monthly returns' sheet with monthly returns data
- Data for 'SWS Growth Equity', 'SWS Dividend Equity', and 'r.1000v' funds

## Output

The optimization report includes:
- Optimal portfolio allocation (percentages)
- Expected annual return and volatility
- Sharpe ratio of optimal portfolio
- Comparison with individual fund performance
- Improvement analysis over single-fund strategies

## Example Results

```
================================================================================
PORTFOLIO OPTIMIZATION REPORT
================================================================================
Risk-Free Rate: 2.00%

OPTIMAL PORTFOLIO ALLOCATION:
  • SWS Growth Equity:    65.0%
  • SWS Dividend Equity:  35.0%

OPTIMAL PORTFOLIO METRICS:
  • Expected Annual Return: 12.50%
  • Annual Volatility:      18.75%
  • Sharpe Ratio:           0.560

COMPARISON WITH INDIVIDUAL FUNDS:
  • Growth Fund Sharpe:     0.545
  • Dividend Fund Sharpe:   0.480

IMPROVEMENT ANALYSIS:
  • vs Growth Fund:         0.015 (2.8%)
  • vs Dividend Fund:       0.080 (16.7%)
```

## Notes

- The optimizer uses actual monthly returns data for maximum accuracy
- Missing SWS Dividend Equity data points are filled with r.1000v returns
- The risk-free rate is set to 2% annually (adjustable in the code)
- Optimization uses scipy's minimize_scalar function with bounded method
'''

# Write the README file
with open('README.md', 'w') as f:
    f.write(readme_content)

print('README.md created successfully!') 