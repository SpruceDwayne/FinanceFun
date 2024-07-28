#=This script will have functions directly relevant with finance stuff like
1. getting stock data
2. mapping to log returns
3. mapping back from log returns to real returns
4. functions related to pricing
5. functions doing portfolio optimization
6. functions related to backtesting
7. Intra day data hopefully
=#

using AlphaVantage
using DataFrames
using DataFramesMeta
using Dates
using Plots
#APIkey is  VWLXYQW5UFLW7OQX