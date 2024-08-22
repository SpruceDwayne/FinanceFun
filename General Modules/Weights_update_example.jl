include("CVaR_functions.jl")
include("finance_functions.jl")
using .CVaR_functions
using .finance_functions
using LinearAlgebra
using Statistics
using SparseArrays
using CSV
using TSFrames, MarketData, DataFrames, Dates,Plots

tickers = ["SPY","TLT","DBC"]
start_date = DateTime(2008, 8, 1)
end_date = DateTime(2024, 8, 20)
df =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date))
df = compute_relative_returns(df)
matrix_data = Matrix(df)
matrix_data = hcat(matrix_data, 0.0*ones(size(matrix_data, 1)))

#We choose monthly 95% cvar to be leq 0.1
optimal_weights = optimize_portfolio_CVaR_const(-matrix_data[1:end,2:end],0.1,false,0.95)
#And arrive almost at the 60-40 portfolio
#Now assume we start with an 50-40-10 portfolio and we wish to rebalance to such that our risk is below 10%, 
n = size(matrix_data,2)-1
println(optimal_weights[1:n])


current_weights = [0.4,0.2,0.1,0.3]
#Risk level below 10% trading costs of 1%, alpha = 0.95
new_weights, updates = update_portfolio_weights(current_weights,-matrix_data[1:end,2:end],0.1,9999,false,0.01,0.95)
#Then we have to sell almost all TLT and DBC but only but 10% of our total capital into SPY and have the last in cash
println(new_weights[1:n])
