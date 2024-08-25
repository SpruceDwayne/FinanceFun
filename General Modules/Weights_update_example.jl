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
start_date = DateTime(2010, 8, 1)
end_date = DateTime(2020, 8, 20)
df =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date))
df = compute_relative_returns(df)
matrix_data = Matrix(df)
#matrix_data = hcat(matrix_data, 0.0*ones(size(matrix_data, 1)))

#We choose monthly 95% cvar to be leq 0.05 And arrive almost at the 60-40 portfolio
optimal_weights = optimize_portfolio_CVaR_const(-matrix_data[1:end,2:end],0.05,false,0.95)
#
#=
Now assume we start with an equal weighted portfolio
and we wish to rebalance to such that our returns are maximized and risk is below 0.05 like above.
=# 
n = size(matrix_data,2)-1
println(optimal_weights[1:n])

current_weights = [0.34,0.33,0.33,0,0,0]
new_weights, updates = update_portfolio_weights(current_weights,-matrix_data[1:end,2:end],0.05,9999,false,0.02,0.95)
#we now arrive at a quite different solution.
#Selling clearly hurts so in order to live with the loss incurred by selling we have to invest it into the safe bonds
println(new_weights[1:n]-new_weights[n+1 : 2*n])
#If we lower the trading costs we get closer and closer to the 60-40 portfolio
new_weights, updates = update_portfolio_weights(current_weights,-matrix_data[1:end,2:end],0.05,9999,false,0.01,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-matrix_data[1:end,2:end],0.05,9999,false,0.005,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
