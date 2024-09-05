include("finance_functions.jl")
using .finance_functions
using LinearAlgebra
using Statistics
using SparseArrays
using CSV
using TSFrames, MarketData, DataFrames, Dates,Plots
using ARCHModels

tickers = ["SPY","TLT","DBC"]
start_date = DateTime(2010, 8, 1)
end_date = DateTime(2020, 8, 20)
df =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date))
df = compute_relative_returns(df)
matrix_data = Array{Float64, 2}(Matrix(df)[:,2:end]) 
m = fit(DCC{1,1,VolatilitySpec,1,1}, matrix_data)
