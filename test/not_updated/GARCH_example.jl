include("finance_functions.jl")
using .finance_functions
using Statistics
using TSFrames, MarketData, DataFrames, Dates,Plots
using ARCHModels
using Random

tickers = ["SPY","TLT","DBC"]
start_date = DateTime(2008, 8, 1)
end_date = DateTime(2024, 8, 20)
df =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date))
df = compute_relative_returns(df)
matrix_data = Array{Float64, 2}(Matrix(df)[:,2:end]) 

mean_vector = mean(matrix_data, dims=1)
demeaned_data = matrix_data .- mean_vector
n_assets = length(tickers)
m = fit(DCC{1,1}, demeaned_data)


predicted_covariance = predict(m, what=:covariance)
#predicted_covariance = predict(m, what=:correlation)


# Mean vector of the multivariate process (computed from your data or assumed)
mean_vector = vec(mean(matrix_data[:, 1:end], dims=1))

# Degrees of freedom for the t-distribution
ν = 3 
n_sim = 100
simulated_data = sample_mv_t(ν, mean_vector, predicted_covariance,n_sim)

