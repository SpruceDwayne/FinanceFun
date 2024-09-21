include("finance_functions.jl")
include("CVaR_functions.jl")

using .finance_functions
using .CVaR_functions
using DataFrames
using Dates
using Statistics
using ARCHModels
using CSV

# Parameters
tickers = [ "TLT","SHY","IEF","SPY"]
start_date = DateTime(2010, 3, 1)
end_date = DateTime(2024, 8, 20)
window_size = 1000
n_sim = 10000
ν = 3  # Degrees of freedom for the t-distribution

df =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date,"1d"))
df = compute_relative_returns(df)

matrix_data = Array{Float64, 2}(Matrix(df)[:,2:end]) 


#df = dropmissing(log_returns[2:end,1:5])

matrix_data = Array{Float64, 2}(Matrix(df)[:, 2:end])
#matrix_data = hcat(matrix_data, 0.00001*ones(size(matrix_data, 1)))
#matrix_data = hcat(matrix_data, 0.05 .+ (0.00002 - 0.00001) .* rand(size(matrix_data, 1)))



# Initialize variables
num_assets = size(matrix_data, 2)
num_rows = size(matrix_data, 1)
all_weights = zeros( num_rows - window_size + 1, 2*num_assets) 
all_weights[1, :] = [0.1333333333333,0.1333333,0.1333333,0.6, 0, 0, 0, 0]

#all_weights[1, :] = [0.1333333333333,0.1333333,0.1333333,0.6, 0, 0, 0, 0, 0,0]


# Iteratively fit the GARCH model and update portfolio weights
for i in window_size: window_size+200#num_rows
    # Define the window of data
    window_data = matrix_data[i-window_size+1:i, :]

    # Demean the data
    mean_vector = 0 #mean(window_data, dims=1)
    demeaned_data = window_data .- mean_vector
    mean_vector = vec(mean(matrix_data[1:i-window_size, 1:end], dims=1))
    # Fit the GARCH model
    m = fit(DCC{1,1}, demeaned_data)
    #predicted_covariance = predict(m, what=:covariance)
    predicted_covariance = cov(matrix_data)  # Replace with your default covariance matrix
    try #Since the predict sometimes fail
        predicted_covariance = predict(m, what=:covariance)
    catch e
        println("Prediction failed, using default covariance matrix.")
    end
    # Sample from corresponding multivariate t-distribution
    simulated_data = transpose(finance_functions.sample_mv_t(ν, mean_vector, predicted_covariance, n_sim))
    #simulated_data .+=mean(window_data, dims=1)
    simulated_data = -window_data
    risk_target = sum(all_weights[i+1-window_size,1:num_assets])*0.05
    # Update portfolio weights
    #local new_weights, updates = CVaR_functions.update_portfolio_weights(all_weights[i+1-window_size,1:end], -window_data[1:end, 1:end], risk_target, 9999, false, 0.001, 0.95)
    new_weights = CVaR_functions.optimize_portfolio_CVaR_const(-window_data,risk_target,0.95,false)[1:2*num_assets]
    println(new_weights)
    #println(updates[1:2*num_assets])
    if i < num_rows
        latest_return = matrix_data[i+1, :]  # Next period's return
        temp =new_weights[1:num_assets]-new_weights[num_assets+1:2*num_assets]
        #temp .*= (1 .+ latest_return)
        temp =temp/sum(temp)
        all_weights[i-window_size+2, 1:num_assets] = temp
        #println(all_weights[i-window_size+2,1:num_assets])
    end
end

# Convert weights to DataFrame for further analysis
df = DataFrame(all_weights[1:200,1:4],:auto)

