include("finance_functions.jl")
include("CVaR_functions.jl")

using .finance_functions
using .CVaR_functions
using DataFrames
using Dates
using Statistics
using ARCHModels

# Parameters
tickers = ["SPY", "TLT", "DBC"]
start_date = DateTime(2006, 3, 1)
end_date = DateTime(2024, 8, 20)
window_size = 1000
n_sim = 1000
ν = 3  # Degrees of freedom for the t-distribution

# Load and process data
df = DataFrame(get_adjclose_dataframe(tickers, start_date, end_date, "1d"))
df = compute_relative_returns(df)
matrix_data = Array{Float64, 2}(Matrix(df)[:, 2:end])
matrix_data = hcat(matrix_data, 0.0*ones(size(matrix_data, 1)))


# Initialize variables
num_assets = size(matrix_data, 2)
num_rows = size(matrix_data, 1)
all_weights = Matrix{Float64}(undef, num_rows - window_size + 1, 2*num_assets) 


new_weights = [1/3, 1/3, 1/3,0,0, 0, 0, 0]
all_weights[1, :] = new_weights

# Iteratively fit the GARCH model and update portfolio weights
for i in window_size:window_size+30
    # Define the window of data
    window_data = matrix_data[i-window_size+1:i, :]

    # Demean the data
    mean_vector = mean(window_data, dims=1)
    demeaned_data = window_data .- mean_vector
    mean_vector = vec(mean(matrix_data[:, 1:end], dims=1))
    # Fit the GARCH model
    m = fit(DCC{1,1}, demeaned_data)
    predicted_covariance = predict(m, what=:covariance)
    # Sample multivariate t-distribution (optional, if needed for further analysis)
    simulated_data = transpose(sample_mv_t(ν, mean_vector, predicted_covariance, n_sim))
    risk_target = sum(all_weights[i+1-window_size,1:end])*0.05
    # Update portfolio weights
    new_weights_new, updates = update_portfolio_weights(all_weights[i+1-window_size,1:end], -window_data[1:end, 1:end], risk_target, 9999, false, 0.001, 0.99)
    new_weights_new[1:num_assets] = new_weights_new[1:num_assets] -new_weights_new[num_assets+1 : 2*num_assets]
    new_weights_new[num_assets+1 : 2*num_assets] .= 0
    #new_weights_new = [1/3, 1/3, 1/3, 0, 0, 0]
    
    if i < num_rows
        latest_return = matrix_data[i+1, :]  # Next period's return
        println("-------------------")
        println(risk_target)
        println(sum(all_weights[i+1-window_size,1:end]))
        println(new_weights_new)
        new_weights_new[1:num_assets] .*=(1 .+ latest_return)
        println(new_weights_new)
        println(latest_return)
        println(".................")

        #new_weights_new = all_weights[i-window_size+1, 1:num_assets] .*(1 .+ latest_return)

        # Update the weights matrix
        #println("-------------")
        #println(i)
        #println(new_weights_new)
        #println(sum(new_weights_new))
        #println("----------------------")
        all_weights[i-window_size+2, 1:num_assets] = new_weights_new[1:num_assets]
    end
end

# Convert weights to DataFrame for further analysis
#weights_df = DataFrame(Date=dates, all_weights...)
