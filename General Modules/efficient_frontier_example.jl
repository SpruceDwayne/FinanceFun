include("CVaR_functions.jl")
include("finance_functions.jl")
using .CVaR_functions
using .finance_functions
using LinearAlgebra
using Statistics
using SparseArrays
using CSV
using TSFrames, MarketData, DataFrames, Dates,Plots


tickers = ["SPY","TLT","DBC","SHY","NVO","AAPL","LLY","AMD","NVDA","MSFT","AMZN","^STOXX"]
start_date = DateTime(2010, 8, 1)
end_date = DateTime(2024, 8, 20)
df =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date,"1d"))
df = compute_relative_returns(dropmissing(df))
matrix_data = Matrix(df)

# Define the function to get the efficient frontier
function get_efficient_frontier(scenario, alpha_, 
                                short_, num_points)
    # Step 1: Get the range of possible returns
    mean_returns = mean(scenario, dims=1)  # Expected returns of each asset
    min_return = minimum(mean_returns)
    max_return = maximum(mean_returns)

    # Generate the list of target returns (equidistant points between min and max)
    return_levels = range(min_return, stop=max_return, length=num_points)

    # Store the results: list of returns and risks (standard deviations)
    frontier_returns = []
    frontier_risks = []

    # Step 2: Optimize for each return level
    for mean_return in return_levels
        # Optimize portfolio for the given return level
        optimized_weights, risk_ = optimize_portfolio_return_const(-scenario, alpha_;
                                                            mean_return=mean_return, 
                                                            short_=short_)
        # Calculate the return and risk for the optimized portfolio
        portfolio_return = dot(optimized_weights[1:size(scenario,2)], vec(mean_returns))  # Portfolio return
        portfolio_risk = risk_  # Portfolio risk (std deviation)

        # Append the results
        push!(frontier_returns, portfolio_return)
        push!(frontier_risks, portfolio_risk)
    end

    # Return the efficient frontier data
    return frontier_returns, frontier_risks
end

# Example usage:
scenario = matrix_data  # Your scenario matrix (rows are simulations, cols are assets)
alpha_ = 0.0           # CVaR confidence level

# Get efficient frontier
frontier_returns, frontier_risks = get_efficient_frontier(scenario[1:end,2:end], alpha_, false, 5)


scatter(frontier_risks, frontier_returns, 
        xlabel="1month CVaR95)", 
        ylabel="Return", 
        title="Efficient Frontier", 
        legend=false,
        markersize=5,  # Size of the points
        markerstrokewidth=0.5)  # Thickness of the outline of each point