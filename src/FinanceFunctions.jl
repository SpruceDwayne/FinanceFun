#=This script will have functions directly relevant with finance stuff like
1. getting stock data check
2. mapping to log returns check
3. mapping back from log returns to real returns
4. functions related to pricing
5. functions doing portfolio optimization
6. functions related to backtesting
7. Intra day data hopefully
=#

module FinanceFunctions

using TSFrames, MarketData, DataFrames, Dates, Plots
using LinearAlgebra
using Distributions


export get_adjclose_dataframe, compute_relative_returns, sample_mv_t

function get_adjclose_dataframe(tickers = ["SPY","TLT","DBC"], start_date::DateTime = DateTime(2024,1,1), end_date:: DateTime = DateTime(2024,7,1),interval = "1mo")
    # Define the YahooOpt struct for date range and interval
    
    options = MarketData.YahooOpt(
        period1 = start_date,
        period2 = end_date,
        interval = interval,   # Daily data
        events = :history
    )

    # Fetch the data for the first ticker and initialize the TSFrame
    first_ticker = tickers[1]
    first_tsframe = TSFrame(yahoo(first_ticker,options)#MarketData.yahoo(first_ticker, options)
                            ; issorted = true)
    
    # Convert the TSFrame to a DataFrame and select the Index (time) and AdjClose columns
    df = DataFrame(first_tsframe)
    df = df[:, [:Index, :AdjClose]]
    DataFrames.rename!(df, :Index => :timestamp)    
    DataFrames.rename!(df, :AdjClose => first_ticker) 
    combined_df = df

    # Loop through the remaining tickers and merge their data
    for ticker in tickers[2:end]
        tsframe = TSFrame(yahoo(ticker,options)#MarketData.yahoo(ticker, options)
                            ; issorted = true)
        df = DataFrame(tsframe)
        df = df[:, [:Index, :AdjClose]]
        #DataFrames.rename!(df, Dict(:Index => :timestamp, :AdjClose => ticker))
        DataFrames.rename!(df, :Index => :timestamp)    
        DataFrames.rename!(df, :AdjClose => ticker) 
        
        # Perform the outerjoin using DataFrames
        combined_df = outerjoin(combined_df, df, on=:timestamp)
    end

    # Convert back to TSFrame with the appropriate timestamp column
    return TSFrame(combined_df, :timestamp; issorted = true)
end


function compute_relative_returns(df::DataFrame)
    # Ensure the timestamp is the first column and is in Date format
    if !isa(df.Index[1], Date)
        df.Index = Date.(df.Index)
    end
    
    # Sort by timestamp to ensure proper calculation
    sort!(df, :Index)
    
    # Calculate relative returns for each ticker
    # Exclude the timestamp column from the calculation
    tickers = names(df)[2:end]
    
    # Initialize a DataFrame to store the relative returns
    # The length of returns_df will be one less than the original df
    returns_df = DataFrame(Index=df.Index[2:end])
    
    for ticker in tickers
        # Compute the percentage change
        returns = diff(df[!,ticker]) ./ df[1:end-1, ticker]
        
        # Create a new column in returns_df with the computed returns
        returns_df[!, ticker] = returns
    end
    
    return returns_df
end


# Generalized function to sample from a multivariate t-distribution
function sample_mv_t(ν, μ, Σ, n_sim)
    d = size(Σ, 1)  # Dimension of the covariance matrix (number of assets)

    # Ensure the covariance matrix is symmetric
    Σ_symmetric = 0.5 * (Σ + Σ')
    
    # Add a small value to the diagonal to ensure positive definiteness
    λ = 1e-6
    Σ_regularized = Σ_symmetric + λ * I(d)

    # Sample from the multivariate normal distribution with covariance Σ
    Z = rand(MvNormal(Σ_regularized), n_sim)  # Z is d x n
    
    # Sample from a chi-squared distribution with degrees of freedom ν
    W = rand(Chisq(ν), n_sim)  # W is a vector of length n_sim
    
    # Preallocate the scaled Z matrix
    Z_scaled = zeros(size(Z))
    
    # Scale each column of Z by the corresponding entry in W
    for j in 1:n_sim
        scaling_factor = sqrt(ν)/sqrt(W[j]) 
        Z_scaled[:, j] = Z[:, j] * scaling_factor
    end
    
    # Add the mean vector μ (broadcasting to match dimensions)
    return μ .+ Z_scaled
end

end # module