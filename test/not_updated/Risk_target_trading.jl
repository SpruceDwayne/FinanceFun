# Add Revise for development
using Revise

# Internal
using FinanceFun

# External
using Dates, DataFrames, Plots
using Statistics
using ARCHModels
using PlotlyJS

#Parameters:
tickers = ["IEF","SPY"]
start_date = DateTime(2010, 3, 1)
end_date = DateTime(2024, 8, 20)
window_size = 100
n_sim = 1000
ν = 3  # Degrees of freedom for the t-distribution

#Data prep
df_price =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date,"1mo"))
df = dropmissing(df_price)
df = compute_relative_returns(df)

matrix_data = Array{Float64, 2}(Matrix(df)[:,2:end]) 
matrix_data = Array{Float64, 2}(Matrix(df)[:, 2:end])
#We also add some cash position with some neglegable noise if one had money in all cash during the period total gain would be 0.1%
matrix_data = hcat(matrix_data, 0.00002 .+ (0.00002 - 0.00001) .* rand(size(matrix_data, 1)))

#Size variable and initialise portfolio at equal weights
num_assets = size(matrix_data, 2)
num_rows = size(matrix_data, 1)
all_weights = zeros( num_rows - window_size + 1, 2*num_assets) 
all_weights[1,:] = vcat(1/num_assets * ones(num_assets),zeros(num_assets))

# Iteratively fit the GARCH model and update portfolio weights according to cvar budget
for i in window_size: num_rows
    # Define the window of data
    window_data = matrix_data[i-window_size+1:i, :] #Use all the previous history available
    mean_vector = mean(window_data, dims=1)
    mean_vector = zeros(num_assets)
    # Fit the GARCH model
    m = fit(DCC{1,1}, window_data)
    predicted_covariance = cov(window_data)  
    try #Since the predict sometimes fail
        predicted_covariance = predict(m, what=:covariance)
    catch e
        #println("Prediction failed, using default covariance matrix.")
    end
    # Sample from corresponding multivariate t-distribution
    simulated_data = transpose(sample_mv_t(ν, mean_vector, predicted_covariance, n_sim))    
    simulated_data .+=mean(window_data, dims=1)
    #Set the risk budget
    risk_target =0.2* sum(all_weights[i-window_size+1,1:num_assets])
    # Update portfolio weights
    local new_weights, updates = update_portfolio_weights(all_weights[i+1-window_size,1:end],
                                                             -simulated_data[1:end, 1:end], risk_target, 9999, false, 0.005, 0.9)
    
    if i < num_rows
        latest_return = matrix_data[i+1, :]  # Next period's return
        #Gymnastics
        temp =round.( new_weights[1:num_assets]-new_weights[num_assets+1:2*num_assets], digits=2)
        temp = max.(0, temp)
        temp .*= (1 .+ latest_return) #Markets affecting portfolio weights before next iteration
        temp =temp/sum(temp)
        all_weights[i-window_size+2, 1:num_assets] = temp
    end
end

df_W = DataFrame(all_weights[:,1:num_assets],:auto)
df_W.Index = df.Index[window_size:end]
new_df = DataFrame(
    Index = df_W.Index,
    Bonds = df_W.x1,
    Stocks = df_W.x2,
    Cash = df_W.x3
)

#Compare stock bond balance
Plot(
    new_df.Index, 
    new_df.Stocks, 
    xlabel="Date", 
    ylabel="Returns", 
    title="Returns of Tickers", 
    legend=:topleft,
)

#Now we compare the profitability of the stragty to 60/40 and buy-hold in the IEF and SPY
#First more dataprep
df_price[!, :Cash] = ones(size(df_price, 1))

normalized_spy =df_price[101:end,:].SPY ./ df_price[101,:].SPY
normalized_BOND =df_price[101:end,:].IEF ./ df_price[101,:].IEF
normalized_6040 =0.6 *normalized_spy+0.4*normalized_BOND

df_sp500 = DataFrame(Value=normalized_spy, Asset="SP500", Date = new_df.Index)
df_bond = DataFrame(Value=normalized_BOND, Asset="IEF", Date = new_df.Index)
df_6040 = DataFrame(Value=normalized_6040, Asset="60/40", Date = new_df.Index)

#Compute strategy returns:
portfolio_values=zeros(size(all_weights,1))
portfolio_values[1] =1
for i in 1+1:size(all_weights,1)
    temp = transpose(all_weights[i,1:num_assets])*(matrix_data[window_size-1+i,1:end].+1)
    portfolio_values[i] =portfolio_values[i-1] * temp
end
df_portfolio = DataFrame(Value=portfolio_values, Asset="Risk target",Date = new_df.Index)
 
# Combine the 3 DataFrames
df_combined = vcat(df_portfolio, df_sp500,df_bond,df_6040)
#Plot trajectories
Plot(
    df_combined, mode="markers+lines",
    x=:Date, y=:Value, color=:Asset
)

function log_returns(values::AbstractVector{<:Real})
    return [NaN; log.(values[2:end] ./ values[1:end-1])]
end

portfolio_returns = log_returns(portfolio_values)
spy_returns = log_returns(normalized_spy)
bond_returns = log_returns(normalized_BOND)


returns_matrix = hcat(portfolio_returns[2:end], spy_returns[2:end], bond_returns[2:end])  # Exclude the first NaN
correlation_matrix = cor(returns_matrix)
#=
Now in a world with more assets to choose from one might have
found that the returns series from the trading strategy have low correlation
to other assets and hence could serve as a diversifier together with "buy and hold"
strategies.
=#
