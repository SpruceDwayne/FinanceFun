# Add Revise for development
using Revise

# Internal
using FinanceFun

# External
using Dates, DataFrames, Plots

#### Tests for FinanceFunctions ####
# Test get_adjclose_dataframe
tickers = ["SPY", "TLT", "IEF","SHY","DBC"]
start_date = DateTime(2010, 1, 6)
end_date = DateTime(2020, 9, 11)
closes = DataFrame(get_adjclose_dataframe(tickers, start_date, end_date))

plot(
    closes.Index, 
    [closes[!,ticker] for ticker in tickers], 
    xlabel="Date", 
    ylabel="Adjusted Close", 
    title="Adjusted Close Prices of Tickers", 
    legend=:topleft
)

# Test compute_relative_returns
returns = compute_relative_returns(closes)

plot(
    returns.Index, 
    [returns[!,ticker] for ticker in tickers], 
    xlabel="Date", 
    ylabel="Returns", 
    title="Returns of Tickers", 
    legend=:topleft,
)

#### Tests for CVaRFunctions ####
# Test optimize_portfolio_CVaR_const
_returns = Matrix(returns[:, 2:end])
weights = optimize_portfolio_CVaR_const(-_returns, 0.05, 0.95, false)
# weights = optimize_portfolio_CVaR_const(_returns, 0.05, 0.95, true) # Fails

# Test update_portfolio_weights
current_weights = [0.34,0.33,0.33,0,0,0,0,0,0,0]
n = size(returns,2)-1

println(weights[1:n]-weights[n+1 : 2*n])
#If we lower the trading costs we get closer and closer to the 60-40 portfolio
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.02,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.01,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.005,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.0005,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.0002,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.0001,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.00005,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.00001,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.00001,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.000001,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.0000001,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.00000001,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.000000001,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])
new_weights, updates = update_portfolio_weights(current_weights,-_returns[1:end,1:end],0.05,9999,false,0.0000000001,0.95)
println(new_weights[1:n]-new_weights[n+1 : 2*n])




