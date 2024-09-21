# Add Revise for development
using Revise

# Internal
using FinanceFun

# External
using Dates, DataFrames, Plots

#### Tests for FinanceFunctions ####
# Test get_adjclose_dataframe
tickers = ["YOU", "ARE", "MAN"]
start_date = DateTime(2022, 1, 6)
end_date = DateTime(2023, 9, 11)
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
weights = optimize_portfolio_CVaR_const(_returns, 0.05, 0.95, false)
# weights = optimize_portfolio_CVaR_const(_returns, 0.05, 0.95, true) # Fails

# Test update_portfolio_weights
current_weights = [0.34,0.33,0.33,0,0,0]
_weights, updates = update_portfolio_weights(current_weights, _returns, 0.05, 9999, false, 0.05, 0.95)
_weights, updates = update_portfolio_weights(current_weights, _returns, 0.05, 9999, true, 0.05, 0.95)
_weights, updates = update_portfolio_weights(current_weights, _returns, 9999, 0.05, false, 0.05, 0.95)
_weights, updates = update_portfolio_weights(current_weights, _returns, 9999, 0.05, true, 0.05, 0.95)
_weights, updates = update_portfolio_weights(current_weights, _returns, 0.05, 0.05, false, 0.05, 0.95)
_weights, updates = update_portfolio_weights(current_weights, _returns, 0.05, 0.05, true, 0.05, 0.95)


