include("finance_functions.jl")
using .finance_functions
using TSFrames, MarketData, DataFrames, Dates
using Plots


tickers = ["SPY","TLT","DBC"]
start_date = DateTime(2008, 8, 1)
end_date = DateTime(2024, 8, 20)
df =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date))

plot(
    df.Index, 
    [df[!,ticker] for ticker in tickers], 
    xlabel="Date", 
    ylabel="Adjusted Close", 
    title="Adjusted Close Prices of ETFs", 
    legend=:topleft
)
