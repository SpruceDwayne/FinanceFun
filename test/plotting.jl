using CSV, DataFrames, Plots

# Step 1: Replace data_path with the relative path to your file
data_path = "./data/all_stocks_5yr.csv"
df = CSV.read(data_path, DataFrame)

# Step 2: Select relevant data from csv
df = select(df, :date, :close, :Name)

# Step 3: Pivot the DataFrame so that stock 'Name' becomes column names and 'close' becomes the values
df = unstack(df, :date, :Name, :close)

# Step 4: Plot
plot(
  df.date, 
  [df[!, ticker] for ticker in names(df)[2:end]], 
  xlabel="Date", 
  ylabel="Adjusted Close", 
  title="Adjusted Close Prices of Tickers", 
  legend=:topleft
)