using CSV
using DataFrames

# Replace "file.csv" with the path to your file
df = CSV.read("all_stocks_5yr.csv", DataFrame)

df_filtered = select(df, :date, :close, :Name)

# Step 2: Pivot the DataFrame so that stock 'Name' becomes column names and 'close' becomes the values
df_pivoted = unstack(df_filtered, :date, :Name, :close)

# Display the pivoted DataFrame (first 5 rows)
println(first(df_pivoted, 1))
