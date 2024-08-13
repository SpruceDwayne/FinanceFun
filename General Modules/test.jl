include("CVaR_functions.jl")
using .CVaR_functions
using LinearAlgebra
using Statistics
using SparseArrays
using CSV
using DataFrames

# Load the CSV file into a DataFrame
df = CSV.read(raw"C:\Users\basti\Coding\Finance-Fun\General Modules\simulations_data.csv", DataFrame)

# Convert the DataFrame to a matrix
matrix_data = Matrix(df)
current_weights = vcat(zeros(5),[0.4,0.6],zeros(9))
println("I made it here")
new_weights, updates  = update_portfolio_weights_new(current_weights,-matrix_data[1:10000,1:8],0.01,9999,true,0.05)
println(new_weights)
println(updates)
