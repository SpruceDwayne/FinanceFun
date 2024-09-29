# Add Revise for development
using Revise

# Internal
using FinanceFun

# External
using Dates, DataFrames, Plots
using Statistics
using LinearAlgebra
using JuMP
using GLPK
#using PlotlyJS #Dont load plotly unless you have to. It messes with stuff

#=
In this script we try to replicate SP500 by using the 30 biggest stocks.
Considering that those 30 giants account for 50% isch of SP500 we should
be able to get pretty close. We will choose our portfolio weights by minimizing
the risk of deviation from the index, with cvar being our risk measure.
=#

tickers1 = ["AAPL","MSFT","NVDA","AMZN","GOOGL","AVGO","TSLA","LLY","JPM","UNH"]#Skips meta because name change is too many Gymnastics
tickers2 =[ "BRK-B","GOOG","XOM","V","PG","MA","COST","HD","JNJ","WMT"]
tickers3 = ["ABBV","NFLX","MRK","KO","AMD","ORCL","CRM","BAC","CVX","TMO"]
tickerspy =[ "SPY"] 
start_date = DateTime(2013, 1, 1)
end_date = DateTime(2024, 8, 20)

#Data prep
df_price1 =DataFrame( get_adjclose_dataframe(tickers1, start_date, end_date,"1wk"))
df_price2 =DataFrame( get_adjclose_dataframe(tickers2, start_date, end_date,"1wk"))
df_price3 =DataFrame( get_adjclose_dataframe(tickers3, start_date, end_date,"1wk"))
df_spy = DataFrame( get_adjclose_dataframe(tickerspy, start_date, end_date,"1wk"))

df =hcat(df_price1,df_price2[:,2:end],df_price3[:,2:end])
df = compute_relative_returns(df)
df_spy = compute_relative_returns(df_spy)
matrix_data = Array{Float64, 2}(Matrix(df)[:, 2:end])
spy_matrix = Array{Float64, 2}(Matrix(df_spy)[:, 2:end])
########We now construct the custom CVaR optimizer for this task##############
#We will optimize like in https://ira.lib.polyu.edu.hk/bitstream/10397/92481/1/Gendreau_Cvar-Lasso_Enhanced_Index.pdf
#In summary the idea is to pose it as a lasso problem where we minimize the risk of underperforming the benchmark
#and add a lasso constraint sum(abs(w)) < leverage that limits the amount of shorting and encourages a sparse solution(see equivalent form of lasso problem in regression)
function replicate_index_w_cvar(scenario,index_returns, alpha_, leverage)
    num_assets = size(scenario, 2)
    num_sim = size(scenario, 1)
        
    # Prepare cost vector
    old_loss = index_returns
    var_index = 2*num_assets +1
    cost_vector = zeros( 2 * num_assets + 1 + num_sim + 2*num_assets)
    cost_vector[2*num_assets+1] =1 
    cost_vector[2*num_assets+2:2 * num_assets + 1 + num_sim] .=1/(num_sim *(1-alpha_)) 

    #Initialize matrix
    constraint_matrix = zeros(num_sim + 1, 2*num_assets + 1 + num_sim+2*num_assets)
    #Aux var rows
    constraint_matrix[1:num_sim, 1:num_assets] = scenario #view_matrix
    constraint_matrix[1:num_sim, num_assets+1: 2*num_assets] = -scenario
    constraint_matrix[1:num_sim, 2*num_assets+1] = -ones(num_sim)
    constraint_matrix[1:num_sim, 2*num_assets+2 : 2*num_assets+1+num_sim] = Matrix(1.0I, num_sim, num_sim)
    #Leverage doesnt get its own row for generating constraints but will instead be coded directly
    #Weights row
    constraint_matrix[num_sim+1, 1:num_assets] .= 1
    constraint_matrix[num_sim+1, num_assets+1:2*num_assets] .= -1

    
    # Model
    m = Model(GLPK.Optimizer)
    num_variables = length(cost_vector)

    @variable(m, x[1:num_variables])
    
    for i in 1:num_variables
        if i != var_index
            @constraint(m, x[i] >= 0)
        end
    end

    for i in 1:num_assets
        @constraint(m, x[i]-x[i+num_assets]-x[ 2 * num_assets + 1 + num_sim+i] <= 0)
        @constraint(m, -x[i]+x[i+num_assets]-x[ 2 * num_assets + 1 + num_sim+i] <= 0)

    end

    @constraint(m, dot(ones(num_assets),x[ 2 * num_assets + 1 + num_sim+1:2 * num_assets + 1 + num_sim+num_assets]) <= leverage)



    @objective(m, Min, dot(cost_vector, x))

    constraint_vector = zeros(size(constraint_matrix,1))
    constraint_vector[1:num_sim] += old_loss
    self_fin_index = num_sim+1

    for i in 1:self_fin_index
        constraint_row_view = view(constraint_matrix, i, :)
        if i ==  self_fin_index
            @constraint(m, dot(constraint_row_view, x) == 1)
        else
            @constraint(m, dot(constraint_row_view, x) <= constraint_vector[i])
        end
    end
    
    optimize!(m)
    new_weights = JuMP.value.(x)
    return new_weights
end

#Now we can find the optimal allocation by
portfolio = replicate_index_w_cvar(matrix_data,spy_matrix,0.9,1.5)

#To backtest this strategy we will start each year by making a "regression" on the previous years.
function run_rolling_optimization(matrix_data, spy_matrix, alpha, beta)
    # Define the row ranges for each year based on 52 weeks per year
    rows_total = size(matrix_data, 1)  # Total number of rows in the data
    weights_by_year = Dict{Int, Vector{Float64}}()
    weeks_per_year = 52
    # Start the optimization from 2014
    for year in 2014:2024
        # Calculate the row indices for the current year
        start_idx = max(1, (year - 2013) * weeks_per_year - (weeks_per_year - 1))  # Start index for data from 2013 onwards
        end_idx = min(rows_total, (year - 2013 + 1) * weeks_per_year)  # End index for the current year
        
        # Use all data from 2013 until the current year for optimization
        data_until_now = matrix_data[1:end_idx, :]
        spy_data_until_now = spy_matrix[1:end_idx, :]

        # Run the optimization for this subset of data
        portfolio = replicate_index_w_cvar(data_until_now, spy_data_until_now, alpha, beta)

        # Extract portfolio weights
        weights = portfolio[1:30] - portfolio[31:60]

        # Store the weights by year
        weights_by_year[year] = weights
    end

    return weights_by_year
end

weights_by_year = run_rolling_optimization(matrix_data, spy_matrix, 0.9, 1.5)

function generate_relative_returns(matrix_data, weights_by_year)
    relative_returns_by_year = Dict{Int, Vector{Float64}}()

    weeks_per_year = 52
    rows_total = size(matrix_data, 1)

    for year in 2014:2024
        # Get the portfolio weights for the current year
        weights = weights_by_year[year]

        # Calculate the row indices for the current year
        start_idx = (year-2014+1)*52        #max(1, (year - 2013) * weeks_per_year - (weeks_per_year - 1))
        end_idx = min(rows_total, start_idx+51)
        println(start_idx)
        println(end_idx)
        # Get the matrix of relative returns for the current year
        matrix_of_relative_returns = matrix_data[start_idx:end_idx, :]

        # Calculate the series of relative returns by multiplying matrix with the weights
        relative_returns = matrix_of_relative_returns * weights

        # Store the relative returns series for the current year
        relative_returns_by_year[year] = relative_returns
    end

    return relative_returns_by_year
end

# Example usage:
relative_returns_by_year = generate_relative_returns(matrix_data, weights_by_year)
function compute_cumulative_returns(relative_returns_by_year)
    cumulative_returns = Float64[]
    cumulative_return = 1.0  # Start with an initial value of 1 for cumulative return
    
    for year in sort(collect(keys(relative_returns_by_year)))  # Loop over years in order
        # Get the relative returns for the current year
        relative_returns = relative_returns_by_year[year]
        
        # Add 1 to each entry in the relative returns and calculate cumulative returns
        for r in relative_returns
            cumulative_return *= (1 + r)
            push!(cumulative_returns, cumulative_return)
        end
    end

    return cumulative_returns
end

# Example usage:
cumulative_returns = compute_cumulative_returns(relative_returns_by_year)

println(cumulative_returns)

using Plots

function collect_all_relative_returns(relative_returns_by_year)
    all_relative_returns = Float64[]  # Empty array to store all relative returns
    
    for year in sort(collect(keys(relative_returns_by_year)))
        # Append the relative returns for each year to the all_relative_returns array
        append!(all_relative_returns, relative_returns_by_year[year])
    end

    return all_relative_returns
end

# Example usage:
all_relative_returns = collect_all_relative_returns(relative_returns_by_year)

# Plot the histogram
spy_cum = cumprod(spy_matrix[52:end,1] .+1)

function plot_cumulative_returns(cumulative_returns, spy_cum,xaxis)
    # Ensure both series have the same length (truncate the longer one if needed)
    min_length = min(length(cumulative_returns), length(spy_cum))
    cumulative_returns = cumulative_returns[1:min_length]
    spy_cum = spy_cum[1:min_length]

    # Generate the plot
    plot(xaxis, cumulative_returns, label="Portfolio Cumulative Returns", xlabel="Time (Weeks)", ylabel="Cumulative Returns", title="Cumulative Returns Comparison")
    plot!(xaxis, spy_cum, label="SPY Cumulative Returns", linestyle=:dash)
end

# Example usage:

plot_cumulative_returns(cumulative_returns, spy_cum,df.Index[52:end])



all_relative_returns