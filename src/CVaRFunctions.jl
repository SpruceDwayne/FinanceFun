#=
This script has functions related to CVaR and optimization involving CVaR
Note we always assume the simulations are losses
We only allow for one risk target in 'optimize_portfolio_CVaR_const' currently but that may be updated at a later time
Later there will also be functions for using CVaR for index tracking
=#



module CVaRFunctions

using LinearAlgebra
using Statistics
using JuMP
using GLPK

export update_portfolio_weights, optimize_portfolio_CVaR_const, optimize_portfolio_return_const

function f_alpha(l, simulations, alpha; probabilities=false, probability_list_=0)
    N = length(simulations)
    if probabilities
        probability_list = probability_list_
    else
        probability_list = fill(1, N)
    end
    
    shortfall = max.(simulations .- fill(l, N), 0)
    mean_shortfall = dot(shortfall, probability_list) * 1 / (1 - alpha)
    value = l + mean_shortfall
    return value
end



function optimize_portfolio_CVaR_const(scenario, risk = 0.02, alpha_ = 0.9, short_ = true)
    num_assets = size(scenario, 2)
    num_sim = size(scenario, 1)
        
    # Prepare cost vector
    mean_scenario = mean(scenario, dims=1)
    mean_scenario_vec = vec(mean_scenario)
    old_loss = zeros(num_sim)
    if short_
        var_index = 2*num_assets +1
        cost_vector = Vector{Float64}(undef, 2 * num_assets + 1 + num_sim)
        cost_vector[1:num_assets] .= -mean_scenario_vec 
        cost_vector[num_assets+1:2*num_assets] .= mean_scenario_vec 
        cost_vector[2*num_assets+1] = 0
        cost_vector[2*num_assets+2:end] .= 0

         #Initialize matrix
        constraint_matrix = Matrix{Float64}(undef, num_sim + 2, 2*num_assets + 1 + num_sim)
        #Aux var rows
        constraint_matrix[1:num_sim, 1:num_assets] = scenario #view_matrix
        constraint_matrix[1:num_sim, num_assets+1: 2*num_assets] = -scenario
        constraint_matrix[1:num_sim, 2*num_assets+1] .= -ones(num_sim)
        constraint_matrix[1:num_sim, 2*num_assets+2 : end] .= Matrix(1.0I, num_sim, num_sim)
        #CVaR row
        constraint_matrix[num_sim+1,1:2*num_assets] .=0
        constraint_matrix[num_sim+1,2*num_assets+1] =1 #VaR variable
        constraint_matrix[num_sim+1,2*num_assets+2:end] .=1/(num_sim *(1-alpha_)) 
        #Weights row
        constraint_matrix[num_sim+2, 1:num_assets] .= 1
        constraint_matrix[num_sim+2, num_assets+1:2*num_assets] .= -1
        
    else
        var_index = num_assets+1
        cost_vector = Vector{Float64}(undef, num_assets + 1 + num_sim)
        cost_vector[1:num_assets] .= -mean_scenario_vec 
         #Initialize matrix
         constraint_matrix = Matrix{Float64}(undef, num_sim + 2, num_assets + 1 + num_sim)
         #Aux var rows
         constraint_matrix[1:num_sim, 1:num_assets] = scenario 
         constraint_matrix[1:num_sim, num_assets+1] = -ones(num_sim)
         constraint_matrix[1:num_sim, num_assets+2 : end] .= Matrix(1.0I, num_sim, num_sim)
         #CVaR row
         constraint_matrix[num_sim+1,1:num_assets] .=0
         constraint_matrix[num_sim+1,num_assets+1] =1 #VaR variable
         constraint_matrix[num_sim+1,num_assets+2:end] .=1/(num_sim *(1-alpha_)) *ones(num_sim)

         #Weights row
         constraint_matrix[num_sim+2, 1:num_assets] .= 1
         constraint_matrix[num_sim+2, num_assets+1:end] .= 0
    end

    # Model
    m = Model(GLPK.Optimizer)
    num_variables = length(cost_vector)

    @variable(m, x[1:num_variables])
    
    for i in 1:num_variables
        if i != var_index
            @constraint(m, x[i] >= 0)
        
        end
    end

    @objective(m, Max, dot(cost_vector, x))

    level_index = num_sim + 1
    constraint_vector = zeros(size(constraint_matrix,1))
    constraint_vector[1:num_sim] += old_loss
    constraint_vector[num_sim+1] += risk
    self_fin_index = num_sim+2

    for i in 1:self_fin_index
        constraint_row_view = view(constraint_matrix, i, :)
        if i == level_index 
            @constraint(m, dot(constraint_row_view, x) <= constraint_vector[i])
        elseif i == self_fin_index
            @constraint(m, dot(constraint_row_view, x) == 1)
        else
            @constraint(m, dot(constraint_row_view, x) <= constraint_vector[i])
        end
    end
    
    optimize!(m)
    new_weights = JuMP.value.(x)

    return new_weights
end


function optimize_portfolio_return_const(scenario, alpha_, mean_return = 0.005, short_ = true)
    num_assets = size(scenario, 2)
    num_sim = size(scenario, 1)
        
    # Prepare cost vector
    mean_scenario = mean(scenario, dims=1)
    mean_scenario_vec = vec(mean_scenario)
    old_loss = zeros(num_sim)
    if short_
        var_index = 2*num_assets +1
        cost_vector = Vector{Float64}(undef, 2 * num_assets + 1 + num_sim)
        cost_vector[1:2*num_assets] .=0
        cost_vector[2*num_assets+1] =1 
        cost_vector[2*num_assets+2:end] .=1/(num_sim *(1-alpha_)) 

         #Initialize matrix
        constraint_matrix = Matrix{Float64}(undef, num_sim + 2, 2*num_assets + 1 + num_sim)
        #Aux var rows
        constraint_matrix[1:num_sim, 1:num_assets] = scenario #view_matrix
        constraint_matrix[1:num_sim, num_assets+1: 2*num_assets] = -scenario
        constraint_matrix[1:num_sim, 2*num_assets+1] .= -ones(num_sim)
        constraint_matrix[1:num_sim, 2*num_assets+2 : end] .= Matrix(1.0I, num_sim, num_sim)
        #CVaR row 
        constraint_matrix[num_sim+1,1:num_assets] .= -mean_scenario_vec 
        constraint_matrix[num_sim+1,num_assets+1:2*num_assets] .= mean_scenario_vec 
        constraint_matrix[num_sim+1,2*num_assets+1] = 0
        constraint_matrix[num_sim+1,2*num_assets+2:end] .= 0

        #Weights row
        constraint_matrix[num_sim+2, 1:num_assets] .= 1
        constraint_matrix[num_sim+2, num_assets+1:2*num_assets] .= -1
        
    else
        var_index = num_assets+1
        cost_vector = Vector{Float64}(undef, num_assets + 1 + num_sim)
        cost_vector[1:num_assets] .=0
        cost_vector[num_assets+1] =1
        cost_vector[num_assets+2 : end ] .=1/(num_sim *(1-alpha_)) *ones(num_sim)
        #Initialize matrix
        constraint_matrix = Matrix{Float64}(undef, num_sim + 2, num_assets + 1 + num_sim)
        #Aux var rows
        constraint_matrix[1:num_sim, 1:num_assets] = scenario 
        constraint_matrix[1:num_sim, num_assets+1] = -ones(num_sim)
        constraint_matrix[1:num_sim, num_assets+2 : end] .= Matrix(1.0I, num_sim, num_sim)
        #CVaR row
        constraint_matrix[num_sim+1,1:num_assets] .= -mean_scenario_vec 
        constraint_matrix[num_sim+1,num_assets+1:end] .= 0
        #Weights row
        constraint_matrix[num_sim+2, 1:num_assets] .= 1
        constraint_matrix[num_sim+2, num_assets+1:end] .= 0
    end

    # Model
    m = Model(GLPK.Optimizer)
    num_variables = length(cost_vector)

    @variable(m, x[1:num_variables])
    
    for i in 1:num_variables
        if i != var_index
            @constraint(m, x[i] >= 0)
        
        end
    end

    @objective(m, Min, dot(cost_vector, x))

    level_index = num_sim + 1
    constraint_vector = zeros(size(constraint_matrix,1))
    constraint_vector[1:num_sim] += old_loss
    constraint_vector[num_sim+1] += mean_return
    self_fin_index = num_sim+2

    for i in 1:self_fin_index
        constraint_row_view = view(constraint_matrix, i, :)
        if i == level_index 
            @constraint(m, dot(constraint_row_view, x) >= constraint_vector[i])
        elseif i == self_fin_index
            @constraint(m, dot(constraint_row_view, x) == 1)
        else
            @constraint(m, dot(constraint_row_view, x) <= constraint_vector[i])
        end
    end
    
    optimize!(m)
    new_weights = JuMP.value.(x)
    opt_val = JuMP.objective_value(m)
    return new_weights, opt_val
end



function update_portfolio_weights(old_weights, scenario, risk = 9999, return_ = 9999, short_ = true, trading_costs_ = 0.02, alpha_ = 0.95)
    if risk == 9999 && return_ == 9999
        println("Must choose either risk level or return level")
        return
    end
    if any(x -> x < -0.01, old_weights)
        println("--------------------------------")
        error("The vector contains negative values.")
    end
    
    num_assets = size(scenario, 2)
    num_sim = size(scenario, 1)
        
    # Prepare cost vector
    mean_scenario = mean(scenario, dims=1)
    mean_scenario_vec = vec(mean_scenario)
    
    #if short_
        old_loss = -scenario*old_weights[1:num_assets]+scenario*old_weights[num_assets+1:2*num_assets]
        var_index = 2*num_assets +1
        cost_vector = Vector{Float64}(undef, 2 * num_assets + 1 + num_sim)
        cost_vector[1:num_assets] .= -mean_scenario_vec .- trading_costs_ #Long gives expected return minus trading cost
        cost_vector[num_assets+1:2*num_assets] .= mean_scenario_vec .- trading_costs_ #Going short gives expected loss - trading cost
        cost_vector[2*num_assets+1] = 0
        cost_vector[2*num_assets+2:end] .= 0

         #Initialize matrix
        constraint_matrix = Matrix{Float64}(undef, num_sim + 2, 2*num_assets + 1 + num_sim)
        #Aux var rows
        constraint_matrix[1:num_sim, 1:num_assets] .= scenario #-trading_costs_ * ones(num_sim,  num_assets)
        constraint_matrix[1:num_sim, num_assets+1: 2*num_assets] = -scenario #-trading_costs_ * ones(num_sim,  num_assets)
        constraint_matrix[1:num_sim, 2*num_assets+1] .= -ones(num_sim)
        constraint_matrix[1:num_sim, 2*num_assets+2 : end] .= Matrix(1.0I, num_sim, num_sim)
        #CVaR row
        constraint_matrix[num_sim+1,1:2*num_assets] .=0
        constraint_matrix[num_sim+1,2*num_assets+1] =1 #VaR variable
        constraint_matrix[num_sim+1,2*num_assets+2:end] .=1/(num_sim *(1-alpha_)) 
        #Self financing row
        constraint_matrix[num_sim+2, 1:end] .= 0
        constraint_matrix[num_sim+2, 1:num_assets] .= (1 + trading_costs_)
        constraint_matrix[num_sim+2, num_assets+1:2*num_assets] .= (-1 + trading_costs_)
        constraint_matrix[num_sim+2, 2*num_assets+1:end] .= 0
    #else
    #    old_loss = -scenario*old_weights[1:num_assets]
    #    var_index = num_assets +1
    #    cost_vector = Vector{Float64}(undef, num_assets + 1 + num_sim)
    #    cost_vector[1:num_assets] .= -mean_scenario_vec .- trading_costs_ #Long gives expected return minus trading cost
    #    cost_vector[num_assets+1:end] .= 0
        

         #Initialize matrix
    #    constraint_matrix = Matrix{Float64}(undef, num_sim + 2, num_assets + 1 + num_sim)
        #Aux var rows
    #    constraint_matrix[1:num_sim, 1:num_assets] .= scenario -trading_costs_ * ones(num_sim, num_assets)
    #    constraint_matrix[1:num_sim, num_assets+1] .= -ones(num_sim)
    #    constraint_matrix[1:num_sim, num_assets+2 : end] .= Matrix(1.0I, num_sim, num_sim)
        #CVaR row
    #    constraint_matrix[num_sim+1,1:num_assets] .=0
    #    constraint_matrix[num_sim+1,num_assets+1] =1 #VaR variable
    #    constraint_matrix[num_sim+1,num_assets+2:end] .=1/(num_sim *(1-alpha_)) 
        #Self financing row
     #   constraint_matrix[num_sim+2, 1:end] .= 0
    #    constraint_matrix[num_sim+2, 1:num_assets] .= (1 + trading_costs_)
    #    constraint_matrix[num_sim+2, num_assets+1:end] .= 0
    #end
    
    # Model
    m = Model(GLPK.Optimizer)
    num_variables = length(cost_vector)

    @variable(m, x[1:num_variables])
    ##############Rethink the constraints below. How to make sure weights dont explode####################
    if short_
        for i in 1 :num_variables
            if i != var_index
                @constraint(m, x[i] >= 0)
            end
        end
    else
        for i in 1:num_variables #num_assets+1 :num_variables
            if i != var_index
                @constraint(m, x[i] >= 0)
            end
        end
        for i in num_assets+1 : 2*num_assets#1:num_assets
            #@constraint(m, x[i] >= -old_weights[i] )
            @constraint(m, x[i] <= old_weights[i-num_assets]-0.00001 )
        end
        for i in 1:num_assets
            @constraint(m, x[i] <= 1-old_weights[i] )
           # @constraint(m, x[i] <= old_weights[i-num_assets] )
        end
    end
    

    @objective(m, Max, dot(cost_vector, x))

    level_index = num_sim + 1
    constraint_vector = zeros(size(constraint_matrix,1))
    constraint_vector[1:num_sim] += -old_loss
    constraint_vector[num_sim+1] += risk
    #constraint_vector[self_fin_index] = 0 #Not needed sinze it is initialized as 0
    self_fin_index = size(constraint_vector, 1)

    for i in 1:self_fin_index
        constraint_row_view = view(constraint_matrix, i, :)
        if i == level_index
            if risk != 9999
                @constraint(m, dot(constraint_row_view, x) <= constraint_vector[i])
            else
                @constraint(m, dot(constraint_row_view, x) >= constraint_vector[i])
            end
        elseif i == self_fin_index
            @constraint(m, dot(constraint_row_view, x) == 0)
        else
            @constraint(m, dot(constraint_row_view, x) <= constraint_vector[i])
        end
    end
    optimize!(m)
    updates = JuMP.value.(x)
    multiplier = 2#short_ ? 2 : 1
    new_weights = old_weights + updates[1:num_assets * multiplier]
    return new_weights, updates
end

end # module