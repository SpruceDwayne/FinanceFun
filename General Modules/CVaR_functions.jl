#=
This script has functions related to CVaR and optimization involving CVaR
For now the results will only work for data and alpha such that VaR postitive (ie a loss)
In later version this will be updated such that the optimizer wont forze VaR to be positive
But since that should be the case a.s. it might take some time before this update.
Note we always assume the simulations are returns so if you have loss data make sure to multiply with -1
We only allow for one risk target in 'optimize_portfolio_CVaR_const' currently but that may be updated at a later time
Later we will also alow for the option to include trading costs by:
    -letting weights sum to a total cash amount !=1 for when simulations are not log_returns
    -Ensuring simulations include a cash account where there are no changes
    -Ensuring that w_new +transactioncosts(w_new-w_old)=w_old, ie self financing portfolio
    => w_new-w_old +0.01*abs(w_new-w_old)
=#

using LinearAlgebra
using JuMP
using GLPK
using SparseArrays
using LinearAlgebra
using Statistics


module CVaR_functions

using LinearAlgebra
using JuMP
using GLPK
using HiGHS
using SparseArrays
using LinearAlgebra
using Statistics

export f_alpha, optimize_portfolio_CVaR_const,optimize_portfolio_return_const, get_constraint_matrix_min_cvar,add_trading_costs,get_constraint_vector_trading_cost,update_portfolio_weights

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


function get_constraint_matrix_risk_budget(simulations,alpha_;short=true)
    n = size(simulations,2) #num_assets
    m= size(simulations,1) #num_sim
    custom_matrix = sparse(ones(m, 1))
    multiplier = short ? 2 : 1

    risk_row = sparse([zeros(multiplier * n); 1.0; fill(1/(m*(1-alpha_)), m);])
    if short
        left_part = hcat(simulations, -simulations)
        custom_matrix = hcat(left_part, custom_matrix)
        weights_row = sparse([ones(n); -ones(n); 0.0; zeros(m)])
    else
        custom_matrix = hcat(-simulations, custom_matrix)
        weights_row = sparse([0.0; zeros(m); ones(n)])
    end
    custom_matrix = hcat(custom_matrix, spdiagm(0 => ones(m)))
    custom_matrix = vcat(custom_matrix, transpose(risk_row))
    custom_matrix = vcat(custom_matrix, transpose(weights_row))


    return custom_matrix
end


function optimize_portfolio_CVaR_const(scenario::Matrix{Float64}, alpha_::Float64; risk::Float64 = 0.05, short = true,total_cash = 1,initial_alloctation = 0)
    num_assets = size(scenario, 2)
    num_sim = size(scenario, 1)
    constraint_matrix = get_constraint_matrix_Risk(scenario, alpha_,short = short)
    multiplier = short ? 2 : 1

    if short
        cost_vector = [mean(scenario, dims=1); -mean(scenario, dims=1); 0; zeros(num_sim)]
    else
        cost_vector = [mean(scenario, dims=1); 0; zeros(num_sim)]
    end


    m = Model(GLPK.Optimizer)
    num_variables = multiplier*num_assets + num_sim + 1
    @variable(m, x[1:num_variables] >= 0)
 
    @objective(m, Max, dot(cost_vector, x))

    for i in 1:num_sim
        @constraint(m, sum(constraint_matrix[i, j] * x[j] for j in 1:length(x)) >= 0)
    end
    @constraint(m, sum(constraint_matrix[num_sim + 1, j] * x[j] for j in 1:length(x)) <= risk)
    @constraint(m, sum(constraint_matrix[num_sim + 2, j] * x[j] for j in 1:length(x)) == total_cash)

    optimize!(m)

    return JuMP.value.(x)
end

function get_constraint_matrix_min_cvar(simulations, alpha; short=true)
    n = size(simulations, 2) # num_assets
    m = size(simulations, 1) # num_sim
    custom_matrix = sparse(ones(m, 1))
    means = mean(-simulations, dims=1) #minus since we assume the simulations are losses
    means = reshape(means, n)
    
    if short
        left_part = hcat(simulations, -simulations)
        custom_matrix = hcat(left_part, custom_matrix)
        weights_row = sparse([ones(n); -ones(n); 0.0; zeros(m)])
        return_row = sparse([means; -means; 0.0; zeros(m)])
    else
        custom_matrix = hcat(-simulations, custom_matrix)
        weights_row = sparse([0.0; zeros(m); ones(n)])
        return_row = sparse([means; 0.0; zeros(m)])
    end
    
    custom_matrix = hcat(custom_matrix, spdiagm(0 => ones(m)))
    custom_matrix = vcat(custom_matrix, transpose(return_row))
    custom_matrix = vcat(custom_matrix, transpose(weights_row))

    return custom_matrix
end

function optimize_portfolio_return_const(scenario::Matrix{Float64}, alpha_::Float64; mean_return = 0.005, short = true)
    num_assets = size(scenario, 2)
    num_sim = size(scenario, 1)
    constraint_matrix = get_constraint_matrix_min_cvar(scenario, alpha_, short = short)
    multiplier = short ? 2 : 1

    cost_vector = sparse([zeros(multiplier * num_assets); 1.0; fill(1/(num_sim*(1-alpha_)), num_sim);])

    m = Model(HiGHS.Optimizer)
    num_variables = multiplier * num_assets + num_sim + 1
    @variable(m, x[1:num_variables] >= 0)
    
    @objective(m, Min, dot(cost_vector, x))

    for i in 1:num_sim
        @constraint(m, sum(constraint_matrix[i, j] * x[j] for j in 1:num_variables) >= 0)
    end
    
    @constraint(m, sum(constraint_matrix[num_sim + 1, j] * x[j] for j in 1:num_variables) >= mean_return)
    @constraint(m, sum(constraint_matrix[num_sim + 2, j] * x[j] for j in 1:num_variables) == 1)
    
    optimize!(m)
    
    return JuMP.value.(x)
end


function add_trading_costs(constraint_matrix , n_sim ,n_assets , risk_role = "constraint", shorting =true,trading_cost = 0.02,)
    #1. trading costs to auxillary variable constraint [0.02*1m, 0.02*1m], dim(1m) = n_assets x num_sim (positive and negative part of difference),1m a square matrix of 1's
    b1 = trading_cost*ones(n_sim,2*n_assets)
    #2. if risk row add 0's since we already added trading costs to the losses, if return row add trading costs 
    #   as -1_k^T*trading_cost, k = 2*n_assets (1 for positive and 1 for negative differences in new weights)
    if risk_role == "constraint"
        b2 =zeros(2*n_assets)
    else
        b2 = ones(2*n_assets) *-1*trading_cost
    end
    
    #3 0 block and identity [(2*n_assets) x n_sim, I_k], k =2*n_assets
    b3_l = zeros(2*n_assets,(n_sim+1)) #+1 to account for VaR variable
    b3_r = spdiagm(0 => ones(2*n_assets))
    #4 if shorting [[-I_k, I_k], [I_k, -I_k]], k=n_assets, else [[-I_k],[I_k]]
    I_k = spdiagm(0 => ones(n_assets))  # Sparse identity matrix of size k
    if shorting 
        top_block = hcat(-I_k, I_k)
        bottom_block = hcat(I_k, -I_k)
        b4 = vcat(top_block, bottom_block)

        b5_l = hcat(transpose(ones(n_assets)),-transpose(ones(n_assets)),
                    transpose(zeros(n_sim+1)))

        b5_r = trading_cost*transpose(ones(2*n_assets))
    else 
        b4 = vcat(-I_k,I_k)
        b5_l = hcat(transpose(ones(n_assets)),
                    transpose(zeros((n_sim+1)))) # +1 to account for VaR variable
        b5_r = trading_cost*transpose(ones(2*n_assets))


    end
    #5 if shorting [1_k^T, 0_m^T,1_g^T*trading_cost], k = n_assets, g=2*k (1 for positive and 1 for negative differences in new weights)
    
    #left_part = rbind(constraint_matrix,   
    #   cbind(rbind(4,left part of 3)
    #     ),
    #   left_part of 5(until and including the 0's)
    #)
    left_part = vcat(constraint_matrix[1:end-1, :],  #Removing the old weights row which we then replace with the self-financing row
        hcat(
            b4, b3_l
        ),b5_l
    )
    #right_part = rbind(1,2, right_part of 3, right_part of 5)
    right_part = vcat(
        b1,transpose(b2),b3_r,b5_r
    )
    #Final = cbind(left_part,right_part)
    final = hcat(left_part,right_part)
    return final
end

function get_constraint_vector_trading_cost(old_weights, n_assets,n_sim,level,short_ = true)
    b1 = zeros(n_sim)
    b2 = level
    positive_part = old_weights[1:n_assets]
    if short_
        negative_part = old_weights[(n_assets+1):(2*n_assets)]
        b3 = negative_part-positive_part 
    else
        b3 = -positive_part
    end

    b4 = -b3
    b5 = sum(old_weights[1:n_assets])-sum(old_weights[(n_assets+1) : (2*n_assets)]) #Until the general money ammount is done this should always be = 1 for the weights to make sense
    println("Sum of new weights minus transactioncosts must be")
    println(b5)
    b = vcat(b1,b2,b3,b4,b5)
    return b
end

function update_portfolio_weights(old_weights :: Vector{Float64}, scenario :: Matrix{Float64}, risk = 9999, return_ = 9999, short_ = true,trading_costs_=0.02, alpha_ = 0.95)
    if risk == 9999 && return_ == 9999
        println("Must choose either risk level or return level")
        return
    end
    num_assets = size(scenario, 2)
    num_sim = size(scenario, 1)
    n_assets = num_assets
    n_sim = num_sim
    
    if return_ != 9999
        level = return_   
    else
        level = risk
    end
    
    
    #In costvector we do minus mean , plus mean,... since we assume the simulations are losses
    if short_
        constraint_vector = vcat(hcat(scenario,-scenario)*old_weights,level,0)
        cost_vector = [-transpose(mean(scenario, dims=1))+ones(n_assets)*trading_costs_;
                        transpose(mean(scenario, dims=1))+ones(n_assets)*trading_costs_; 0;
                        zeros(num_sim)]
        var_index = 2*num_assets +1
        constraint_matrix = vcat(get_constraint_matrix_min_cvar(scenario, alpha_, short = short_)[1:end-1, :]+vcat(
                -trading_costs_*hcat(ones(n_sim,n_assets),ones(n_sim,n_assets),zeros(n_sim,n_sim+1)), #plus one for the var variable
                transpose(zeros(2*n_assets+1+n_sim))),
                hcat((1+trading_costs_)*transpose(ones(n_assets)),
                (-1+trading_costs_)*transpose(ones(n_assets)),transpose(zeros(1+n_sim)))
                )

    else
        constraint_vector = vcat(scenario*old_weights,level,0)
        cost_vector = [-transpose(mean(scenario, dims=1))+ones(n_assets)*trading_costs_;
                        0;
                        zeros(num_sim)]
        var_index = num_assets+1
        constraint_matrix = get_constraint_matrix_risk_budget(scenario, alpha_, short = short_)+vcat(
                                            -trading_costs_*hcat(ones(n_sim,n_assets),ones(n_sim,n_assets),zeros(n_sim,n_sim+1)), #plus one for the var variable
                                            transpose(zeros(2*n_assets+1+n_sim)),
                                            trading_costs_*hcat(transpose(ones(2*n_assets)),transpose(zeros(1+n_sim)))
                                            )
    end
    #constraint_matrix =constraint_matrix
    cost_vector = constraint_matrix[num_sim+1,1:end]
    m = Model(GLPK.Optimizer)
    num_variables = size(cost_vector,1) # = to num sim + 1 + num_assets*{1 if no shorting, 2 if shorting} + num_assets*2 (last term is to track the changes in portfolio)

    #@variable(m, x[1:num_variables] >= 0)
    @variable(m, x[i=1:num_variables])
    for i in 1:num_variables
        if i != var_index
            @constraint(m, x[i] >= 0)
        end
    end

    @objective(m, Max, dot(cost_vector, x))

    level_index = num_sim+1
    self_fin_index = size(constraint_vector,1)
    
    for i in 1:self_fin_index
        if i == level_index
            if risk != 9999
                @constraint(m, dot(constraint_matrix[i, :], x) <= constraint_vector[i])
            else
                @constraint(m, dot(constraint_matrix[i, :], x) >= constraint_vector[i])
            end
        elseif i == self_fin_index
            @constraint(m, dot(constraint_matrix[i, :], x) == 0 )# constraint_vector[i])
            println("weights_new minus costs must sum to")
            println(constraint_vector[i])
        else
            @constraint(m, dot(constraint_matrix[i, :], x) >= constraint_vector[i])
        end
    end
    optimize!(m)
    updates = JuMP.value.(x)
    new_weights = old_weights + updates[1:n_assets*( short ? 2 : 1)]
    return new_weights , updates

end


end  # End of the module block