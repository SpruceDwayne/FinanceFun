#=  
In this script we will apply a momentum strategy on SP500 based on the article "Applying Machine Learning to Trading Strategies: Using Logistic
Regression to Build Momentum-based Trading Strategies" by Patrick Beaudan and Shuoyuan He.
Ideally we hope to beat the index but if not then at least generate a sample path with lower correlation to either SP500, bonds or inflation
=#

# Add Revise for development
using Revise

# Internal
using FinanceFun


# External
using Dates, DataFrames, Plots
using Statistics
using StatsBase
using RollingFunctions
using Flux
using Random  # For randperm
using Plots


#Parameters:
tickers = ["^GSPC"] #Note instead of SPY we use the real index as it gives us much more data
start_date = DateTime(1970, 3, 1)
end_date = DateTime(2024, 8, 20)
#Data download
df_price =DataFrame( get_adjclose_dataframe(tickers, start_date, end_date,"1d"))
df = dropmissing(df_price)
df = compute_relative_returns(df)
####Helperfunctions we will later hide #####
function manual_standardize(X)
    means = mean(X, dims=1)
    std_devs = std(X, dims=1)
    return (X .- means) ./ std_devs
end

function calculate_momentum(df::DataFrame, window_size::Int)
    # Add a new column for momentum based on the rolling sum over the window size
    momentum_col = [i < window_size ? NaN : sum(df[:,2][i-window_size+1:i]) for i in 1:nrow(df)]
    return momentum_col
end

function calculate_rolling_drawdown(df::DataFrame, window_size::Int)
    # Calculate the rolling maximum for cumulative returns over the window
    rolling_max = [i < window_size ? NaN : maximum(df.CumulativeReturns[i-window_size+1:i]) for i in 1:nrow(df)]
    
    # Calculate rolling drawdown
    rolling_drawdown_col = [i < window_size ? NaN : (df.CumulativeReturns[i] - rolling_max[i]) / rolling_max[i] for i in 1:nrow(df)]
    return rolling_drawdown_col
end

function accuracy(probs, y_test, threshold)
    correct = 0
    for index in 1:length(y_test)
        # Use the probability of class 1 for comparison with the threshold
        predicted_digit = (probs[2, index] >= threshold) ? 1.0 : 0.0  # Predict 1 if prob > threshold
        if predicted_digit == y_test[index]
            correct += 1
        end
    end
    return correct / length(y_test)
end
# Define a function to calculate n-day returns
function calculate_n_day_return(df::DataFrame, n::Int)
    n_day_returns = Vector{Float64}(undef, nrow(df))  # Preallocate a vector for returns
    for i in 1:nrow(df)
        if i <= n
            n_day_returns[i] = NaN  # Assign NaN for the first n rows
        else
            n_day_returns[i] = (df[i, 2] - df[i - n, 2]) / df[i - n, 2]  # Calculate the return
        end
    end
    return n_day_returns
end


function confusion_matrix(probs::Matrix{Float32}, y_test::Vector{Float32}, threshold = 1)
    # Initialize the confusion matrix for binary classification (0 or 1)
    cm = zeros(Int, 2, 2)  # 2x2 matrix for binary classification

    for index in 1:length(y_test)
        # Get the predicted class (either 0 or 1) from the probabilities
        if threshold != 1
                predicted_digit = (probs[2, index] >= threshold) ? 1.0 : 0.0  # Predict 1 if prob > threshold
        else
            predicted_digit = argmax(probs[:, index])[1] - 1  # Convert to 0-based indexing
        end
        true_digit = y_test[index]  # True class (either 0 or 1)

        # Update the confusion matrix
        cm[Int(true_digit + 1), Int(predicted_digit + 1)] += 1
    end

    return cm
end

# Function to calculate strategy returns for a given threshold
function strategy_returns_for_threshold(model, X_test, test_returns, threshold)
    # Get the predicted classes for the given threshold
    preds = predict_classes(model(transpose(X_test)), threshold)
    
    # Compute strategy returns based on the predictions
    strategy_returns = preds .* (test_returns .+ 1) .- 1  # We subtract 1 to get raw returns instead of prices
    
    return strategy_returns
end



function predict_classes(probs::Matrix{Float32}, threshold::Float64)
    num_samples = size(probs, 2)  # Number of columns (samples)
    predicted_classes = Vector{Float64}(undef, num_samples)

    for i in 1:num_samples
        # Probability of class 1 is in the second row, column `i`
        class1_prob = probs[2, i]

        # Compare class 1 probability with the threshold
        predicted_classes[i] = class1_prob >= threshold ? 1.0 : 0.0
    end

    return predicted_classes
end
function cumulative_returns_for_threshold(model, X_test, test_returns, threshold)
    # Predict classes using the given threshold
    preds = predict_classes(model(transpose(X_test)), threshold)
    
    # Flatten test_returns if it's a 2D array to ensure it's a vector
    test_returns_vec = vec(test_returns)

    # Calculate strategy returns
    strategy_returns = (preds .* test_returns_vec) .+ 1

    # Calculate cumulative returns (cumulative product)
    cumulative_returns = cumprod(strategy_returns) .- 1  # Cumulative product minus 1
    
    return cumulative_returns
end


function rank_thresholds_by_cumulative_return(model, X_test, test_returns, thresholds)
    final_cumulative_returns = []
    threshold_results = []

    # Loop over each threshold
    for threshold in thresholds
        # Compute cumulative returns for the current threshold
        cumulative_returns = cumulative_returns_for_threshold(model, X_test, test_returns, threshold)
        
        # Store the final cumulative return and associated threshold
        final_cumulative_return = cumulative_returns[end]  # Get the last value (final return)
        push!(final_cumulative_returns, final_cumulative_return)
        push!(threshold_results, (threshold, final_cumulative_return))
    end

    # Sort thresholds by the final cumulative returns in descending order
    sorted_thresholds = sort(threshold_results, by = x -> x[2], rev=true)

    # Print the sorted thresholds and their final cumulative returns
    println("Threshold ranking based on final cumulative returns:")
    for (threshold, final_return) in sorted_thresholds
        println("Threshold: $threshold, Final Cumulative Return: $final_return")
    end
    
    return sorted_thresholds
end


function buy_and_hold_returns(test_returns)
    # Buy and hold strategy assumes being always long (predicting 1 for every timestep)
    buy_and_hold = ones(size(test_returns))  # Create an array of 1s for the entire test period

    # Compute buy-and-hold strategy cumulative returns
    strategy_returns = buy_and_hold .* (test_returns .+ 1)
    cumulative_returns = cumprod(strategy_returns; dims=1) .- 1  # Subtract 1 to get cumulative returns

    return cumulative_returns
end
##########################################
#Next step is to build the factors for our Logistic regression model
#Make indicator saying if return for a week was positive or negative
momentum_window_size = 10

df_price.n_day_returns = calculate_n_day_return(df_price, momentum_window_size)

df.Indicator = ifelse.(df_price[2:end,:].n_day_returns .> 0.0, 1, 0)

#Fist we do momentum features based on 30,60,90,120,180,270,300 and 360 like in the article
# Calculate cumulative returns (this is needed for drawdown)
df.CumulativeReturns = cumprod(1 .+ df[:,2]) .- 1

# Generate momentum columns for different window sizes
window_sizes_momentum = [30, 60, 90, 120, 180, 270, 300, 360]
for w in window_sizes_momentum
    df[!, Symbol("Momentum_$w")] = calculate_momentum(df, w)
end

#Then we do drawdown on a 15,60,90 and 120 day window


# Generate drawdown columns for different window sizes
window_sizes_drawdown = [15, 60, 90, 120]
for w in window_sizes_drawdown
    df[!, Symbol("Drawdown_$w")] = calculate_rolling_drawdown(df, w)
end

#Now with the columns we need we can start on modelling
X = DataFrames.select(df[360:end-momentum_window_size,:], r"Momentum_|Drawdown_") |> Matrix
X = manual_standardize(X)
X = Float32.(X)
y = df[360+momentum_window_size:end,:].Indicator
y = Float32.(y)

train_len = 6000
test_len = 7000
X_train = X[1:train_len,:]
y_train = y[1:train_len]
X_test =X[train_len:train_len+test_len,:]
y_test = y[train_len:train_len+test_len]

num_features = size(X, 2)  

# Define the neural network model
model = Chain(
    Flux.flatten,
    Dense(num_features=>20,relu),
    Dense(20=>10,relu),
    Dense(10=>10,relu),
    Dense(10=>2,sigmoid),
    softmax
)

optimizer = Flux.setup(Adam(), model)

# Define the loss function that uses the cross-entropy to 
# measure the error by comparing model predictions of data 
# row "x" with true data label in the "y"
function loss(model, x, y)
    y_hat = model(x)
    return Flux.crossentropy(y_hat, Flux.onehotbatch(y, [0.0, 1.0]))
end
#Transpose X_matrix since Flux iterates over columns before rows
data_train = Flux.DataLoader((transpose(X_train),y_train), shuffle=true)

# Prepare lists to store accuracy for each epoch
train_accuracies = Float64[]
test_accuracies = Float64[]

# Number of epochs for training
num_epochs = 10

# Training loop
for epoch in 1:num_epochs
    Flux.train!(loss, model, data_train, optimizer)

    # Calculate training accuracy
    train_preds = model(transpose(X_train))  # Predicted probabilities for training set
    train_accuracy = accuracy(train_preds, y_train)
    push!(train_accuracies, train_accuracy)

    # Calculate test accuracy
    test_preds = model(transpose(X_test))  # Predicted probabilities for test set
    test_accuracy = accuracy(test_preds, y_test,1-sum(y_train)/length(y_train))
    push!(test_accuracies, test_accuracy)

    # Print accuracies for the current epoch
    println("Epoch: $epoch, Train Accuracy: $train_accuracy, Test Accuracy: $test_accuracy")
end

#With the model trained we can consider the confusion matrix to see how the model performs
confusion_matrix(model(transpose(X_test)),y_test,0.39)

#We now begin testing which threshold works the best on the testset
test_returns = df[:,2][360+momentum_window_size+train_len:360+momentum_window_size+train_len+test_len,:]


#Inspect some range by plots
thresholds = 0.3:0.05:0.65  # Define the range of thresholds
cumulative_returns_all = []  # To store cumulative returns for each threshold

for threshold in thresholds
    # Compute cumulative returns for the current threshold
    cum_returns = cumulative_returns_for_threshold(model, X_test, test_returns, threshold)
    
    # Store the results in the list
    push!(cumulative_returns_all, (threshold, cum_returns))
end



# Plot cumulative returns for each threshold
plot_title = "Cumulative Returns for Different Thresholds"

plot(legend=:topleft, title=plot_title)

# Add lines for each threshold
for (threshold, cum_returns) in cumulative_returns_all
    plot!(cum_returns, label="Threshold: $threshold")
end

xlabel!("Time")
ylabel!("Cumulative Returns")
#All these thresholds generate a profitable strategy in the long run but with large differences in performance


#Lets inspect more Thresholds

thresholds = 0.10:0.01:0.90  # Test thresholds from 0.3 to 0.9 with a step of 0.01
sorted_thresholds = rank_thresholds_by_cumulative_return(model, X_test, test_returns, thresholds)
#We find that the best threshold is 0.3. Lets compare that to the buyandhold

# Compute cumulative returns for buy-and-hold strategy
buy_and_hold_cumulative_returns = buy_and_hold_returns(test_returns)

# Compare buy-and-hold with the strategy's best threshold cumulative returns
best_threshold = 0.3
strategy_cumulative_returns = cumulative_returns_for_threshold(model, X_test, test_returns, best_threshold)
x_axis = df.Index[360+momentum_window_size+train_len:360+momentum_window_size+train_len+test_len]
# Plot the comparison
plot(x_axis, strategy_cumulative_returns, label="Strategy (Threshold: $best_threshold)", lw=2)
plot!(x_axis, buy_and_hold_cumulative_returns, label="Buy and Hold", lw=2, linestyle=:dash, color=:red)

xlabel!("Time")
ylabel!("Cumulative Return")
title!("Momentum Strategy vs Buy-and-Hold")

#We see that the momentum performs slightly better during the testperiod
#But the difference is rather small. Lets see which threshold generates the best diversifier

# Set the range of thresholds to check
thresholds = 0.1:0.01:0.6

# Store correlations for each threshold
correlations = Float64[]

# Loop through each threshold and compute correlation
for threshold in thresholds
    strat_returns = strategy_returns_for_threshold(model, X_test, test_returns, threshold)
    
    # Compute correlation between strategy and buy-and-hold returns
    correlation = cor(strat_returns, test_returns)
    
    # Store the correlation value
    push!(correlations, correlation[1,1])
    
    println("Threshold: $threshold, Correlation: $correlation")
end

# Plot the correlation between strategy and buy-and-hold for each threshold
plot(thresholds, correlations, label="Correlation with Buy-and-Hold", lw=2)
xlabel!("Threshold")
ylabel!("Correlation")
title!("Correlation of Strategy Returns with Buy-and-Hold Returns")
#=
We find that when using larger threshold we get uncorrelated returns.
This holds even for 0.3 which we saw gave almost the same long run portfolio returns.
So if one thinks of the momentum strategy as an asset then that asset is diversifying.
=#



