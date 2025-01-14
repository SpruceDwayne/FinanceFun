This is a free time project I do to practice/learn Julia and explore various interest within quantitative finance. Progress varies according to how busy I am IRL but hopefully over time the amount of financefun will build up.


# Notes
To start a Julia REPL inside the folder. There you enter package mode to activate & instantiate the project. Write `]` followed by `activate FinanceFun` and `instantiate`. It should then read the Manifest.toml and Project.toml file to initialise the Julia project named FinanceFun. These files track the package dependencies.

Goal: use and develop this package in one of two ways. Either, activate the project, or `dev` the project to use in another Julia project e.g. your base project '(@1.1x)'.  See https://pkgdocs.julialang.org/v1/managing-packages/ & https://pkgdocs.julialang.org/v1/environments/.

# TODO
CVaR parts:
3. Index tracking

Modelling parts:
1. GAN
2. VAE
3. Garch Multivariate
4. Copula modelling, including garch modelling of t copula correlation
5. Bayseian modelling

Trading parts:
1. General backtesting function
2. Risk-target strat
3. Risk-return ratio strat
4. Momentum strat
5. Mean reversion strat
6. Index tracking strat
7. Trend following

Plot functions
1. Dependence plots like correlation
2. Scatter plots
3. Dashboards for portfolio optimisation
    -Page for initial portfolio
    -Page for portfolio updates
    -Being able to choose list of tickrs seached on tickrs file
    -Being able to choose modelling methods
    -Being able to visualize efficient frontier
    -Being able to choose risk measures like CVaR, VaR, variance and Perhaps also max drawdown
