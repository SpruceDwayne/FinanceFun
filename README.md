# Notes for Bastian
Thanks for pulling. Welcome to your Julia project FinanceFun.

To work properly, we need to rename the repo to FinanceFun (without -), so that Julia accepts it as a valid project. After renaming follow https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository to make pushing & pulling consistent with the new project name and URL.

You need to start a Julia REPL inside the folder. There you enter package mode to activate & instantiate the project. Write `]` followed by `activate FinanceFun` and `instantiate`. It should then read the Manifest.toml and Project.toml file to initialise the Julia project named FinanceFun. These files track the package dependencies.

From now on, you can use and develop this package in two ways. Either, you activate the project, or you `dev` the project to use in another Julia project e.g. your base project '(@1.1x)'.  See https://pkgdocs.julialang.org/v1/managing-packages/ & https://pkgdocs.julialang.org/v1/environments/.

As you can see, the folder structure is a bit cleaner now. Any source code is located in `src`, data in `data`, and scripts/examples/tests in `test`.

In `src/FinanceFun.jl` you see the main module script. Here it shows how you import exports from submodules and make them visible to importers of the main module. Note also the CamelCase naming convention common to Julia.

In `test/marcustest.jl` I have tried to recreate some of your scripts using the new module structure. Notice how much cleaner it is with regards to imports.

If you need to add more package dependencies, you need to activate the project before writing `add Package`. 

Lastly, I notice in many of your functions you give arguments default values. It would make for nicer function signatures if you make these optional key-value arguments by writing `function foo(bar1, bar2; baroptional1=100, baroptional2=200)`. This allows you to call both
`foo(bar1, bar2)`, `foo(bar1, bar2; baroptional1=100)` & `foo(bar1, bar2; baroptional2=100)`, and of course also `foo(bar1, bar2; baroptional1=100)`.

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
