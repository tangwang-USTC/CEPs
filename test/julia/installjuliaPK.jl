
using Pkg

pknames1 = ["Plots","PyPlot","DataFrames","CSV","Format","Parsers","BenchmarkTools","OhMyREPL"]  # Plots, Datas, files, System
pknames2 = ["MutableArithmetics","KahanSummation"]  # Arithmetics
pknames3 = ["AssociatedLegendrePolynomials","ChebyshevApprox","FastTransforms","SpecialFunctions","HypergeometricFunctions","SavitzkyGolay"]  # Special Functions
pknames4 = ["GaussQuadrature","FastGaussQuadrature","QuadGK","Romberg","Trapz","NumericalIntegration","ForwardDiff","FiniteDifferences"]   # Integrals, Differences
pknames5 = ["LinearAlgebra","LinearAlgebraX","ToeplitzMatrices"]  # Linear Algebra
pknames6 = ["OrdinaryDiffEq","DifferentialEquations","DifferentialEquations","DiffEqCallbacks","IterativeSolvers","NLsolve"] # PDE, ODE
pknames7 = ["Loess","Dierckx","DataInterpolations","SmoothingSplines"]  # Interpolations
pknames8 = ["Optim","Optimization","Optimisers","LeastSquaresOptim","Ipopt","LsqFit","DeconvOptim","OptimizationProblems","HiGHS","Evolutionary","Convex"] # Optimisers
pknames9 = ["JuMP","Flux","Zygote","ModelingToolkit","Plasmo","MathOptInterface"]   # Optimization
pknames10 = ["StableRNGs","Distances","DataDrivenDiffEq"]
pknames11 = ["CUDA","Metalhead","Clustering"]  # Parallel
pknames12 = ["Latexify","LaTeXStrings"]  # LaTeX
pknames = [pknames1; pknames2; pknames3;pknames4; pknames5;pknames6;pknames7;pknames8;pknames9;pknames10]
pknames = [pknames; pknames11; pknames12]
# pknames = [pknames9]
pknames = ["MutableArithmetics"]
for pkname in pknames
    import Pkg; Pkg.add(pkname)
end

using Plots,PyPlot,DataFrames,CSV,Format,OhMyREPL,Parsers

# failure

import Pkg; Pkg.add("DiffEqCallbacks")
# import Pkg; Pkg.add("FastKmeans") 


["BayesianOptimization"]

