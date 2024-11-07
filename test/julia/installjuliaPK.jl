
using Pkg

pknames1 = ["Plots","PyPlot","DataFrames","CSV","Format","OhMyREPL","Parsers"]
pknames2 = ["Latexify","LaTeXStrings","Clustering","BenchmarkTools"]  # LaTeXStrings
pknames3 = ["AssociatedLegendrePolynomials","ChebyshevApprox","FastTransforms","SpecialFunctions","HypergeometricFunctions","SavitzkyGolay"]
pknames4 = ["GaussQuadrature","FastGaussQuadrature","QuadGK","Romberg","Trapz","KahanSummation"]
pknames5 = ["LinearAlgebra","LinearAlgebraX","ForwardDiff","FiniteDifferences"]
pknames6 = ["OrdinaryDiffEq","DifferentialEquations","DifferentialEquations","DiffEqCallbacks","IterativeSolvers","NLsolve"]
pknames7 = ["Loess","Dierckx","DataInterpolations","NumericalIntegration","SmoothingSplines"]
pknames8 = ["Optim","Optimization","Optimisers","LeastSquaresOptim","Ipopt","LsqFit","DeconvOptim","OptimizationProblems","HiGHS","Evolutionary"]
pknames9 = ["JuMP","Flux","Zygote","ModelingToolkit","Plasmo","MathOptInterface"]
pknames10 = ["StableRNGs","Distances","DataDrivenDiffEq"]
pknames11 = ["CUDA","Metalhead"]  # Para
pknames12 = ["ToeplitzMatrices"]  # Array
pknames = [pknames1; pknames2; pknames3;pknames4; pknames5;pknames6;pknames7;pknames8;pknames9;pknames10]
pknames = [pknames; pknames11; pknames12]
# pknames = [pknames9]
pknames = ["Evolutionary"]
for pkname in pknames
    import Pkg; Pkg.add(pkname)
end

using Plots,PyPlot,DataFrames,CSV,Format,OhMyREPL,Parsers

# failure

import Pkg; Pkg.add("DiffEqCallbacks")
# import Pkg; Pkg.add("FastKmeans") 


["BayesianOptimization"]

