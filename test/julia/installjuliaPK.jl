
using Pkg

pknames1 = ["Plots","PyPlot","DataFrames","CSV","Format","Parsers","BenchmarkTools","OhMyREPL"]  # Plots, Datas, files, System
pknames2 = ["MutableArithmetics","KahanSummation"]  # Arithmetics
pknames3 = ["AssociatedLegendrePolynomials","ChebyshevApprox","FastTransforms","SpecialFunctions","HypergeometricFunctions","SavitzkyGolay"]  # Special Functions
pknames4 = ["GaussQuadrature","FastGaussQuadrature","QuadGK","Romberg","Trapz","NumericalIntegration","ForwardDiff","ReverseDiff","Zygote","FiniteDifferences","FiniteDiff"]   # Integrals, Differences, AD
pknames5 = ["LinearAlgebra","LinearAlgebraX","ToeplitzMatrices","StaticArrays","IncompleteLU","AlgebraicMultigrid"]  # Linear Algebra
pknames6 = ["OrdinaryDiffEq","DifferentialEquations","DiffEqCallbacks","IRKGaussLegendre"] # PDE, ODE
pknames62= ["FEniCS"]  # FVM/FEM
pknames7 = ["LinearSolve","Krylov","IterativeSolvers","KrylovKit"]  # Linear solver
pknames72= ["NonlinearSolve","JuMP","InfiniteOpt","NLPModels","NLPModelsJuMP","NLsolve","CaNNOLeS","JSOSolvers","MadNLP"]          # Nonlinear solver
pknames8 = ["Optimization","Optim","NLPModelsModifiers","OptimizationEvolutionary","LeastSquaresOptim","OptimizationOptimJL","OptimizationMOI","Optimisers","SteadyStateDiffEq","Ipopt","LsqFit","DeconvOptim","OptimizationProblems","HiGHS","Evolutionary","Convex"] # Optimisers
pknames9 = ["ModelingToolkit","NeuralPDE","Lux","Flux","Zygote","Plasmo","MathOptInterface"]# ["DiffEqFlux"]   # ML/PINN
pknames92 = ["ADTypes","DifferentiationInterface","SparseDiffTools","Symbolics"]        # AD: Automatic Differentiation
pknames10 = ["StableRNGs","Distances","DataDrivenDiffEq"]
pknames11 = ["Loess","Dierckx","DataInterpolations","SmoothingSplines"]  # Interpolations
pknames12 = ["CUDA","Metalhead","Clustering","PreallocationTools"]  # Parallel
pknames13 = ["Latexify","LaTeXStrings","SafeTestsets","BenchmarkTools","TimerOutputs"]  # LaTeX
pknames = [pknames1; pknames2; pknames3;pknames4; pknames5;pknames6;pknames7;pknames72;pknames8;pknames9;pknames92;pknames10]
pknames = [pknames8;pknames9;pknames92;pknames10]
pknames = [pknames; pknames11; pknames12; pknames13]
# pknames = [pknames9]
# pknames = ["InfiniteOpt"]
for pkname in pknames
    import Pkg; Pkg.add(pkname)
end

# "NonlinearSolve" 与 "DifferentialEquations", "Functors" 等可能有版本冲突。 前者需要降低版本（julia 1.11.2）。
 
# ["DiffEqFlux"]

using Plots,PyPlot,DataFrames,CSV,Format,OhMyREPL,Parsers

# failure

# import Pkg; Pkg.add("FastKmeans") 


# pkgname_rm = ["BayesianOptimization","SciMLSensitivity","DiffEqFlux","Tracker","Enzyme","SafeTestsets"]
# for pkname in pknames
#     import Pkg; Pkg.rm(pkgname_rm)
# end

import Pkg; 
if 11 == 1
    Pkg.rm("Tracker")               # Replaced by `Zygote.jl`
    Pkg.rm("Enzyme")                # 些微降低`CUDA`、`OrdinaryDiffEq`及其相关库的版本
    Pkg.rm("DiffEqFlux")            # 些微降低`CUDA`、`OrdinaryDiffEq`及其相关库的版本
    Pkg.rm("SafeTestsets")          # 降低大量库的版本，避免对其安装以及使用
    Pkg.rm("SciMLSensitivity")
    Pkg.rm("BayesianOptimization")
    Pkg.update()    
end


# ## AD
# ChainRulesCore.jl
# Enzyme.jl
# FastDifferentiation.jl
# FiniteDiff.jl
# FiniteDifferences.jl
# ForwardDiff.jl
# Mooncake.jl
# PolyesterForwardDiff.jl
# ReverseDiff.jl
# Symbolics.jl
# Tracker.jl
# Zygote.jl


# ENV["CPLEX_STUDIO_BINARIES"] = "C:\\Program Files\\IBM\\ILOG\\CPLEX_Studio221\\cplex\\bin\\x64_win64\\"
# import Pkg
# Pkg.add("CPLEX")
# Pkg.build("CPLEX")