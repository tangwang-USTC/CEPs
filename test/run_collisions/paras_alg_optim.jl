
const nhMax = 1e8       # nₑ = 10n₂₀  ~  nfuel = 10⁶n₂₀; nEP = 10¹³. The maximum absolute span of the submoments;
const uhMax = 100.0     # uh ≤ 3.0    ~  L_limit = 45. The maximum absolute span of the submoments;
const uhMin = 1e-4
const vhthMax = 50.0    # TD = 1 keV  ~  Tα = 3 MeV. The maximum absolute span of the submoments;
const vhthMin = 1e-6    # TD = 1 keV  ~  Tα = 3 MeV. The maximum absolute span of the submoments;
const vhthInitial = 4e-1# vthi
const nhInitial = 2e-1  # nai

## Parameters: Optimization solver
## Algorithm `King`: Optimizations for `fvL`, `jMsmax = ℓ + dj * (3(nMod-1) + (3-1))`.

# [:LeastSquaresOptim, :NL_solve, :JuMP]
NL_solve = :LeastSquaresOptim             #  More stability
# NL_solve = :NLsolve

if NL_solve == :LeastSquaresOptim
    NL_solve_method = :trust_region    # [:trust_region, :newton, :anderson]
elseif NL_solve == :NLsolve
    NL_solve_method = :trust_region    # [:trust_region, :newton, :anderson]
    # NL_solve_method = :newton        # always to be falure.
else
end 
is_Jacobian = true      # (=true, default) Whether Jacobian matrix will be used to improve the performance of the optimizations.
show_trace = false      # (=false, default) 

# NL_solve == :LeastSquaresOptim
(optimizer, optimizer_abbr) = (LeastSquaresOptim.Dogleg, :dl)  # More stability than `:lm`
(optimizer, optimizer_abbr) = (LeastSquaresOptim.LevenbergMarquardt, :lm)

factorMethod = :QR     # factorMethod = [:QR, :Cholesky, :LSMR]    # `:QR` is more stability, maybe not conservative
# factorMethod = :Cholesky
(factor, factor_abbr) = (LeastSquaresOptim.QR(), :QR)     # `=QR(), default`  # More stability, maybe not conservative
factor = LeastSquaresOptim.Cholesky()
factor = LeastSquaresOptim.LSMR()

# autodiff = :forward  #
autodiff = :central    # A little more complicated but maybe obtaining no benefits.


# ########## The initial solution noises
maxIterKing = 500       # (=200 default) The maximum inner iteration number for King's optimization
p_tol = epsT / 1000           # (= 1e-13, default)
g_tol = epsT / 1000 
f_tol = epsT / 1000 
