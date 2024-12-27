
using NLsolve

is_JJJ = false
is_JJJ = true
x0 = [ 0.1, 1.2]
# x0 = -[ 2.0, 2.0]
# ubs = [Inf, Inf]
lbs = [-Inf, -Inf]
ubs = [2.0, 2.0]
lbs = [-0.1, -1.2]
if is_JJJ 
    function f!(F, x)
        F[1] = (x[1]+3)*(x[2]^3-7)+18
        F[2] = sin(x[2]*exp(x[1])-1)
    end
    
    function j!(J, x)
        J[1, 1] = x[2]^3-7
        J[1, 2] = 3*x[2]^2*(x[1]+3)
        u = exp(x[1])*cos(x[2]*exp(x[1])-1)
        J[2, 1] = x[2]*u
        J[2, 2] = u
    end
    
    # sol = nlsolve(f!, j!, x0)
    # sol = nlsolve(f!, j!, x0; autodiff=:forward)
    sol = nlsolve(f!, j!, x0;method=:trust_region)      # default
    # sol = nlsolve(f!, j!, x0;method=:newton)
    # sol = nlsolve(f!, j!, x0;method=:anderson)
    


    # sol = mcpsolve(f!, x0, ubs, lbs;method=:trust_region,reformulation=:smooth,autodiff=:forward)

    # df = OnceDifferentiable(f!, j!, x0, initial_F)
    # df = OnceDifferentiable(f!, j!, fj!, x0, initial_F)
    # df = OnceDifferentiable(fj!, x0, initial_F)
    # sol = nlsolve(df, x0)
else
    function f(x)
        [(x[1]+3)*(x[2]^3-7)+18,
        sin(x[2]*exp(x[1])-1)]
    end
    
    sol = nlsolve(f, [ 0.1, 1.2])


    function f!(F, x)
        F[1]=3*x[1]^2+2*x[1]*x[2]+2*x[2]^2+x[3]+3*x[4]-6
        F[2]=2*x[1]^2+x[1]+x[2]^2+3*x[3]+2*x[4]-2
        F[3]=3*x[1]^2+x[1]*x[2]+2*x[2]^2+2*x[3]+3*x[4]-1
        F[4]=x[1]^2+3*x[2]^2+2*x[3]+3*x[4]-3
    end
    
    sol = mcpsolve(f!, [0., 0., 0., 0.], [1.5, 0.1, 0.1, 1.0],
                 -[1.5, 0.1, 0.1, 1.0], reformulation = :smooth, autodiff = :forward)
    1
end


sol.zero



