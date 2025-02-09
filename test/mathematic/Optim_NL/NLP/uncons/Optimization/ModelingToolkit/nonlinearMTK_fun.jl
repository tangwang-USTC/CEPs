


using ModelingToolkit, NonlinearSolve

function nonlinearMIT()

    # Define a nonlinear system
    @variables x y z xx[1:3]
    @parameters σ ρ β
    
    yx = y- x
    
    eqs3 = [0 ~ σ * (y - x)
           0 ~ x * (ρ - z) - y
           0 ~ x * y - β * z]
    
    eqs3 = Vector{Equation}(undef,3)
    eqs3[1] = 0 ~ σ * yx
    eqs3[2] = 0 ~ x * (ρ - z) - y
    eqs3[3] = 0 ~ x * y - β * z
    
    eqs3 = Vector{Equation}(undef,3)
    # aaaa = Symbolics.Arr{Num, 1}
    function eqsf(eqs3::Vector{Equation},nn::Int)
        
        eqs3[1] = 0 ~ σ * yx
        eqs3[2] = 0 ~ x * (ρ - z) - y
        eqs3[3] = 0 ~ x * y - β * z
        return eqs3
    end
    eqs3 = eqsf(eqs3,3)
    
    eqs3 = Vector{Equation}(undef,3)
    function eqsf!(eqs3::Vector{Equation},nn::Int)
        
        eqs3[1] = 0 ~ σ * yx
        a = x * (ρ - z) 
        a -= y
        # @show typeof(a)
        eqs3[2] = 0 ~ a
        eqs3[3] = 0 ~ x * y - β * z
    end
    eqsf!(eqs3,3)
    
    # function eqsf!(eqs3::Vector{Equation},nn::Int)
        
    #     eqs3[1] = 0 ~ σ * (xx[2] - xx[1])
    #     a = xx[1] * (ρ - xx[3]) 
    #     a -= xx[2]
    #     # @show typeof(a)
    #     eqs3[2] = 0 ~ a
    #     eqs3[3] = 0 ~ xx[1] * xx[2] - β * xx[3]
    # end
    # eqsf!(eqs3,3)
    
    @mtkbuild ns = NonlinearSystem(eqs3)
    
    guesses = [x => 1.0, y => 0.0, z => 0.0]
    # guesses = [xx[1] => 1.0, xx[2] => 0.0, xx[3] => 0.0]
    
    ps = [σ => 10.0, ρ => 26.0, β => 8 / 3]
    
    prob = NonlinearProblem(ns, guesses, ps)
    sol = solve(prob, NewtonRaphson())
    return sol    

end


nonlinearMIT()
