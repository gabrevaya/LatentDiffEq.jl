

################################################################################
## Problem Definition -- frictionless pendulum

struct pendulum{T,P,F} <: AbstractSystem
    u₀::T
    p::T
    prob::P
    transform::F

    function pendulum()
        # Default parameters and initial conditions
        u₀ = Float32[1.0, 1.0]
        p = Float32[1.]
        tspan = (0.f0, 1.f0)

        # Define differential equations
        function f!(du, u, p, t)
                x, y = u
                G = 10.0f0
                L = p[1]
                
                du[1] = y
                du[2] =  -G/L*sin(x)
        end

        output_transform(u) = u

        # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob = ODEProblem(ODEFunc, u₀, tspan, p)

        T = typeof(u₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(u₀, p, prob, output_transform)
    end
    
end