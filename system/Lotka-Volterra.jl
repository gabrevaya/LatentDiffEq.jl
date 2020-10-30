

################################################################################
## Problem Definition -- Lotka-Volterra

struct LV{T,P} <: AbstractSystem
    u₀::T
    p::T
    prob::P
    transform

    function LV()
        # Default parameters and initial conditions
        u₀ = Float32[1.0, 1.0]
        p = Float32[1.25, 1.5, 1.75, 2]
        tspan = (0.f0, 1.f0)

        # Define differential equations
        function f!(du, u, p, t)
                x, y = u
                α, β, δ, γ = p
                du[1] = α*x - β*x*y
                du[2] = -δ*y + γ*x*y
        end

        output_transform(u) = u

        # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        @info "Optimizing ODE Problem"
        # prob,_ = auto_optimize(_prob, verbose = false, static = false)
        sys = modelingtoolkitize(_prob)
        # prob = ODEProblem(sys,_prob.u0,_prob.tspan,_prob.p,
        #                        jac = true, tgrad = true, simplify = true,
        #                        sparse = false,
        #                        parallel = ModelingToolkit.SerialForm(),
        #                        eval_expression = false)
        prob = create_prob("Lotka-Volterra", 1, sys, u₀, tspan, p)

        T = typeof(u₀)
        P = typeof(prob)
        new{T,P}(u₀, p, prob, output_transform)
    end
    
end
