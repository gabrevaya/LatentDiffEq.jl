

################################################################################
## Problem Definition -- Lotka-Volterra

struct LV{T,P,F} <: AbstractSystem
    u₀::T
    p::T
    prob::P
    transform::F

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
        sys = modelingtoolkitize(_prob)
        prob = create_prob("Lotka-Volterra", 1, sys, u₀, tspan, p)

        T = typeof(u₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(u₀, p, prob, output_transform)
    end
    
end
