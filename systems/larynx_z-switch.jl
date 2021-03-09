

################################################################################
## Problem Definition -- z-switch model for bird larynx

struct z_switch{T,P,F} <: AbstractSystem
    u₀::T
    p::T
    prob::P
    transform::F

    function z_switch()
        # Default parameters and initial conditions
        u₀ = Float32[1.0, 1.0]
        p = Float32[1., 1., 1., 1., 1., 1., 0.01]
        tspan = (0.f0, 1.f0)

        # Define differential equations
        function f!(du, u, p, t)
                y, z = u
                a, b, c, m, zₒ, α, ϵ = p
                du[1] = a*z
                du[2] = (zₒ + α*y - z)*(-zₒ + α*y - z)*(m*y + b - z)/ϵ - c*z
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
