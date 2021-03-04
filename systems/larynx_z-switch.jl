

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
        # prob,_ = auto_optimize(_prob, verbose = false, static = false)
        sys = modelingtoolkitize(_prob)
        # prob = ODEProblem(sys,_prob.u0,_prob.tspan,_prob.p,
        #                        jac = true, tgrad = true, simplify = true,
        #                        sparse = false,
        #                        parallel = ModelingToolkit.SerialForm(),
        #                        eval_expression = false)
        prob = create_prob("z-switch", 1, sys, u₀, tspan, p)

        T = typeof(u₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(u₀, p, prob, output_transform)
    end
    
end
