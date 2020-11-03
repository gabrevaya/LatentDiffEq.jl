

################################################################################
## Problem Definition -- Stochastic Lotka-Volterra
using StochasticDiffEq, DiffEqSensitivity

struct SLV{T,P} <: AbstractSystem
    u₀::T
    p::T
    prob::P

    function SLV()
        # Default parameters and initial conditions
        u₀ = Float32[1.0, 1.0]
        # p = Float32[1.25, 1.5, 1.75, 2]
        p = Float32[1.5,1.0,3.0,1.0]#,0.3,0.3]
        tspan = (0.f0, 1.f0)

        # Define differential equations
        function f!(du, u, p, t)
                x, y = u
                α, β, δ, γ = p
                du[1] = α*x - β*x*y
                du[2] = -δ*y + γ*x*y
        end


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
        prob = create_prob("Stochastic_Lotka-Volterra", 1, sys, u₀, tspan, p)

        # function σ(du,u,p,t)
        #     du .= 0.001f0
        # end

        function σ(du,u,p,t)
            du .= 0.1f0*u
        end

        prob_sde = SDEProblem(prob.f.f, σ, prob.u0, prob.tspan, prob.p, jac = prob.f.jac, tgrad = prob.f.tgrad)


        T = typeof(u₀)
        P = typeof(prob_sde)
        new{T,P}(u₀, p, prob_sde)
    end
end
