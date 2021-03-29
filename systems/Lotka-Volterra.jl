

################################################################################
## Problem Definition -- Lotka-Volterra

struct LV{P,S,T}

    prob::P
    solver::S
    sensealg::T

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

        # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob = ODEProblem(ODEFunc, u₀, tspan, p)
        
        solver = Tsit5()
        sensalg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))

        P = typeof(prob)
        S = typeof(solver)
        T = typeof(sensalg)
        new{P,S,T}(prob, solver, sensalg)
    end
    
end



################################################################################
## Problem Definition -- Stochastic Lotka-Volterra
using StochasticDiffEq, DiffEqSensitivity

struct SLV{P,S,T}

    prob::P
    solver::S
    sensealg::T

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
        sys = modelingtoolkitize(_prob)
        ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob = ODEProblem(ODEFunc, u₀, tspan, p)
        
        function σ(du,u,p,t)
            du .= 0.1f0*u
        end

        prob_sde = SDEProblem(prob.f.f, σ, prob.u0, prob.tspan, prob.p, jac = prob.f.jac, tgrad = prob.f.tgrad)

        solver = SOSRI()
        sensalg = ForwardDiffSensitivity()

        P = typeof(prob_sde)
        S = typeof(solver)
        T = typeof(sensalg)
        new{P,S,T}(prob_sde, solver, sensalg)
    end
end
