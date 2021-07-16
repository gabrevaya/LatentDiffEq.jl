using DynamicalSystems

################################################################################
## Problem Definition -- double pendulum

struct DoublePendulum{P,S,T}

    prob::P
    solver::S
    sensealg::T

    function DoublePendulum()
        ds = Systems.double_pendulum()

        u₀ = [ds.u0[i] for i in 1:length(ds.u0)]
        p = ds.p
        u₀ = Float32.(u₀)
        p = Float32.(p)
        tspan = (0.f0, 1.f0)

        # Build ODE Problem
        _prob = ODEProblem(ds.f, u₀, tspan, p)

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