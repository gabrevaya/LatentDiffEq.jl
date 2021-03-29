

################################################################################
## Problem Definition -- z-switch model for bird larynx

struct z_switch{P,S,T}

    prob::P
    solver::S
    sensealg::T

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
