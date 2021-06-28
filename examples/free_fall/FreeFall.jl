

################################################################################
## Problem Definition -- Free fall

struct FreeFall{P,S,T}

    prob::P
    solver::S
    sensealg::T

    function FreeFall()
        # Default parameters and initial conditions
        u₀ = Float32[10.0, 0.0]
        p = Float32[10.0]
        tspan = (0.f0, 1.f0)

        # Define differential equations
        function f!(du, u, p, t)
                x, v = u
                g = p[1] #changed
                du[1] = v
                du[2] = g
                # @info "du: $(du), $(du[1]), $(du[2]), u: $u"
        end

        # Build ODE Problem

        _prob = ODEProblem(f!, u₀, tspan, p)

        @info "Optimizing ODE Problem"

        sys = modelingtoolkitize(_prob)
        ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob = ODEProblem(ODEFunc, u₀, tspan, p)
        
        solver = Tsit5()
        sensalg = ForwardDiffSensitivity()
        # https://github.com/SciML/DiffEqFlux.jl/blob/master/docs/src/ControllingAdjoints.md

        P = typeof(prob)
        S = typeof(solver)
        T = typeof(sensalg)
        new{P,S,T}(prob, solver, sensalg)
    end
    
end


