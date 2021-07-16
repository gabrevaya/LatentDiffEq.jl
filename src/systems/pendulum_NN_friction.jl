

################################################################################
## Problem Definition -- "pendulum" with NN inside
using DiffEqFlux

struct Pendulum_NN_friction{P,S,T}

    prob::P
    solver::S
    sensealg::T

    function Pendulum_NN_friction()
        # Default parameters and initial conditions
        u₀ = Float32[1.0, 1.0]
        p_ODE = Float32[1.]
        tspan = (0.f0, 1.f0)

        model_univ = FastChain(FastDense(2, 10, tanh),
                       FastDense(10, 10, tanh),
                       FastDense(10, 1))

        # The model weights are destructured into a vector of parameters
        p_NN = initial_params(model_univ)

        p = [p_ODE; p_NN]

        # Define differential equations
        function f!(du, u, p, t)
                x, y = u
                G = 10.0f0
                L = p[1]
                p_NN = p[2:end]

                du[1] = y
                du[2] = -G/L*sin(x) + model_univ(u, p_NN)[1]
        end

        # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        # @info "Optimizing ODE Problem"
        # sys = modelingtoolkitize(_prob)
        # ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        # prob = ODEProblem(ODEFunc, u₀, tspan, p)
        prob = _prob

        solver = Tsit5()
        sensalg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))

        P = typeof(prob)
        S = typeof(solver)
        T = typeof(sensalg)
        new{P,S,T}(prob, solver, sensalg)
    end
    
end