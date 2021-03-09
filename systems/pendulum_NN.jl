

################################################################################
## Problem Definition -- "pendulum" with NN inside
using DiffEqFlux

struct pendulum_NN{T,P,F} <: AbstractSystem
    u₀::T
    p::T
    prob::P
    transform::F

    function pendulum_NN()
        # Default parameters and initial conditions
        u₀ = Float32[1.0, 1.0]
        tspan = (0.f0, 1.f0)

        model_univ = FastChain(FastDense(2, 16, tanh),
                       FastDense(16, 16, tanh),
                       FastDense(16, 1))

        # The model weights are destructured into a vector of parameters
        p = initial_params(model_univ)

        # Define differential equations
        function f!(du, u, p, t)
                x, y = u

                du[1] = y
                du[2] = model_univ(u, p)[1]
        end

        output_transform(u) = u

        # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        # @info "Optimizing ODE Problem"
        # sys = modelingtoolkitize(_prob)
        # ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        # prob = ODEProblem(ODEFunc, u₀, tspan, p)
        prob = _prob

        T = typeof(u₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(u₀, p, prob, output_transform)
    end
    
end