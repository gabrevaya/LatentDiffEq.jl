

################################################################################
## Problem Definition -- Kuramoto Oscillators

struct Kuramoto_basic{T,P} <: AbstractSystem

    u₀::T
    p::T
    prob::P
    transform::F

    function Kuramoto_basic(N)
        # Default parameters and initial conditions
        Random.seed!(1)
        θ₀ = randn(Float32, N)
        ω = rand(Float32, N)
        K = rand(Float32)
        p = [ω; K]
        tspan = (0.f0, 1.f0)

        # Define differential equations
        function f!(dθ, θ, p, t)
            N = length(θ)
            ω = p[1:N]
            K = p[end]

            for i in 1:N
                dθ[i] = ω[i] + K/N * sum(j -> sin(θ[j] - θ[i]), 1:N)
            end
        end

        output_transform(θ) = sin.(θ)

        # Build ODE Problem
        _prob = ODEProblem(f!, θ₀, tspan, p)

        @info "Optimizing ODE Problem"
        # prob,_ = auto_optimize(_prob, verbose = false, static = false)
        sys = modelingtoolkitize(_prob)
        # prob = ODEProblem(sys,_prob.u0,_prob.tspan,_prob.p,
        #                        jac = true, tgrad = true, simplify = true,
        #                        sparse = false,
        #                        parallel = ModelingToolkit.SerialForm(),
        #                        eval_expression = false)
        prob = create_prob("Kuramoto", N, sys, u₀, tspan, p)

        T = typeof(θ₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(θ₀, p, prob, output_transform)
    end
end



struct Kuramoto{T,P} <: AbstractSystem

    u₀::T
    p::T
    prob::P
    transform::F

    function Kuramoto(N)
        # Default parameters and initial conditions
        Random.seed!(1)
        θ₀ = randn(Float32, N)
        ω = rand(Float32, N)
        C = rand(Float32)
        W = rand(Float32, N^2)
        p = [ω; C; W]

        # Define differential equations
        function f!(dθ, θ, p, t)
            N = length(θ)
            ω = p[1:N]
            C = p[N+1]
            W = reshape(p[N+2:end], (N,N))

            for i in 1:N
                dθ[i] = ω[i] + C * sum(j -> W[i,j]*sin(θ[j] - θ[i]), 1:N)
            end
        end

        output_transform(θ) = sin.(θ)

        # Build ODE Problem
        _prob = ODEProblem(f!, θ₀, (0.f0, 1.f0), p)

        @info "Optimizing ODE Problem"
        # prob,_ = auto_optimize(_prob, verbose = false, static = false)
        sys = modelingtoolkitize(_prob)
        prob = ODEProblem(sys,_prob.u0,_prob.tspan,_prob.p,
                               jac = true, tgrad = true, simplify = true,
                               sparse = false,
                               parallel = ModelingToolkit.SerialForm(),
                               eval_expression = false)

        T = typeof(θ₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(θ₀, p, prob, output_transform)
    end

end
