

################################################################################
## Problem Definition -- Kuramoto Oscillators

struct Kuramoto_basic{T,P,F} <: AbstractSystem

    u₀::T
    p::T
    prob::P
    transform::F

    function Kuramoto_basic(N)
        # Default parameters and initial conditions
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
        sys = modelingtoolkitize(_prob)
        ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob = ODEProblem(ODEFunc, θ₀, tspan, p)

        T = typeof(θ₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(θ₀, p, prob, output_transform)
    end
end



struct Kuramoto{T,P,F} <: AbstractSystem

    u₀::T
    p::T
    prob::P
    transform::F

    function Kuramoto(N)
        # Default parameters and initial conditions
        θ₀ = randn(Float32, N)
        ω = rand(Float32, N)
        C = rand(Float32)
        W = rand(Float32, N^2)
        p = [ω; C; W]
        tspan = (0.f0, 1.f0)

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
        _prob = ODEProblem(f!, θ₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob = ODEProblem(ODEFunc, θ₀, tspan, p)

        T = typeof(θ₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(θ₀, p, prob, output_transform)
    end

end
