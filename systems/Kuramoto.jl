

################################################################################
## Problem Definition -- Kuramoto Oscillators
abstract type Kuramoto end


struct Kuramoto_basic{P,S,T} <: Kuramoto

    prob::P
    solver::S
    sensealg::T

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

        # Build ODE Problem
        _prob = ODEProblem(f!, θ₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob = ODEProblem(ODEFunc, θ₀, tspan, p)

        solver = Tsit5()
        sensalg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))

        P = typeof(prob)
        S = typeof(solver)
        T = typeof(sensalg)
        new{P,S,T}(prob, solver, sensalg)
    end
end



struct Kuramoto_full{P,S,T} <: Kuramoto

    prob::P
    solver::S
    sensealg::T

    function Kuramoto_full(N)
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

        # Build ODE Problem
        _prob = ODEProblem(f!, θ₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        ODEFunc = ODEFunction(sys, tgrad=true, jac = true, sparse = false, simplify = false)
        prob = ODEProblem(ODEFunc, θ₀, tspan, p)

        solver = Tsit5()
        sensalg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))

        P = typeof(prob)
        S = typeof(solver)
        T = typeof(sensalg)
        new{P,S,T}(prob, solver, sensalg)
    end

end

function transform_after_diffeq!(θ, diffeq::Kuramoto)
    θ = sin.(θ)
    return nothing
end