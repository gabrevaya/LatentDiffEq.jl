

################################################################################
## Problem Definition -- Wilson-Cowan

struct WC_full{T, P, F} <: AbstractSystem

    u₀::T
    p::T
    prob::P
    transform::F

    function WC_full(k::Int64)
        # Default parameters and initial conditions
        α₁ = randn(Float32,k)
        α₂ = randn(Float32,k)
        α₃ = randn(Float32,k)
        α₄ = randn(Float32,k)
        W = rand(Float32, k^2)
        u₀ = rand(Float32,2*k)
        p = [α₁; α₂; α₃; α₄; W]
        tspan = (0.f0, 1.f0)

        # Define differential equations
        function f!(dx,x,p,t)
            k = length(x) ÷ 2
            α₁ = p[1:k]
            α₂ = p[k+1:2*k]
            α₃ = p[2*k+1:3*k]
            α₄ = p[3*k+1:4*k]
            W_vec = p[4*k+1:end]
            W = reshape(W_vec, (k,k))

            x₁ = x[1:k]
            x₂ = x[k+1:end]

            Wσx = W*σ.(x₁)
            @. dx[1:k] = -x₁ + σ(α₁*x₁ - x₂ + α₂) + α₃*Wσx
            @. dx[k+1:end] = -x₂ + σ(α₄*x₁)
        end

        output_transform(x) = x

            # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        prob = create_prob("Wilson-Cowan_full_$k", k, sys, u₀, tspan, p)

        T = typeof(u₀)
        P = typeof(prob)
        F = typeof(output_transform)

        new{T,P,F}(u₀, p, prob, output_transform)
    end
end



struct WC{T, P, F} <: AbstractSystem

    u₀::T
    p::T
    prob::P
    transform::F

    function WC(k::Int64)
        # Default parameters and initial conditions
        α₁ = randn(Float32,k)
        α₂ = randn(Float32,k)
        α₃ = randn(Float32,k)
        W = rand(Float32, k^2)
        u₀ = rand(Float32,2*k)
        p = [α₁; α₂; α₃; W]


        # Define differential equations
        function f!(dx,x,p,t)
            k = length(x) ÷ 2
            α₁ = p[1:k]
            α₂ = p[k+1:2*k]
            α₃ = p[2*k+1:3*k]
            W_vec = p[4*k+1:end]
            W = reshape(W_vec, (k,k))

            x₁ = x[1:k]
            x₂ = x[k+1:end]

            Wσx = W*σ.(x₁)
            @. dx[1:k] = -x₁ + σ(α₁*x₁ - x₂ + α₂) + Wσx
            @. dx[k+1:end] = -x₂ + σ(α₃*x₁)
        end

        output_transform(x) = x

        # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        prob = create_prob("Wilson-Cowan_$k", k, sys, u₀, tspan, p)
 

        T = typeof(u₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P,F}(u₀, p, prob, output_transform)
    end
end



struct WC_identical_local{T, P,F} <: AbstractSystem

    u₀::T
    p::T
    prob::P
    transform::F

    function WC_identical_local(k::Int64)
        # Default parameters and initial conditions
        α₁ = randn(Float32)
        α₂ = randn(Float32)
        α₃ = randn(Float32)
        W = rand(Float32, k^2)
        u₀ = rand(Float32,2*k)
        p = [α₁; α₂; α₃; W]


        # Define differential equations
        function f!(dx,x,p,t)
            k = length(x) ÷ 2
            α₁ = p[1]
            α₂ = p[2]
            α₃ = p[3]
            W_vec = p[4:end]
            W = reshape(W_vec, (k,k))

            x₁ = x[1:k]
            x₂ = x[k+1:end]

            Wσx = W*σ.(x₁)
            @. dx[1:k] = -x₁ + σ(α₁*x₁ - x₂ + α₂) + Wσx
            @. dx[k+1:end] = -x₂ + σ(α₃*x₁)
        end

        output_transform(x) = x

        # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        prob = create_prob("Wilson-Cowan_identical_local_$k", k, sys, u₀, tspan, p)

        T = typeof(u₀)
        P = typeof(prob)
        F = typeof(output_transform)
        new{T,P, F}(u₀, p, prob, output_transform)
    end
end



import NNlib:σ

# σ(x::Num) = ModelingToolkit.Constant(1) / ( ModelingToolkit.Constant(1) + exp(-x))
σ(x::Num) = Num(1) / ( Num(1) + exp(-x))
