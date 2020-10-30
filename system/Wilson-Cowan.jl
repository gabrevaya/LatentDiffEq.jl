

################################################################################
## Problem Definition -- Wilson-Cowan

struct WC_full{T, P} <: AbstractSystem
    u₀::T
    p::T
    prob::P

    function WC_full(k::Int64)
        # Default parameters and initial conditions
        Random.seed!(1)
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


        # Build ODE Problem
       _prob = ODEProblem(f!, u₀, tspan, p)

       @info "Optimizing ODE Problem"
       # prob,_ = auto_optimize(_prob, verbose = false, static = false)
       sys = modelingtoolkitize(_prob)
       # prob = ODEProblem(sys,_prob.u0,_prob.tspan,_prob.p,
       #                        jac = true, tgrad = true, simplify = true,
       #                        sparse = false,
       #                        parallel = ModelingToolkit.SerialForm(),
       #                        eval_expression = false)
       prob = create_prob("Wilson-Cowan", k, sys, u₀, tspan, p)

        T = typeof(u₀)
        P = typeof(prob)
        new{T,P}(u₀, p, prob)
    end
end



struct WC{T, P} <: AbstractSystem
    u₀::T
    p::T
    prob::P

    function WC(k::Int64)
        # Default parameters and initial conditions
        Random.seed!(1)
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


        # Build ODE Problem
       _prob = ODEProblem(f!, u₀, (0.f0, 1.f0), p)

       @info "Optimizing ODE Problem"
       # prob,_ = auto_optimize(_prob, verbose = false, static = false)
       sys = modelingtoolkitize(_prob)
       prob = ODEProblem(sys,_prob.u0,_prob.tspan,_prob.p,
                              jac = true, tgrad = true, simplify = true,
                              sparse = false,
                              parallel = ModelingToolkit.SerialForm(),
                              eval_expression = false)

        T = typeof(u₀)
        P = typeof(prob)
        new{T,P}(u₀, p, prob)
    end
end



struct WC_identical_local{T, P} <: AbstractSystem
    u₀::T
    p::T
    prob::P

    function WC_identical_local(k::Int64)
        # Default parameters and initial conditions
        Random.seed!(1)
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


        # Build ODE Problem
       _prob = ODEProblem(f!, u₀, (0.f0, 1.f0), p)

       @info "Optimizing ODE Problem"
       # prob,_ = auto_optimize(_prob, verbose = false, static = false)
       sys = modelingtoolkitize(_prob)
       prob = ODEProblem(sys,_prob.u0,_prob.tspan,_prob.p,
                              jac = true, tgrad = true, simplify = true,
                              sparse = false,
                              parallel = ModelingToolkit.SerialForm(),
                              eval_expression = false)

        T = typeof(u₀)
        P = typeof(prob)
        new{T,P}(u₀, p, prob)
    end
end



import NNlib:σ
σ(x::Operation) = ModelingToolkit.Constant(1) / ( ModelingToolkit.Constant(1) + exp(-x))
