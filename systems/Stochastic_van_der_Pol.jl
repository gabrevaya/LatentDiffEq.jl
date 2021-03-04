

################################################################################
## Problem Definition -- Stochastic van der Pol
using StochasticDiffEq, DiffEqSensitivity

struct SvdP_full{T, P, F} <: AbstractSystem
    u₀::T
    p::T
    prob::P
    transform::F

    function SvdP_full(k::Int64)
        # Default parameters and initial conditions
        α₁ = fill(0.6f0, k) + 0.2f0*randn(Float32,k)
        α₂ = fill(10.f0, k) + 0.2f0*randn(Float32,k)
        W = rand(Float32, k^2)
        u₀ = rand(Float32,2*k)
        p = [α₁; α₂; W]
        tspan = (0.f0, 1.f0)

        # Define differential equations
        function f!(dx,x,p,t)
            k = length(x) ÷ 2
            α₁ = p[1:k]
            α₂ = p[k+1:2*k]
            W_vec = p[2*k+1:end]
            W = reshape(W_vec, (k,k))

            x₁ = x[1:k]
            x₂ = x[k+1:end]

            Wx = W*x₁
            @. dx[1:k] = α₁ * x₁ * (1 - x₁^2) + x₂ * α₂ + Wx
            @. dx[k+1:end] = -x₁
        end


        # Build ODE Problem
        _prob = ODEProblem(f!, u₀, tspan, p)

        @info "Optimizing ODE Problem"
        sys = modelingtoolkitize(_prob)
        prob = create_prob("Stochastic_van_der_Pol_$k", k, sys, u₀, tspan, p)

        function σ(du,u,p,t)
            du .= 0.2f0*u
        end

        prob_sde = SDEProblem(prob.f.f, σ, prob.u0, prob.tspan, prob.p, jac = prob.f.jac, tgrad = prob.f.tgrad)

        output_transform(x) = x
        
        T = typeof(u₀)
        P = typeof(prob_sde)
        F = typeof(output_transform)
        new{T,P,F}(u₀, p, prob_sde, output_transform)
    end
end