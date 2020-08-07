

################################################################################
## Problem Definition -- van der Pol unrestricted

struct vdP{T, P} <: AbstractSystem
    u₀::T
    p::T
    prob::P

    function vdP(k::Int64)
        # Default parameters and initial conditions
        Random.seed!(1)
        α₁ = fill(0.6f0, k) + 0.1f0*randn(Float32,k)
        α₂ = fill(10.f0, k) + 0.1f0*randn(Float32,k)
        W = rand(Float32, k^2)
        u₀ = rand(Float32,2*k)
        p = [α₁; α₂; W]


        # Define differential equations
        function f!(dx,x,p,t)
            k = length(x) ÷ 2
            α₁ = p[1:k]
            α₂ = p[k+1:2*k]
            W_vec = p[2*k+1:end]
            W = reshape( W_vec, (k,k))

            x₁ = x[1:k]
            x₂ = x[k+1:end]

            Wx = W*x₁
            @. dx[1:k] = α₁ * x₁ * (1 - x₁^2) + x₂ * α₂ + Wx
            @. dx[k+1:end] = -x₁
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
