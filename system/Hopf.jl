

################################################################################
## Problem Definition -- Normal form of a supercritical Hopf bifurcation
# Deco et al. (2017), https://www.nature.com/articles/s41598-017-03073-5
# Ipiña et al. (2019), https://arxiv.org/abs/1907.04412 & https://www.sciencedirect.com/science/article/pii/S1053811920303207

struct Hopf{T, P} <: AbstractSystem
    u₀::T
    p::T
    prob::P

    function Hopf(k::Int64)
        # Default parameters and initial conditions
        Random.seed!(1)
        a = 0.2f0*randn(Float32,k)
        ω = fill(0.3f0, k) + 0.1f0*randn(Float32,k)
        G = 0.5f0
        C = 0.1rand(Float32,k^2)
        u₀ = rand(Float32,2*k)
        p = [a; ω; G; C]
        tspan = (0.f0, 1.f0)


        # Define differential equations
        function f!(du,u,p,t)
            k = length(u) ÷ 2
            a = p[1:k]
            ω = p[k+1:2*k]
            G = p[2*k+1]
            C_vec = p[2*k+2:end]
            C = reshape(C_vec, (k,k))

            x = u[1:k]
            y = u[k+1:end]

            for j ∈ 1:k
                coupling_x = 0.f0
                coupling_y = 0.f0
                for i ∈ 1:k
                    coupling_x += C[i,j]*(x[i] - x[j])
                    coupling_y += C[i,j]*(y[i] - y[j])
                end

                du[j] = (a[j] - x[j]^2 - y[j]^2) * x[j] - ω[j]*y[j] + G*coupling_x
                du[j+k] = (a[j] - x[j]^2 - y[j]^2) * y[j] - ω[j]*x[j] + G*coupling_y
            end

            # coupling_x = [sum(i -> C[i,j]*(x[i] - x[j]), 1:k) for j in 1:k]
            # coupling_y = [sum(i -> C[i,j]*(y[i] - y[j]), 1:k) for j in 1:k]
            #
            # @. du[1:k] = (a - x^2 - y^2) * x - ω*y + G*coupling_x
            # @. du[k+1:end] = (a - x^2 - y^2) * y - ω*x + G*coupling_y
        end


        # Build ODE Problem
       _prob = ODEProblem(f!, u₀, tspan, p)

       @info "Optimizing ODE Problem"
       # prob,_ = auto_optimize(_prob, verbose = false, static = false)
       sys = modelingtoolkitize(_prob)
       #prob = ODEProblem(sys,_prob.u0,_prob.tspan,_prob.p,
       #                       jac = true, tgrad = true, simplify = true,
       #                       sparse = true,
       #                       parallel = ModelingToolkit.SerialForm(),
       #                       eval_expression = false)
       prob = create_prob(sys, u₀, tspan, p)
       
       T = typeof(u₀)
       P = typeof(prob)
       new{T,P}(u₀, p, prob)
    end
end
