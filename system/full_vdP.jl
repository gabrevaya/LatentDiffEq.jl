

################################################################################
## Problem Definition -- Lotka-Volterra

struct vdP{T<: AbstractArray} <: AbstractSystem
    f!
    jac!
    tgrad
    u₀::T
    p::T

    function vdP(k::Int64)

        # ODE function for solving in the GOKU-net architecture (decoder)

        function f!(dx,x,p,t)
        	k = length(x) ÷ 2
        	α₁ = @view p[1:k]
        	α₂ = @view p[k+1:2*k]
        	W_vec = @view p[2*k+1:end]
        	W = reshape( W_vec, (k,k))

        	x₁ = @view x[1:k]
        	x₂ = @view x[k+1:end]
            Wx = similar(x₁)


        	mul!(Wx, W, x₁)
        	@. dx[1:k] = α₁ * x₁ * (1 - x₁^2) + x₂ * α₂ + Wx
        	@. dx[k+1:end] = -x₁
        end

        Random.seed!(1)
        α₁ = fill(0.6f0, k) + 0.1f0*randn(Float32,k)
        α₂ = fill(10.f0, k) + 0.1f0*randn(Float32,k)
        W = rand(Float32, k^2)
        u₀ = rand(Float32,2*k)
        p = [α₁; α₂; W]

        T = typeof(u₀)
        new{T}(f!, nothing, nothing, u₀, p)
    end
end


# Deterministic neural-net used to get from state to imput sample for the GOKU architecture (input_dim ≂̸ ode_dim)
gen(system::vdP, hidden_dim_gen, input_dim) = Chain(Dense(length(system.u₀), hidden_dim_gen, relu),
                                                   Dense(hidden_dim_gen, input_dim, relu))
