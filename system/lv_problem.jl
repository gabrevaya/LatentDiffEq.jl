

################################################################################
## Problem Definition -- Lotka-Volterra

struct LV{T<: AbstractArray} <: AbstractSystem
    u₀::T
    p::T

    function LV()
        # Default parameters and initial conditions
        u₀ = Float32[1.0, 1.0]
        p = Float32[1.25, 1.5, 1.75, 2]
        T = typeof(u₀)
        new{T}(u₀, p)
    end
end


function generate_func(system::LV)

   # ODE function for solving in the GOKU-net architecture (decoder)
   function f!(du, u, p, t)
       # @inbounds begin
       #     x = u[1]
       #     y = u[2]
       #     α = p[1]
       #     β = p[2]
       #     δ = p[3]
       #     γ = p[4]
           x, y = u
           α, β, δ, γ = p
           du[1] = α*x - β*x*y
           du[2] = -δ*y + γ*x*y
       # end
       # nothing
   end

   func = ODEFunction(f!)
end
