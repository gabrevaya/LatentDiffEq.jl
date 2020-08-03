

################################################################################
## Problem Definition -- Lotka-Volterra

# ODE function for solving in the GOKU-net architecture (decoder)
function lv_func(du, u, p, t)
    @inbounds begin
        x = u[1]
        y = u[2]
        α = p[1]
        β = p[2]
        δ = p[3]
        γ = p[4]
        # x, y = u
        # α, β, δ, γ = p
        du[1] = α*x - β*x*y
        du[2] = -δ*y + γ*x*y
    end
    nothing
end

function lv_jac(J, u, p, t)
    @inbounds begin
        x = u[1]
        y = u[2]
        α = p[1]
        β = p[2]
        δ = p[3]
        γ = p[4]
        # x, y = u
        # α, β, δ, γ = p
        J[1,1] = α - β*y
        J[1,2] = -β*x
        J[2,1] = γ*y
        J[2,2] = -δ + γ*x
    end
    nothing
end

function lv_tgrad(J, u, p, t)
    nothing
end

# Deterministic neural-net used to get from state to imput sample for the GOKU architecture (input_dim ≂̸ ode_dim)
lv_gen(ode_dim, hidden_dim_gen, input_dim) = Chain(Dense(ode_dim, hidden_dim_gen, relu),
                                                   Dense(hidden_dim_gen, input_dim, relu))
