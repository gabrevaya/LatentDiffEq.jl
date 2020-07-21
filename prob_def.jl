

################################################################################
## Problem Definition -- Lotka-Volterra

# ODE function for solving in the GOKU-net architecture (decoder)
function lv_func(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    @inbounds begin
        du[1] = α*x - β*x*y
        du[2] = -δ*y + γ*x*y
    end
    nothing
end

# Deterministic neural-net used to get from state to imput sample for the GOKU architecture (input_dim =/= ode_dim)
struct lv_gen

      linear

      function lv_gen(ode_dim, hidden_dim_gen, input_dim)
            linear = Chain(Dense(ode_dim, hidden_dim_gen, relu),
                           Dense(hidden_dim_gen, input_dim, relu))
            new(linear)
      end
end

function (gen::lv_gen)(z)

      return gen.linear.(z)
end
