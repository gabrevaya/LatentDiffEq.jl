

################################################################################
## Problem Definition -- neural ODE

struct NODE{D,S,N,L}

    dudt::D
    solver::S
    neural_model::N
    latent_dim::L

    function NODE(latent_dim; hidden_dim=200, device=cpu)
        dudt = Chain(Dense(latent_dim, hidden_dim, relu),
                        Dense(hidden_dim, hidden_dim, relu),
                        Dense(hidden_dim, latent_dim)) |> device
        solver = Tsit5()
        neural_model = NeuralODE
        # sensalg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))

        D = typeof(dudt)
        S = typeof(solver)
        N = typeof(neural_model)
        L = typeof(latent_dim)
        new{D,S,N,L}(dudt, solver, neural_model,latent_dim)
    end
    
end