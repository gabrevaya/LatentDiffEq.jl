################################################################################
## Problem Definition -- neural ODE
struct NODE{D,S,N,L,A,K}

    dudt::D
    solver::S
    neural_model::N
    latent_dim_in::L
    latent_dim_out::L
    augment_dim::A
    kwargs::K

    function NODE(latent_dim_in; hidden_dim=200, augment_dim=0, device=cpu, kwargs...)
        dudt = Chain(Dense(latent_dim_in+augment_dim, hidden_dim, relu),
                        Dense(hidden_dim, hidden_dim, relu),
                        Dense(hidden_dim, latent_dim_in+augment_dim)) |> device
        solver = Tsit5()
        neural_model = NeuralODE
        # sensalg = BacksolveAdjoint(autojacvec=ReverseDiffVJP(true))

        latent_dim_out = latent_dim_in + augment_dim

        D = typeof(dudt)
        S = typeof(solver)
        N = typeof(neural_model)
        L = typeof(latent_dim_in)
        A = typeof(augment_dim)
        K = typeof(kwargs)
        new{D,S,N,L,A,K}(dudt, solver, neural_model,latent_dim_in,
                            latent_dim_out, augment_dim, kwargs)
    end
    
end