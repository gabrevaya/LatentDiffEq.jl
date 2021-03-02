# GOKU-NET MODEL
#
# Based on
# https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl
# https://arxiv.org/abs/1806.07366
# https://arxiv.org/abs/2003.10775

################################################################################
## Encoder definition

struct GOKU_encoder <: AbstractEncoder

    linear
    rnn
    rnn_μ
    rnn_logσ²
    lstm_forward        # TODO: Implement bidirectional LSTM : https://github.com/maetshju/flux-blstm-implementation/blob/master/01-blstm.jl
    lstm_backward
    lstm_μ                                           # https://github.com/AzamatB/Tacotron2.jl
    lstm_logσ²

    device

    function GOKU_encoder(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, rnn_input_dim, rnn_output_dim, latent_dim, device)

        rnn = Chain(RNN(input_dim, rnn_output_dim, relu),
                    RNN(rnn_output_dim, rnn_output_dim, relu)) |> device

        rnn_μ = Dense(rnn_output_dim, latent_dim) |> device
        rnn_logσ² = Dense(rnn_output_dim, latent_dim) |> device

        lstm_forward = Chain(LSTM(rnn_input_dim, rnn_output_dim),
                                LSTM(rnn_output_dim, rnn_output_dim)) |> device

        lstm_backward = Chain(LSTM(rnn_input_dim, rnn_output_dim),
                                LSTM(rnn_output_dim, rnn_output_dim)) |> device


        lstm_μ = Dense(rnn_output_dim*2, latent_dim) |> device
        lstm_logσ² = Dense(rnn_output_dim*2, latent_dim) |> device

        new(linear, rnn, rnn_μ, rnn_logσ², lstm_forward, lstm_backward, lstm_μ, lstm_logσ², device)
    end
end


function (encoder::GOKU_encoder)(x)

    # Pass all states in the time series in dense layer
    h = encoder.linear.(x)
    h_rev = reverse(h)

    # Pass an RNN and an LSTM through latent states
    rnn_out = encoder.rnn.(h_rev)[end]
    lstm_out_f = encoder.lstm_forward.(h)[end]
    lstm_out_b = encoder.lstm_backward.(h_rev)[end]
    bi_lstm_out = vcat(lstm_out_f, lstm_out_b)

    reset!(encoder.rnn)
    reset!(encoder.lstm_forward)
    reset!(encoder.lstm_backward)

    # Return RNN/BiLSTM ouput passed trough dense layers
    z₀_μ = encoder.rnn_μ(rnn_out)
    z₀_logσ² = encoder.rnn_logσ²(rnn_out)
    θ_μ = encoder.lstm_μ(bi_lstm_out)
    θ_logσ² = encoder.lstm_logσ²(bi_lstm_out)

    z₀_μ, z₀_logσ², θ_μ, θ_logσ²
end

################################################################################
## Decoder definition

struct GOKU_decoder <: AbstractDecoder

    solver
    ode_prob
    transform

    z₀_linear
    θ_linear
    gen_linear

    device

    function GOKU_decoder(input_dim, hidden_dim1, hidden_dim2, hidden_dim3,
                            latent_dim, hidden_dim_latent_ode, ode_dim, θ_dim,
                            ode_prob, transform, solver, device)

        z₀_linear = Chain(Dense(latent_dim, hidden_dim_latent_ode, relu),
                          Dense(hidden_dim_latent_ode, ode_dim)) |> device
        θ_linear = Chain(Dense(latent_dim, hidden_dim_latent_ode, relu),
                         Dense(hidden_dim_latent_ode, θ_dim, softplus)) |> device


        l1 = Dense(ode_dim, hidden_dim1, relu)
        l2 = Dense(hidden_dim1, hidden_dim1, relu)
        l3 = Dense(hidden_dim1, hidden_dim1, relu)
        l4 = Dense(hidden_dim1, input_dim, σ)

        gen_linear = Chain(l1,
                        SkipConnection(l2, +),
                        SkipConnection(l3, +),
                        l4)  |> device

        new(solver, ode_prob, transform, z₀_linear, θ_linear, gen_linear, device)
    end
end


function (decoder::GOKU_decoder)(ẑ₀, θ̂, t)

    ## Pass sampled latent states in dense layers
    ẑ₀ = decoder.z₀_linear(ẑ₀)
    θ̂ = decoder.θ_linear(θ̂)

    #####
    # Function definition for ensemble problem
    prob_func = (prob,i,repeat) -> remake(prob, u0=ẑ₀[:,i], p = θ̂[:,i]) # TODO: try using views and switching indexes to see if the performance improves
    function output_func(sol, i)
        # Check if solve was successful, if not fill z_pred with zeros to avoid problems with dimensions matches
        if sol.retcode != :Success
            return (zeros(Float32, size(ẑ₀, 1), size(t,1)), false)
            # return (1000*ones(Float32, size(ẑ₀, 1), size(t,1)), false)
        else
            return (Array(sol), false)
        end
    end
    #####

    ## Adapt problem to given time span and create ensemble problem definition
    prob = remake(decoder.ode_prob; tspan = (t[1],t[end]))
    ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)

    ## Solve
    ẑ = solve(ens_prob, decoder.solver, EnsembleSerial(), sensealg=BacksolveAdjoint(autojacvec=ReverseDiffVJP(true), checkpointing=true), trajectories=size(θ̂, 2), saveat = t) |> decoder.device
    
    # Transform the resulting output (Mainly used for Kuramoto system to pass from phase -> time space)
    ẑ = decoder.transform(ẑ)

    ## Create output data shape
    recon_batch = Flux.unstack(ẑ, 2)

    return recon_batch, ẑ, ẑ₀, θ̂
end

################################################################################
## Goku definition (Encoder/decoder container)

struct Goku <: AbstractModel

    encoder::GOKU_encoder
    decoder::GOKU_decoder

    variational::Bool

    device

    function Goku(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, rnn_input_dim, rnn_output_dim, latent_dim, hidden_dim_latent_ode, ode_dim, θ_dim, ode_prob, transform, solver, variational, device)

        encoder = GOKU_encoder(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, rnn_input_dim, rnn_output_dim, latent_dim, device)
        decoder = GOKU_decoder(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, latent_dim, hidden_dim_latent_ode, ode_dim, θ_dim, ode_prob, transform, solver, device)

        new(encoder, decoder, variational, device)
    end
end


function (goku::Goku)(x, t)
    ## Get encoded latent initial states and parameters
    z₀_μ, z₀_logσ²,  θ_μ,  θ_logσ² = goku.encoder(x)
    
    ## Sample from the distributions
    if goku.variational
        ẑ₀ = z₀_μ + goku.device(randn(Float32, size(z₀_logσ²))) .* exp.(z₀_logσ²/2f0)
        θ̂ =  θ_μ + goku.device(randn(Float32, size( θ_logσ²))) .* exp.( θ_logσ²/2f0)
    else
        ẑ₀ = z₀_μ
        θ̂ =  θ_μ
    end
    ## Get predicted output
    x̂, ẑ, ẑ₀, θ̂ = goku.decoder(ẑ₀, θ̂, t)

    return ((z₀_μ, z₀_logσ²), ( θ_μ,  θ_logσ²)), x̂, (ẑ₀, θ̂), ẑ
end


Flux.@functor GOKU_encoder
Flux.@functor GOKU_decoder
Flux.@functor Goku
