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
    lstm        # TODO: Implement bidirectional LSTM : https://github.com/maetshju/flux-blstm-implementation/blob/master/01-blstm.jl
    lstm_μ                                           # https://github.com/AzamatB/Tacotron2.jl
    lstm_logσ²

    device

    function GOKU_encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)

        linear = Chain(Dense(input_dim, hidden_dim, relu),
                       Dense(hidden_dim, rnn_input_dim, relu)) |> device
        rnn = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                    RNN(rnn_output_dim, rnn_output_dim, relu)) |> device

        rnn_μ = Dense(rnn_output_dim, latent_dim) |> device
        rnn_logσ² = Dense(rnn_output_dim, latent_dim) |> device

        lstm = Chain(LSTM(rnn_input_dim, rnn_output_dim),
                     LSTM(rnn_output_dim, rnn_output_dim)) |> device

        lstm_μ = Dense(rnn_output_dim, latent_dim) |> device
        lstm_logσ² = Dense(rnn_output_dim, latent_dim) |> device

        new(linear, rnn, rnn_μ, rnn_logσ², lstm, lstm_μ, lstm_logσ², device)
    end
end

function (encoder::GOKU_encoder)(x)
    h = encoder.linear.(x)
    h_rev = reverse(h)
    rnn_out = encoder.rnn.(h_rev)[end]
    lstm_out = encoder.lstm.(h)[end]
    reset!(encoder.rnn)
    reset!(encoder.lstm)
    encoder.rnn_μ(rnn_out), encoder.rnn_logσ²(rnn_out), encoder.lstm_μ(lstm_out), encoder.lstm_logσ²(lstm_out)
end

################################################################################
## Decoder definition

struct GOKU_decoder <: AbstractDecoder

    solver
    ode_func
    ode_prob

    z₀_linear
    p_linear
    gen_linear

    device

    function GOKU_decoder(input_dim, latent_dim, hidden_dim, ode_dim, p_dim, ode_func, solver, device)

        z₀_linear = Chain(Dense(latent_dim, hidden_dim, relu),
                          Dense(hidden_dim, ode_dim, softplus)) |> device
        p_linear = Chain(Dense(latent_dim, hidden_dim, relu),
                         Dense(hidden_dim, p_dim, softplus)) |> device
        gen_linear = Chain(Dense(ode_dim, hidden_dim, relu),
                           Dense(hidden_dim, input_dim)) |> device

        ode_prob = ODEProblem(ode_func, [0., 0.], (0., 1.), [0., 0., 0., 0.])

        new(solver, ode_func, ode_prob, z₀_linear, p_linear, gen_linear, device)

    end

end

function (decoder::GOKU_decoder)(latent_z₀, latent_p, t)

    z₀ = decoder.z₀_linear(latent_z₀)
    p = decoder.p_linear(latent_p)

    function output_func(sol, i)
        # Check if solve was successful, if not fill z_pred with zeros to avoid problems with dimensions matches
        if sol.retcode != :Success
            return (zeros(Float32, size(z₀, 1), size(t)), false)
        else
            return (Array(sol), false)
        end
    end
    prob_func = (prob,i,repeat) -> remake(prob, u0=z₀[:,i], p = p[:,i])

    prob = remake(decoder.ode_prob; tspan = (t[1],t[end]))

    ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)#, reduction = reduction)
    pred_z = solve(ens_prob, decoder.solver,  EnsembleThreads(), trajectories=size(p, 2), saveat = t) |> decoder.device

    # pred_x = decoder.gen_linear. (pred_z) # TODO : create new dataset from a trained generation function

    return Flux.unstack(pred_z, 2), z₀, p

end

################################################################################
## Goku definition (Encoder/decoder container)

struct Goku <: AbstractModel

    encoder::GOKU_encoder
    decoder::GOKU_decoder

    device

    function Goku(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, hidden_dim, ode_dim, p_dim, ode_sys, solver, device)

        encoder = GOKU_encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)
        decoder = GOKU_decoder(input_dim, latent_dim, hidden_dim, ode_dim, p_dim, ode_sys, solver, device)

        new(encoder, decoder, device)

    end

end

function (goku::Goku)(x, t)

    latent_z₀_μ, latent_z₀_logσ², latent_p_μ, latent_p_logσ² = goku.encoder(x)

    latent_z₀ = latent_z₀_μ + goku.device(randn(Float32, size(latent_z₀_logσ²))) .* exp.(latent_z₀_logσ²/2f0)
    latent_p = latent_p_μ + goku.device(randn(Float32, size(latent_p_logσ²))) .* exp.(latent_p_logσ²/2f0)

    pred_x, pred_z₀, pred_p = goku.decoder(latent_z₀, latent_p, t)

    return ((latent_z₀_μ, latent_z₀_logσ²), (latent_p_μ, latent_p_logσ²)), pred_x, (pred_z₀, pred_p)

end
