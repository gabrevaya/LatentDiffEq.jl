# Latent ODE
#
# Based on
# https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl
# https://arxiv.org/abs/1806.07366
# https://arxiv.org/abs/2003.10775

module Latent_ODE_model

export Latent_ODE

using OrdinaryDiffEq
using Flux
using Flux: reset!
using DiffEqFlux
using Statistics

include("../utils/utils.jl")

################################################################################
## Model Definition
struct Encoder
    linear
    rnn
    μ
    logσ²

    function Encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)

        linear = Chain(Dense(input_dim, hidden_dim, relu),
                       Dense(hidden_dim, rnn_input_dim, relu)) |> device
        rnn = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                    RNN(rnn_output_dim, rnn_output_dim, relu)) |> device
        μ = Dense(rnn_output_dim, latent_dim) |> device
        logσ² = Dense(rnn_output_dim, latent_dim) |> device

        new(linear, rnn, μ, logσ²)
    end
end

function (encoder::Encoder)(x)
    h1 = encoder.linear.(x)
    # reverse time and pass to the rnn
    h = encoder.rnn.(h1[end:-1:1])[end]
    reset!(encoder.rnn)
    encoder.μ(h), encoder.logσ²(h)
end

struct Decoder
    neuralODE
    linear

    function Decoder(input_dim, latent_dim, hidden_dim, hidden_dim_node, time_size, t_max, device)

        dudt2 = Chain(Dense(latent_dim, hidden_dim_node, relu),
                        Dense(hidden_dim_node, hidden_dim_node, relu),
                        Dense(hidden_dim_node, latent_dim)) |> device
        tspan = (zero(t_max), t_max)
        t = range(tspan[1], tspan[2], length=time_size)

        node = NeuralODE(dudt2, tspan, Tsit5(), saveat = t)
        linear = Chain(Dense(latent_dim, hidden_dim, relu),
                       Dense(hidden_dim, input_dim)) |> device

        new(node, linear)
        end
end

function (decoder::Decoder)(x, device)
    h = Array(decoder.neuralODE(x)) |> device
    h2 = Flux.unstack(h, 3)
    out = decoder.linear.(h2)
end

struct Latent_ODE

    encoder::Encoder
    decoder::Decoder

    loss_batch::Function

    device

    function Latent_ODE(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, hidden_dim_node, time_size, t_max, device)

        encoder = Encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)
        decoder = Decoder(input_dim, latent_dim, hidden_dim, hidden_dim_node, time_size, t_max, device)

        loss_batch = function loss_batch(latent_ODE::Latent_ODE, λ, x, t, af)
            μ, logσ², pred_x = latent_ODE(x)
            reconstruction_loss = rec_loss(x, pred_x)
            kl_loss = mean(sum(KL.(μ, logσ²), dims = 1))
            return reconstruction_loss + af * kl_loss
        end

        new(encoder, decoder, loss_batch, device)

    end

end

function (latent_ODE::Latent_ODE)(x)
    μ, logσ² = latent_ODE.encoder(x)
    z = μ + latent_ODE.device(randn(Float32, size(logσ²))) .* exp.(logσ²/2f0)
    μ, logσ², latent_ODE.decoder(z, latent_ODE.device)
end

end # Module Latent_ODE_model
