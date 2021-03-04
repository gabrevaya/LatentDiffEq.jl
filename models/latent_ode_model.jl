# Latent ODE
#
# Based on
# https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl
# https://arxiv.org/abs/1806.07366
# https://arxiv.org/abs/2003.10775

################################################################################
## Encoder Definition

struct LODE_encoder <: AbstractEncoder

    linear
    rnn
    μ
    logσ²

    device

    function LODE_encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)

        linear = Chain(Dense(input_dim, hidden_dim, relu),
                       Dense(hidden_dim, rnn_input_dim, relu)) |> device
        rnn = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                    RNN(rnn_output_dim, rnn_output_dim, relu)) |> device
        μ = Dense(rnn_output_dim, latent_dim) |> device
        logσ² = Dense(rnn_output_dim, latent_dim) |> device

        new(linear, rnn, μ, logσ², device)
    end
end

Flux.@functor LODE_encoder

function (encoder::LODE_encoder)(x)

    h1 = encoder.linear.(x)
    # reverse time and pass to the rnn
    h = encoder.rnn.(h1[end:-1:1])[end]
    reset!(encoder.rnn)
    encoder.μ(h), encoder.logσ²(h)
    
end

################################################################################
## LODE_decoder definition

struct LODE_decoder <: AbstractDecoder

    dudt
    linear

    device

    function LODE_decoder(input_dim, latent_dim, hidden_dim, hidden_dim_node, time_size, t_max, device)

        dudt = Chain(Dense(latent_dim, hidden_dim_node, relu),
                        Dense(hidden_dim_node, hidden_dim_node, relu),
                        Dense(hidden_dim_node, latent_dim)) |> device
        # tspan = (zero(t_max), t_max)
        # t = range(tspan[1], tspan[2], length=time_size)
        #
        # node = NeuralODE(dudt, tspan, Tsit5(), saveat = t)
        linear = Chain(Dense(latent_dim, hidden_dim, relu),
                       Dense(hidden_dim, input_dim)) |> device

        new(dudt, linear, device)
        end
end

Flux.@functor LODE_decoder

function (decoder::LODE_decoder)(x, t)

    nODE = NeuralODE(decoder.dudt, (t[1], t[end]), Tsit5(), saveat = t)
    h = Array(nODE(x)) |> decoder.device
    h2 = Flux.unstack(h, 3)
    decoder.linear.(h2)
end

################################################################################
## Latent ODE model definition (Encoder/decoder container)

struct Latent_ODE <: AbstractModel

    encoder::LODE_encoder
    decoder::LODE_decoder

    device

    function Latent_ODE(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, hidden_dim_node, time_size, t_max, device)

        encoder = LODE_encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)
        decoder = LODE_decoder(input_dim, latent_dim, hidden_dim, hidden_dim_node, time_size, t_max, device)

        new(encoder, decoder, device)

    end

end

Flux.@functor Latent_ODE

function (latent_ODE::Latent_ODE)(x, t)

    μ, logσ² = latent_ODE.encoder(x)
    z = μ + latent_ODE.device(randn(Float32, size(logσ²))) .* exp.(logσ²/2f0)
    ((μ, logσ²),), latent_ODE.decoder(z, t), (z)

end
