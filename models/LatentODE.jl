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
    rnn_μ
    rnn_logσ²

    device

    function LODE_encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)

        linear = Chain(Dense(input_dim, hidden_dim, relu),
                       Dense(hidden_dim, rnn_input_dim, relu)) |> device
        rnn = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                    RNN(rnn_output_dim, rnn_output_dim, relu)) |> device
        rnn_μ = Dense(rnn_output_dim, latent_dim) |> device
        rnn_logσ² = Dense(rnn_output_dim, latent_dim) |> device

        new(linear, rnn, rnn_μ, rnn_logσ², device)
    end
end

function (encoder::LODE_encoder)(x)

    h0 = encoder.linear.(x)
    # reverse time and pass to the rnn
    h0_rev = reverse(h0)
    h = encoder.rnn.(h0_rev)[end]
    reset!(encoder.rnn)
    encoder.rnn_μ(h), encoder.rnn_logσ²(h)
end

################################################################################
## LODE_decoder definition

struct LODE_decoder <: AbstractDecoder

    dudt
    linear

    device

    function LODE_decoder(input_dim, latent_dim, hidden_dim_node, hidden_dim_linear, device)

        dudt = Chain(Dense(latent_dim, hidden_dim_node, relu),
                        Dense(hidden_dim_node, hidden_dim_node, relu),
                        Dense(hidden_dim_node, latent_dim)) |> device

        linear = Chain(Dense(latent_dim, hidden_dim_linear, relu),
                       Dense(hidden_dim_linear, input_dim)) |> device

        new(dudt, linear, device)
        end
end

function (decoder::LODE_decoder)(x, t)

    nODE = NeuralODE(decoder.dudt, (t[1], t[end]), Tsit5(), saveat = t)
    ẑ = Array(nODE(x)) |> decoder.device
    x̂ = decoder.linear.(Flux.unstack(ẑ, 3))
    return x̂, ẑ
end

################################################################################
## Latent ODE model definition (Encoder/decoder container)

struct LatentODE <: AbstractModel

    encoder::LODE_encoder
    decoder::LODE_decoder

    device

    function LatentODE(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, hidden_dim_node, device)

        encoder = LODE_encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)
        decoder = LODE_decoder(input_dim, latent_dim, hidden_dim, hidden_dim_node, device)

        new(encoder, decoder, device)

    end

end

function (LatentODE::LatentODE)(x, t)

    μ, logσ² = latent_ODE.encoder(x)
    ẑ₀ = μ + LatentODE.device(randn(Float32, size(logσ²))) .* exp.(logσ²/2f0)
    x̂, ẑ = LatentODE.decoder(ẑ₀, t)
    return ((μ, logσ²),), x̂, (ẑ₀,), ẑ
end

Flux.@functor LODE_encoder
Flux.@functor LODE_decoder
Flux.@functor LatentODE
