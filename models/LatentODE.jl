# Latent ODE
#
# Based on
# https://arxiv.org/pdf/1806.07366.pdf
# https://arxiv.org/abs/1806.07366
# https://arxiv.org/abs/2003.10775

struct LatentODE <: LatentDiffEq end

struct LatentODE_encoder{L1,L2,L3,L4} <: AbstractEncoder

    layer1::L1
    layer2_z₀::L2
    layer3_μ_z₀::L3
    layer3_logσ²_z₀::L4

    function LatentODE_encoder(encoder_layers)
        L1,L2,L3,L4 = typeof.(encoder_layers)
        new{L1,L2,L3,L4}(encoder_layers...)
    end
end

function (encoder::LatentODE_encoder)(x)

    # Pass all states in the time series in dense layer
    l1_out = encoder.layer1.(x)

    # Pass an RNN and an BiLSTM through latent states
    l2_z₀_out = apply_layers2(encoder, l1_out)

    # Return RNN/BiLSTM ouput passed trough dense layers
    z₀_μ = encoder.layer3_μ_z₀(l2_z₀_out)
    z₀_logσ² = encoder.layer3_logσ²_z₀(l2_z₀_out)

    (z₀_μ,), (z₀_logσ²,)
end

function apply_layers2(encoder::LatentODE_encoder, l1_out)
    # reverse sequence
    l1_out_rev = reverse(l1_out)

    # pass it through the recurrent layer
    l2_z₀_out = encoder.layer2_z₀.(l1_out_rev)[end]

    # reset hidden states
    reset!(encoder.layer2_z₀)

    return l2_z₀_out
end

Flux.@functor LatentODE_encoder


struct LatentODE_decoder{D,O} <: AbstractDecoder

    diffeq::D
    layer_output::O

    function LatentODE_decoder(decoder_layers, diffeq)
        D = typeof(diffeq)
        O = typeof(decoder_layers)
        new{D,O}(diffeq, decoder_layers)
    end
end

function (decoder::LatentODE_decoder)(l̃, t)

    ẑ₀ = l̃

    ẑ = diffeq_layer(decoder, ẑ₀, t)

    ## Create output data shape
    x̂ = decoder.layer_output.(ẑ)

    return (x̂, ẑ, ẑ₀,)
end

function diffeq_layer(decoder::LatentODE_decoder, ẑ₀, t)
    dudt = decoder.diffeq.dudt
    solver = decoder.diffeq.solver
    neural_model = decoder.diffeq.neural_model
    # sensealg = decoder.diffeq.sensealg

    # nODE = NeuralODE(dudt, (t[1], t[end]), solver, sensealg = sensealg, saveat = t)
    nODE = neural_model(dudt, (t[1], t[end]), solver, saveat = t)
    ẑ = Array(nODE(ẑ₀))

    # Transform the resulting output (Mainly used for Kuramoto system to pass from phase -> time space)
    transform_after_diffeq!(ẑ, decoder.diffeq)
    ẑ = Flux.unstack(ẑ, 3)

    return ẑ
end

Flux.@functor LatentODE_decoder

built_encoder(model_type::LatentODE, encoder_layers) = LatentODE_encoder(encoder_layers)
built_decoder(model_type::LatentODE, decoder_layers, diffeq) = LatentODE_decoder(decoder_layers, diffeq)

function variational(model_type::LatentODE, μ::T, logσ²::T) where T <: Tuple{Flux.CUDA.CuArray}
    z₀_μ, = μ
    z₀_logσ², = logσ²

    ẑ₀ = z₀_μ + gpu(randn(Float32, size(z₀_logσ²))) .* exp.(z₀_logσ²/2f0)

    return ẑ₀
end

function variational(model_type::LatentODE, μ::T, logσ²::T) where T <: Tuple{Array}
    z₀_μ, = μ
    z₀_logσ², = logσ²

    ẑ₀ = z₀_μ + randn(Float32, size(z₀_logσ²)) .* exp.(z₀_logσ²/2f0)

    return ẑ₀
end


function default_layers(model_type::LatentODE, input_dim, diffeq, device;
                            hidden_dim = 200, rnn_input_dim = 32,
                            rnn_output_dim = 16,
                            latent_to_diffeq_dim = 200, θ_activation = x -> 5*σ(x),
                            output_activation = σ)

    latent_dim = diffeq.latent_dim

    ######################
    ### Encoder layers ###
    ######################

    layer1 = Chain(Dense(input_dim, hidden_dim, relu),
                        Dense(hidden_dim, rnn_input_dim, relu)) |> device
    layer2_z₀ = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                        RNN(rnn_output_dim, rnn_output_dim, relu)) |> device
    layer3_μ_z₀ = Dense(rnn_output_dim, latent_dim) |> device
    layer3_logσ²_z₀ = Dense(rnn_output_dim, latent_dim) |> device

    encoder_layers = (layer1, layer2_z₀, layer3_μ_z₀, layer3_logσ²_z₀)

    ######################
    ### Decoder layers ###
    ######################

    layer_output = Chain(Dense(latent_dim, hidden_dim, relu),
                    Dense(hidden_dim, input_dim)) |> device

    decoder_layers = layer_output
    # should we do:
    # diffeq.dudt = diffeq.dudt |> device
    # so that we control here the usage of GPU
    # instead of during the definition of the system
    # However, we are supposed to use ! in the name
    # of the function but that bring troubles with the other calls of default_layers
    return encoder_layers, decoder_layers
end
