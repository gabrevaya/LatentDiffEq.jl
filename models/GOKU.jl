# GOKU-NET MODEL
#
# Based on
# https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl
# https://arxiv.org/abs/1806.07366
# https://arxiv.org/abs/2003.10775

struct GOKU_encoder{L1,L2,L3,L4,L5,L6,L7,L8} <: AbstractEncoder

    layer1::L1
    layer2_z₀::L2
    layer2_θ_forward::L3
    layer2_θ_backward::L4
    layer3_μ_z₀::L5
    layer3_logσ²_z₀::L6
    layer3_μ_θ::L7
    layer3_logσ²_θ::L8

    function GOKU_encoder(encoder_layers)
        L1,L2,L3,L4,L5,L6,L7,L8 = typeof.(encoder_layers)
        new{L1,L2,L3,L4,L5,L6,L7,L8}(encoder_layers...)
    end
end

function (encoder::GOKU_encoder)(x)

    # Pass all states in the time series in dense layer
    l1_out = encoder.layer1.(x)

    # Pass an RNN and an BiLSTM through latent states
    l2_z₀_out, l2_θ_out = apply_layers2(encoder, l1_out)

    # Return RNN/BiLSTM ouput passed trough dense layers
    z₀_μ = encoder.layer3_μ_z₀(l2_z₀_out)
    z₀_logσ² = encoder.layer3_logσ²_z₀(l2_z₀_out)

    θ_μ = encoder.layer3_μ_θ(l2_θ_out)
    θ_logσ² = encoder.layer3_logσ²_θ(l2_θ_out)

    (z₀_μ, θ_μ), (z₀_logσ², θ_logσ²)
end


# rewrite this. Test performance when computing the layer2_z₀ (RNN) and layer2_θ (BiLSTM) separately
# at the cost of calculating again the reverse(l1_out) 
function apply_layers2(encoder::GOKU_encoder, l1_out)
    # reverse sequence
    l1_out_rev = reverse(l1_out)

    # pass the 
    l2_z₀_out = encoder.layer2_z₀.(l1_out_rev)[end]
    l2_θ_out_f = encoder.layer2_θ_forward.(l1_out)[end]
    l2_θ_out_b = encoder.layer2_θ_backward.(l1_out_rev)[end]
    l2_θ_out = vcat(l2_θ_out_f, l2_θ_out_b)

    reset!(encoder.layer2_z₀)
    reset!(encoder.layer2_θ_forward)
    reset!(encoder.layer2_θ_backward)

    return l2_z₀_out, l2_θ_out
end

Flux.@functor GOKU_encoder


struct GOKU_decoder{Z,T,O,D} <: AbstractDecoder

    layer_z₀::Z
    layer_θ::T

    layer_output::O

    diffeq::D

    function GOKU_decoder(decoder_layers, diffeq)
        Z,T,O = typeof.(decoder_layers)
        # D = typeof(diffeq)
        D = typeof(diffeq)
        new{Z,T,O,D}(decoder_layers..., diffeq)
    end
end

function (decoder::GOKU_decoder)(l̃, t)

    z̃₀, θ̃ = l̃

    ## Pass sampled latent states in dense layers
    ẑ₀ = decoder.layer_z₀(z̃₀)
    θ̂ = decoder.layer_θ(θ̃)

    ẑ = diffeq_layer(decoder, ẑ₀, θ̂, t)

    ## Create output data shape
    x̂ = decoder.layer_output.(ẑ)

    return x̂, ẑ, ẑ₀, θ̂
end

function diffeq_layer(decoder::GOKU_decoder, ẑ₀, θ̂, t)
    prob = decoder.diffeq.prob
    solver = decoder.diffeq.solver
    sensealg = decoder.diffeq.sensealg

    # Function definition for ensemble problem
    prob_func(prob,i,repeat) = remake(prob, u0=ẑ₀[:,i], p = θ̂[:,i]) # TODO: try using views and switching indexes to see if the performance improves
    
    # Check if solve was successful, if not, return NaNs to avoid problems with dimensions matches
    output_func(sol, i) = sol.retcode == :Success ? (Array(sol), false) : (fill(NaN32,(size(ẑ₀, 1), length(t))), false)  # check if this is compatible with CUDA (probably not)

    ## Adapt problem to given time span and create ensemble problem definition
    prob = remake(prob; tspan = (t[1],t[end]))
    ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)

    ## Solve
    ẑ = solve(ens_prob, solver, EnsembleSerial(), sensealg = sensealg, trajectories = size(θ̂, 2), saveat = t) # |> device I THINK THAT IF u0 IS A CuArray, then the solution will be a CuArray and there will be no need for this |> device. Although maybe the outer array is not a Cuda one and that would be needed.
    # Transform the resulting output (Mainly used for Kuramoto system to pass from phase -> time space)
    transform_after_diffeq!(ẑ, decoder.diffeq)

    ẑ = Flux.unstack(ẑ, 2)

    return ẑ
end

# Think how pass the ensemble_parallel argument
# maybe with a function like
# ensemble_parallel(u0::CuArray) = EnsembleGPUArray()
# ensemble_parallel(u0::Array) = EnsembleSerial()

# nothing by default (different method for Kuramoto)
transform_after_diffeq!(x, diffeq) = nothing

# has_transform(x) = error("Not implemented.")

Flux.@functor GOKU_decoder

built_encoder(model_type::GOKU, encoder_layers) = GOKU_encoder(encoder_layers)
built_decoder(model_type::GOKU, decoder_layers, diffeq) = GOKU_decoder(decoder_layers, diffeq)

function variational(model_type::GOKU, μ::T, logσ²::T) where T <: Flux.CUDA.CuArray
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + gpu(randn(Float32, size(z₀_logσ²))) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + gpu(randn(Float32, size( θ_logσ²))) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

function variational(model_type::GOKU, μ::T, logσ²::T) where T <: Array
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + randn(Float32, size(z₀_logσ²)) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + randn(Float32, size( θ_logσ²)) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end


function deafault_layers(model_type::GOKU, input_dim, diffeq, device;
                            hidden_dim_resnet = 200, rnn_input_dim = 32,
                            rnn_output_dim = 16, latent_dim = 16,
                            latent_to_diffeq_dim = 200, θ_activation = x -> 5*σ(x),
                            output_activation = σ)

    z_dim = length(diffeq.prob.u0)
    θ_dim = length(diffeq.prob.p)

    ######################
    ### Encoder layers ###
    ######################
    # Resnet
    l1 = Dense(input_dim, hidden_dim_resnet, relu)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu)
    l4 = Dense(hidden_dim_resnet, rnn_input_dim, relu)
    layer1 = Chain(l1,
                    SkipConnection(l2, +),
                    SkipConnection(l3, +),
                    l4) |> device

    # RNN
    layer2_z₀ = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                       RNN(rnn_output_dim, rnn_output_dim, relu)) |> device

    # for building a Bidirectional LSTM
    layer2_θ_forward = Chain(LSTM(rnn_input_dim, rnn_output_dim),
                       LSTM(rnn_output_dim, rnn_output_dim)) |> device

    layer2_θ_backward = Chain(LSTM(rnn_input_dim, rnn_output_dim),
                        LSTM(rnn_output_dim, rnn_output_dim)) |> device

    # final linear layers
    layer3_μ_z₀ = Dense(rnn_output_dim, latent_dim) |> device
    layer3_logσ²_z₀ = Dense(rnn_output_dim, latent_dim) |> device
    
    layer3_μ_θ = Dense(rnn_output_dim*2, latent_dim) |> device
    layer3_logσ²_θ = Dense(rnn_output_dim*2, latent_dim) |> device
    
    encoder_layers = (layer1, layer2_z₀, layer2_θ_forward, layer2_θ_backward,
                        layer3_μ_z₀, layer3_logσ²_z₀, layer3_μ_θ, layer3_logσ²_θ)

    ######################
    ### Decoder layers ###
    ######################

    # post variational but pre diff eq layer
    layer_z₀ = Chain(Dense(latent_dim, latent_to_diffeq_dim, relu),
                        Dense(latent_to_diffeq_dim, z_dim)) |> device

    layer_θ = Chain(Dense(latent_dim, latent_to_diffeq_dim, relu),
                        Dense(latent_to_diffeq_dim, θ_dim, θ_activation)) |> device

    # going back to the input dimensions
    # Resnet
    l1 = Dense(z_dim, hidden_dim_resnet, relu)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu)
    l4 = Dense(hidden_dim_resnet, input_dim, output_activation)
    layer_output = Chain(l1,
                    SkipConnection(l2, +),
                    SkipConnection(l3, +),
                    l4)  |> device

    decoder_layers = (layer_z₀, layer_θ, layer_output)

    return encoder_layers, decoder_layers
end
