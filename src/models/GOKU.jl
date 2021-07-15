# GOKU-NET MODEL
#
# Based on
# https://arxiv.org/abs/2003.10775

struct GOKU <: LatentDE end

struct GOKU_encoder{FE,PE,LI} <: AbstractEncoder

    feature_extractor::FE
    pattern_extractor::PE
    latent_in::LI

    function GOKU_encoder(encoder_layers)
        FE, PE, LI = typeof.(encoder_layers)
        new{FE,PE,LI}(encoder_layers...)
    end
end

function (encoder::GOKU_encoder)(x)

    # Pass every time frame independently through the feature extractor
    fe_out = encoder.feature_extractor.(x)

    # Process sequentially with the pattern extractor
    pe_z₀_out, pe_θ_out = apply_pattern_extractor(encoder, fe_out)

    # Pass trough a last layer before sampling
    μ, logσ² = apply_latent_in(encoder, pe_z₀_out, pe_θ_out)

    return μ, logσ²
end

# Test performance when computing the pe_z₀ (RNN) and pe_θ (BiLSTM) separately
# at the cost of calculating again the reverse(l1_out) 
function apply_pattern_extractor(encoder::GOKU_encoder, fe_out)
    pe_z₀, pe_θ_forward, pe_θ_backward = encoder.pattern_extractor

    # reverse sequence
    fe_out_rev = reverse(fe_out)

    # pass it through the recurrent layers
    pe_z₀_out = map(pe_z₀, fe_out_rev)[end]
    pe_θ_out_f = map(pe_θ_forward, fe_out)[end]
    pe_θ_out_b = map(pe_θ_backward, fe_out_rev)[end]
    pe_θ_out = vcat(pe_θ_out_f, pe_θ_out_b)

    # reset hidden states
    reset!(pe_z₀)
    reset!(pe_θ_forward)
    reset!(pe_θ_backward)

    return pe_z₀_out, pe_θ_out
end

function apply_latent_in(encoder::GOKU_encoder, pe_z₀_out, pe_θ_out)
    li_μ_z₀, li_logσ²_z₀, li_μ_θ, li_logσ²_θ = encoder.latent_in

    z₀_μ = li_μ_z₀(pe_z₀_out)
    z₀_logσ² = li_logσ²_z₀(pe_z₀_out)

    θ_μ = li_μ_θ(pe_θ_out)
    θ_logσ² = li_logσ²_θ(pe_θ_out)

    return (z₀_μ, θ_μ), (z₀_logσ², θ_logσ²)
end

Flux.@functor GOKU_encoder

struct GOKU_decoder{LI,R,D} <: AbstractDecoder

    latent_out::LI
    reconstructor::R
    diffeq::D

    function GOKU_decoder(decoder_layers, diffeq)
        LI, R = typeof.(decoder_layers)
        D = typeof(diffeq)
        new{LI,R,D}(decoder_layers..., diffeq)
    end
end

function (decoder::GOKU_decoder)(l̃, t)

    z̃₀, θ̃ = l̃

    ## Pass sampled latent states throue a latent_out layer
    ẑ₀, θ̂ = apply_latent_out(decoder, z̃₀, θ̃)

    ## Integrate differential equations
    ẑ = diffeq_layer(decoder, ẑ₀, θ̂, t)

    ## Apply reconstructor independently to each time frame
    x̂ = decoder.reconstructor.(ẑ)

    return x̂, ẑ, ẑ₀, θ̂
end

function apply_latent_out(decoder::GOKU_decoder, z̃₀, θ̃)
    lo_z₀, lo_θ = decoder.latent_out

    ẑ₀ = lo_z₀(z̃₀)
    θ̂ = lo_θ(θ̃)

    return ẑ₀, θ̂
end

function diffeq_layer(decoder::GOKU_decoder, ẑ₀, θ̂, t)
    prob = decoder.diffeq.prob
    solver = decoder.diffeq.solver
    sensealg = decoder.diffeq.sensealg

    # Function definition for ensemble problem
    prob_func(prob,i,repeat) = remake(prob, u0=ẑ₀[:,i], p = θ̂[:,i])
    
    # Check if solve was successful, if not, return NaNs to avoid problems with dimensions matches
    output_func(sol, i) = sol.retcode == :Success ? (Array(sol), false) : (fill(NaN32,(size(ẑ₀, 1), length(t))), false)

    ## Adapt problem to given time span and create ensemble problem definition
    prob = remake(prob; tspan = (t[1],t[end]))
    ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)

    ## Solve
    ẑ = solve(ens_prob, solver, EnsembleThreads(), sensealg = sensealg, trajectories = size(θ̂, 2), saveat = t)
    
    # Transform the resulting output (mainly used for Kuramoto-like systems)
    ẑ = transform_after_diffeq(ẑ, decoder.diffeq)
    ẑ = Flux.unstack(ẑ, 2)
    return ẑ
end

# Identity by default
transform_after_diffeq(x, diffeq) = x

Flux.@functor GOKU_decoder

built_encoder(model_type::GOKU, encoder_layers) = GOKU_encoder(encoder_layers)
built_decoder(model_type::GOKU, decoder_layers, diffeq) = GOKU_decoder(decoder_layers, diffeq)

function variational(μ::T, logσ²::T, model::LatentDiffEqModel{GOKU}) where T <: Tuple{Flux.CUDA.CuArray}
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + gpu(randn(Float32, size(z₀_logσ²))) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + gpu(randn(Float32, size( θ_logσ²))) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

function variational(μ::T, logσ²::T, model::LatentDiffEqModel{GOKU}) where T <: Tuple{Array}
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + randn(Float32, size(z₀_logσ²)) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + randn(Float32, size( θ_logσ²)) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

function default_layers(model_type::GOKU, input_dim, diffeq, device;
                            hidden_dim_resnet = 200, rnn_input_dim = 32,
                            rnn_output_dim = 16, latent_dim = 16,
                            latent_to_diffeq_dim = 200, θ_activation = softplus,
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
    feature_extractor = Chain(l1,
                    SkipConnection(l2, +),
                    SkipConnection(l3, +),
                    l4) |> device

    # RNN
    pe_z₀ = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                       RNN(rnn_output_dim, rnn_output_dim, relu)) |> device

    # Bidirectional LSTM
    pe_θ_forward = Chain(LSTM(rnn_input_dim, rnn_output_dim),
                       LSTM(rnn_output_dim, rnn_output_dim)) |> device

    pe_θ_backward = Chain(LSTM(rnn_input_dim, rnn_output_dim),
                        LSTM(rnn_output_dim, rnn_output_dim)) |> device

    pattern_extractor = (pe_z₀, pe_θ_forward, pe_θ_backward)

    # final fully connected layers before sampling
    li_μ_z₀ = Dense(rnn_output_dim, latent_dim) |> device
    li_logσ²_z₀ = Dense(rnn_output_dim, latent_dim) |> device
    
    li_μ_θ = Dense(rnn_output_dim*2, latent_dim) |> device
    li_logσ²_θ = Dense(rnn_output_dim*2, latent_dim) |> device

    latent_in = (li_μ_z₀, li_logσ²_z₀, li_μ_θ, li_logσ²_θ)

    encoder_layers = (feature_extractor, pattern_extractor, latent_in)

    ######################
    ### Decoder layers ###
    ######################

    # after sampling in the latent space but before the differential equation layer
    lo_z₀ = Chain(Dense(latent_dim, latent_to_diffeq_dim, relu),
                        Dense(latent_to_diffeq_dim, z_dim)) |> device

    lo_θ = Chain(Dense(latent_dim, latent_to_diffeq_dim, relu),
                        Dense(latent_to_diffeq_dim, θ_dim, θ_activation)) |> device

    latent_out = (lo_z₀, lo_θ)

    # going back to the input space
    # Resnet
    l1 = Dense(z_dim, hidden_dim_resnet, relu)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu)
    l4 = Dense(hidden_dim_resnet, input_dim, output_activation)
    reconstructor = Chain(l1,
                    SkipConnection(l2, +),
                    SkipConnection(l3, +),
                    l4)  |> device

    decoder_layers = (latent_out, reconstructor)

    return encoder_layers, decoder_layers
end
