# GOKU-net model
#
# Based on
# https://arxiv.org/abs/2003.10775

struct GOKU <: LatentDE end

apply_feature_extractor(encoder::Encoder{GOKU}, x) = encoder.feature_extractor.(x)

@doc raw"""
    apply_pattern_extractor(encoder::Encoder{GOKU}, fe_out)

Passes `fe_out` through the pattern_extractor layer contained in the `encoder`, returning a tuple associated to the initial latent states and parameters, respectively.

# Arguments
`encoder`: Encoder structure containing all the encoder layers.\
`fe_out`: Output of feature extractor layer.
"""
function apply_pattern_extractor(encoder::Encoder{GOKU}, fe_out)
    pe_z₀, pe_θ_forward, pe_θ_backward = encoder.pattern_extractor

    # reverse sequence
    fe_out_rev = reverse(fe_out)

    # pass it through the recurrent layers
    pe_z₀_out = map(pe_z₀, fe_out_rev)[end]
    pe_θ_out_f = map(pe_θ_forward, fe_out)[end]
    pe_θ_out_b = map(pe_θ_backward, fe_out_rev)[end]
    pe_θ_out = vcat(pe_θ_out_f, pe_θ_out_b)

    # reset hidden states
    Flux.reset!(pe_z₀)
    Flux.reset!(pe_θ_forward)
    Flux.reset!(pe_θ_backward)

    return pe_z₀_out, pe_θ_out
end

function apply_latent_in(encoder::Encoder{GOKU}, pe_out)
    pe_z₀_out, pe_θ_out = pe_out
    li_μ_z₀, li_logσ²_z₀, li_μ_θ, li_logσ²_θ = encoder.latent_in

    z₀_μ = li_μ_z₀(pe_z₀_out)
    z₀_logσ² = li_logσ²_z₀(pe_z₀_out)

    θ_μ = li_μ_θ(pe_θ_out)
    θ_logσ² = li_logσ²_θ(pe_θ_out)

    return (z₀_μ, θ_μ), (z₀_logσ², θ_logσ²)
end

@doc raw"""
    apply_latent_out(decoder::Decoder{GOKU}, l̃)
Applies the `decoder`'s `latent_out` layer to `l̃`, returning a tuple with initial conditions and parameters for use in the differential equation layer.

# Arguments
`decoder`: Decoder structure containing all the decoder layers.\
`l̃`: Tuple containing sampled abstract representations of the initial conditions and parameters, respectively.
"""
function apply_latent_out(decoder::Decoder{GOKU}, l̃)
    z̃₀, θ̃ = l̃
    lo_z₀, lo_θ = decoder.latent_out

    ẑ₀ = lo_z₀(z̃₀)
    θ̂ = lo_θ(θ̃)

    return ẑ₀, θ̂
end

@doc raw"""
    diffeq_layer(decoder::Decoder{GOKU}, l̂, t)

Solves the differential equations contained in the `diffeq` layer of the `decoder` using the initial conditions and parameters contained in `l̂`, and saving at times `t`.
"""
function diffeq_layer(decoder::Decoder{GOKU}, l̂, t)
    ẑ₀, θ̂ = l̂
    prob = decoder.diffeq.prob
    solver = decoder.diffeq.solver
    sensealg = decoder.diffeq.sensealg
    kwargs = decoder.diffeq.kwargs

    # Function definition for ensemble problem
    prob_func(prob,i,repeat) = remake(prob, u0=ẑ₀[:,i], p = θ̂[:,i])
    
    # Check if solve was successful, if not, return NaNs to avoid problems with dimensions matches
    output_func(sol, i) = sol.retcode == :Success ? (Array(sol), false) : (fill(NaN32,(size(ẑ₀, 1), length(t))), false)

    ## Adapt problem to given time span and create ensemble problem definition
    prob = remake(prob; tspan = (t[1],t[end]))
    ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)

    ## Solve
    ẑ = solve(ens_prob, solver, EnsembleThreads(); sensealg = sensealg, trajectories = size(θ̂, 2), saveat = t, kwargs...)
    
    # Transform the resulting output (mainly used for Kuramoto-like systems)
    ẑ = transform_after_diffeq(ẑ, decoder.diffeq)
    ẑ = Flux.unstack(ẑ, 2)
    return ẑ
end

# Identity by default
transform_after_diffeq(x, diffeq) = x

apply_reconstructor(decoder::Decoder{GOKU}, ẑ) = decoder.reconstructor.(ẑ)

function sample(μ::T, logσ²::T, model::LatentDiffEqModel{GOKU}) where T <: Tuple{Array, Array}
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + randn(Float32, size(z₀_logσ²)) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + randn(Float32, size( θ_logσ²)) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

function sample(μ::T, logσ²::T, model::LatentDiffEqModel{GOKU}) where T <: Tuple{Flux.CUDA.CuArray, Flux.CUDA.CuArray}
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + gpu(randn(Float32, size(z₀_logσ²))) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + gpu(randn(Float32, size( θ_logσ²))) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

function default_layers(model_type::GOKU, input_dim, diffeq; device=cpu,
                            hidden_dim_resnet = 200, rnn_input_dim = 32,
                            rnn_output_dim = 16, latent_dim = 16,
                            latent_to_diffeq_dim = 200, θ_activation = softplus,
                            output_activation = σ, init = Flux.kaiming_uniform(gain = 1/sqrt(3)))

    z_dim = length(diffeq.prob.u0)
    θ_dim = length(diffeq.prob.p)

    ######################
    ### Encoder layers ###
    ######################
    # Resnet
    l1 = Dense(input_dim, hidden_dim_resnet, relu, init = init)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu, init = init)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu, init = init)
    l4 = Dense(hidden_dim_resnet, rnn_input_dim, relu, init = init)
    feature_extractor = Chain(l1,
                                SkipConnection(l2, +),
                                SkipConnection(l3, +),
                                l4) |> device

    # RNN
    pe_z₀ = Chain(RNN(rnn_input_dim, rnn_output_dim, relu, init = init),
                       RNN(rnn_output_dim, rnn_output_dim, relu, init = init)) |> device
    
    # Bidirectional LSTM
    pe_θ_forward = Chain(LSTM(rnn_input_dim, rnn_output_dim, init = init),
                       LSTM(rnn_output_dim, rnn_output_dim, init = init)) |> device

    pe_θ_backward = Chain(LSTM(rnn_input_dim, rnn_output_dim, init = init),
                        LSTM(rnn_output_dim, rnn_output_dim, init = init)) |> device

    pattern_extractor = (pe_z₀, pe_θ_forward, pe_θ_backward)

    # final fully connected layers before sampling
    li_μ_z₀ = Dense(rnn_output_dim, latent_dim, init = init) |> device
    li_logσ²_z₀ = Dense(rnn_output_dim, latent_dim, init = init) |> device
    
    li_μ_θ = Dense(rnn_output_dim*2, latent_dim, init = init) |> device
    li_logσ²_θ = Dense(rnn_output_dim*2, latent_dim, init = init) |> device

    latent_in = (li_μ_z₀, li_logσ²_z₀, li_μ_θ, li_logσ²_θ)

    encoder_layers = (feature_extractor, pattern_extractor, latent_in)

    ######################
    ### Decoder layers ###
    ######################

    # after sampling in the latent space but before the differential equation layer
    lo_z₀ = Chain(Dense(latent_dim, latent_to_diffeq_dim, relu, init = init),
                        Dense(latent_to_diffeq_dim, z_dim, init = init)) |> device

    lo_θ = Chain(Dense(latent_dim, latent_to_diffeq_dim, relu, init = init),
                        Dense(latent_to_diffeq_dim, θ_dim, θ_activation, init = init)) |> device

    latent_out = (lo_z₀, lo_θ)

    # going back to the input space
    # Resnet
    l1 = Dense(z_dim, hidden_dim_resnet, relu, init = init)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu, init = init)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, relu, init = init)
    l4 = Dense(hidden_dim_resnet, input_dim, output_activation, init = init)
    reconstructor = Chain(l1,
                            SkipConnection(l2, +),
                            SkipConnection(l3, +),
                            l4)  |> device

    decoder_layers = (latent_out, diffeq, reconstructor)

    return encoder_layers, decoder_layers
end