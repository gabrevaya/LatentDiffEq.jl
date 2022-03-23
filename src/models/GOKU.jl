# GOKU-net model
#
# Based on
# https://arxiv.org/abs/2003.10775

abstract type GOKU <: LatentDE end
struct GOKU_basic <: GOKU end

@doc raw"""
    apply_feature_extractor(encoder::Encoder{GOKU}, x)
    apply_feature_extractor(encoder::Encoder{LatentODE}, x)

Applies the feature extractor layer contained in the `encoder` to the batch of input data x, usually reducing its dimensionality (i.e. extracting features).

# Arguments
`encoder`: Encoder structure containing all the encoder layers.\
`x`: Input data. When using the default architecture, `x` correspond to an array of size `pixels` x `batch size` x `time`.
"""
apply_feature_extractor(encoder::Encoder{T}, x) where {T<:GOKU} = encoder.feature_extractor(x)

@doc raw"""
    apply_pattern_extractor(encoder::Encoder{GOKU}, fe_out)

Passes `fe_out` through the pattern_extractor layer contained in the `encoder`, returning a tuple associated to the initial latent states and parameters, respectively.

# Arguments
`encoder`: Encoder structure containing all the encoder layers.\
`fe_out`: Output of feature extractor layer.
"""
function apply_pattern_extractor(encoder::Encoder{T}, fe_out) where {T<:GOKU}
    pe_z₀, pe_θ_forward, pe_θ_backward = encoder.pattern_extractor
    fe_out = Flux.unstack(fe_out, 3)
    
    # reverse sequence
    fe_out_rev = reverse(fe_out)

    # pass it through the recurrent layers
    pe_z₀_out = [pe_z₀(x) for x in fe_out_rev][end]
    pe_θ_out_f = [pe_θ_forward(x) for x in fe_out][end]
    pe_θ_out_b = [pe_θ_backward(x) for x in fe_out_rev][end]
    pe_θ_out = vcat(pe_θ_out_f, pe_θ_out_b)

    # reset hidden states
    Flux.reset!(pe_z₀)
    Flux.reset!(pe_θ_forward)
    Flux.reset!(pe_θ_backward)

    return pe_z₀_out, pe_θ_out
end

@doc raw"""
    apply_latent_in(encoder::Encoder{GOKU}, pe_out)
    apply_latent_in(encoder::Encoder{LatentODE}, pe_out)

Applies the `encoder`'s `latent_in` layer to `pe_out`, returning the mean and log-variance of the latent variables to use for sampling.

# Arguments
`encoder`: Encoder structure containing all the encoder layers.\
`pe_out`: Output of pattern extractor layer.
"""
function apply_latent_in(encoder::Encoder{T}, pe_out) where {T<:GOKU}
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
function apply_latent_out(decoder::Decoder{T}, l̃) where {T<:GOKU}
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
function diffeq_layer(decoder::Decoder{T}, l̂, t) where {T<:GOKU}
    ẑ₀_, θ̂_ = l̂

    # make sure the diff eq  solving is done on cpu
    ẑ₀ = cpu(ẑ₀_)
    θ̂ = cpu(θ̂_)

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

    # Optionally transform the latent state variables
    ẑ = transform_after_diffeq(Array(ẑ), decoder.diffeq)
    ẑ = permutedims(ẑ, [1,3,2])

    # go back to gpu if it corresponds
    ẑ = device(ẑ₀_, ẑ)
    return ẑ
end

device(x::Flux.CUDA.CuArray, y) = gpu(y)
device(x::Array, y) = y

# Identity by default
transform_after_diffeq(x, diffeq) = x

@doc raw"""
    apply_reconstructor(decoder::Decoder{GOKU}, ẑ)
    apply_reconstructor(decoder::Decoder{LatentODE}, ẑ)

Passes latent trajectories `ẑ` through the reconstructor layer contained in the `decoder`.

# Arguments
`decoder`: Decoder structure containing all the decoder layers.\
`ẑ`: Latent trajectories, consists of matrices corresponding to different time frames and having size `latent data dimension` x `batch size`.
"""
apply_reconstructor(decoder::Decoder{T}, ẑ) where {T<:GOKU} = decoder.reconstructor(ẑ)

@doc raw"""
    sample(μ::T, logσ²::T, model::LatentDiffEqModel{LatentODE}) where T <: Array

Samples latent variables from the normal distribution with mean μ and variance exp(logσ²).
"""
function sample(μ::P, logσ²::P, model::LatentDiffEqModel{T}) where {P<:Tuple{Array, Array}, T<:GOKU}
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + randn(Float32, size(z₀_logσ²)) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + randn(Float32, size( θ_logσ²)) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

function sample(μ::P, logσ²::P, model::LatentDiffEqModel{T}) where {P<:Tuple{Flux.CUDA.CuArray, Flux.CUDA.CuArray}, T<:GOKU}
    z₀_μ, θ_μ = μ
    z₀_logσ², θ_logσ² = logσ²

    ẑ₀ = z₀_μ + gpu(randn(Float32, size(z₀_logσ²))) .* exp.(z₀_logσ²/2f0)
    θ̂ =  θ_μ + gpu(randn(Float32, size( θ_logσ²))) .* exp.(θ_logσ²/2f0)

    return ẑ₀, θ̂
end

@doc raw"""
    default_layers(model_type, input_dim::Int, diffeq;
        device = cpu,
        hidden_dim_resnet = 200, rnn_input_dim = 32,
        rnn_output_dim = 16, latent_dim = 16,
        latent_to_diffeq_dim = 200, θ_activation = softplus,
        output_activation = σ, init = Flux.kaiming_uniform(gain = 1/sqrt(3)))

Generates default encoder and decoder layers that are to be fed into the LatentDiffEqModel.

# Arguments
`model_type`: GOKU() or LatentODE()\
`input_dim`: Dimension of input\
`diffeq`: Differential equations structure, containing fields `prob`, `solver` and `sensealg`, which correspond to DifferentialEquations.jl's problem, solver and sensitivity algorithm, respectively.

# Keyword Arguments
`device`: Flux.jl's `cpu` or `gpu`.
`hidden_dim_resnet`: Hidden dimension of the feature_extractor's resnet .
`rnn_input_dim`: Input dimension of the pattern_extractor layers.
`latent_dim`: Diemension of the latent variables.
`latent_to_diffeq_dim`: Hidden dimension of the dense layers from latent_out.
`θ_activation`: Activation function used in the last dense layer from latent_out corresponding to the parameters of the differential equation `θ`, which can be useful for imposing contrains.
`output_activation`: Activation function used in the last layer of the reconstructor.
"""
function default_layers(model_type::GOKU_basic, input_dim, diffeq; device=cpu,
                            hidden_dim_resnet = 200, rnn_input_dim = 32,
                            rnn_output_dim = 16, latent_dim_z₀ = 16, latent_dim_θ = 16,
                            latent_to_diffeq_dim = 200, general_activation = relu,
                            z₀_activation = identity, θ_activation = softplus,
                            output_activation = σ, init = Flux.kaiming_uniform(gain = 1/sqrt(3)),
                            verbose = false)

    z_dim = length(diffeq.prob.u0)
    θ_dim = length(diffeq.prob.p)

    ######################
    ### Encoder layers ###
    ######################
    # Resnet
    l1 = Dense(input_dim, hidden_dim_resnet, general_activation, init = init)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l4 = Dense(hidden_dim_resnet, rnn_input_dim, general_activation, init = init)
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
    li_μ_z₀ = Dense(rnn_output_dim, latent_dim_z₀, init = init) |> device
    li_logσ²_z₀ = Dense(rnn_output_dim, latent_dim_z₀, init = init) |> device

    li_μ_θ = Dense(rnn_output_dim*2, latent_dim_θ, init = init) |> device
    li_logσ²_θ = Dense(rnn_output_dim*2, latent_dim_θ, init = init) |> device

    latent_in = (li_μ_z₀, li_logσ²_z₀, li_μ_θ, li_logσ²_θ)

    encoder_layers = (feature_extractor, pattern_extractor, latent_in)

    ######################
    ### Decoder layers ###
    ######################

    # after sampling in the latent space but before the differential equation layer
    lo_z₀ = Chain(Dense(latent_dim_z₀, latent_to_diffeq_dim, general_activation, init = init),
                  Dense(latent_to_diffeq_dim, z_dim, z₀_activation, init = init)) |> device

    lo_θ = Chain(Dense(latent_dim_θ, latent_to_diffeq_dim, general_activation, init = init),
                 Dense(latent_to_diffeq_dim, θ_dim, θ_activation, init = init)) |> device

    latent_out = (lo_z₀, lo_θ)

    # going back to the input space
    # Resnet
    l1 = Dense(z_dim, hidden_dim_resnet, general_activation, init = init)
    l2 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l3 = Dense(hidden_dim_resnet, hidden_dim_resnet, general_activation, init = init)
    l4 = Dense(hidden_dim_resnet, input_dim, output_activation, init = init)
    reconstructor = Chain(l1,
                            SkipConnection(l2, +),
                            SkipConnection(l3, +),
                            l4)  |> device

    decoder_layers = (latent_out, diffeq, reconstructor)

    return encoder_layers, decoder_layers
end
