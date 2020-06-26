# Latent ODE
#
# Based on
# https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl
# https://arxiv.org/abs/1806.07366
# https://arxiv.org/abs/2003.10775

using OrdinaryDiffEq
using Base.Iterators: partition
using BSON:@save, @load
using BSON
using CUDAapi: has_cuda_gpu
using DrWatson: struct2dict
using DiffEqFlux
using Flux
using Flux.Data: DataLoader
import Flux.Data: _getobs
using Flux: reset!
using Logging: with_logger
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random
using MLDataUtils
using Statistics



# overload data loader function so that it picks random start times for each
# sample, of size seq_len

# Flux needs to be in v0.11.0 (currently master, which is not compatible with
# DiffEqFlux compatibility, that's why I didn't include it in the Project.toml)

function time_idxs(seq_len, time_len)
    start_time = rand(1:time_len - seq_len)
    idxs = start_time:start_time+seq_len-1
end

function Flux.Data._getobs(data::AbstractArray, i)
    features, time_len, obs = size(data)
    seq_len::Int
    data_ = Array{Float32, 3}(undef, (features, seq_len, length(i)))

    for (idx, batch_idx) in enumerate(i)
        data_[:,:, idx] =
        data[:, time_idxs(seq_len, time_len), batch_idx]
    end
    return data_
end

struct Encoder{F,G,H}
    linear::F
    rnn::G
    μ::H
    logσ²::H

    function Encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)

        linear = Chain(Dense(input_dim, hidden_dim, relu),
                       Dense(hidden_dim, rnn_input_dim, relu))|> device         # linear
        rnn = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                    RNN(rnn_output_dim, rnn_output_dim, relu))|> device      # rnn
        μ = Dense(rnn_output_dim, latent_dim) |> device                 # μ
        logσ² = Dense(rnn_output_dim, latent_dim) |> device                     # logσ²
        F = typeof(linear)
        G = typeof(rnn)
        H = typeof(μ)
        new{F,G,H}(linear, rnn, μ, logσ²)
    end
end

function (encoder::Encoder)(x)
    states, seq_len, samples = size(x)
    h1 = cat([encoder.linear(x[:,:,i]) for i in 1:samples]..., dims = 3)

    # pass all the samples of the batch,
    # one time slice at a time, in reverse order
    for time_slice in eachslice(h1[:,end:-1:1,:], dims = 2)
        encoder.rnn(time_slice)
    end

    h = encoder.rnn.layers[2].state
    reset!(encoder)
    encoder.μ(h), encoder.logσ²(h)
end

struct Decoder{F,G}
    neuralODE::F
    linear::G

    function Decoder(input_dim, latent_dim, hidden_dim, hidden_dim_node, time_size, t_max, device)

        dudt2 = Chain(Dense(latent_dim, hidden_dim_node, relu),
                        Dense(hidden_dim_node, hidden_dim_node, relu),
                        Dense(hidden_dim_node, latent_dim))
        tspan = (zero(t_max), t_max)
        t = range(tspan[1], tspan[2], length=time_size)

        node = NeuralODE(dudt2, tspan, Tsit5(), saveat = t) #|> device   # neuralODE
        linear = Chain(Dense(latent_dim, hidden_dim, relu),
                       Dense(hidden_dim, input_dim)) |> device          # linear

        F = typeof(node)
        G = typeof(linear)
        new{F,G}(node, linear)
        end
end

function (decoder::Decoder)(x)
    h = Array(decoder.neuralODE(x))
    samples = size(h, 2)
    out = cat([decoder.linear(h[:,i,:]) for i in 1:samples]..., dims = 3)
end

function reconstruct(encoder, decoder, x, device)
    μ, logσ² = encoder(x)
    z = μ + device(randn(Float32, size(logσ²))) .* exp.(logσ²/2f0)
    μ, logσ², decoder(z)
end

KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0
# the calculation via log(var) = log(σ²) is more numerically efficient than through log(σ)
# KL(μ, logσ) = (exp(2f0 * logσ) + μ^2)/2f0 - 0.5f0 - logσ

function loss_batch(encoder, decoder, λ, x, device)
    μ, logσ², pred_x = reconstruct(encoder, decoder, x, device)
    # reconstruction loss
    reconstruction_loss = sum(mean((pred_x - x).^2, dims = (2,3)))
    # KL loss
    kl_loss = mean(sum(KL.(μ, logσ²), dims = 1))
    return reconstruction_loss + kl_loss
end



# arguments for the `train` function
@with_kw mutable struct Args
    η = 1e-3                    # learning rate
    λ = 0.01f0                  # regularization paramater
    batch_size = 256            # batch size
    seq_len = 100               # sampling size for output
    epochs = 20                 # number of epochs
    seed = 1                    # random seed
    cuda = true                 # use GPU
    hidden_dim = 10#200         # hidden dimension
    rnn_input_dim = 10#32       # rnn input dimension
    rnn_output_dim = 10#32      # rnn output dimension
    latent_dim = 2#4            # latent dimension
    hidden_dim_node = 100#200   # hiddend dimension of the neuralODE
    t_max = 4.95f0              # edge of time interval for integration
    save_path = "output"        # results path
    data_name = "lv_data.bson"  # data file name
    data_var_name = "full_data" # data file name
end


function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && has_cuda_gpu()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load data
    @load args.data_name full_data
    full_data = Float32.(full_data)
    input_dim, time_size, observations = size(full_data)
    train_set, test_set = splitobs(full_data, 0.9)
    train_set, val_set = splitobs(train_set, 0.9)

    # the following assumes that the data is (states, time, observations)
    # train_set = train_set[:,:,1:1000]
    loader_train = DataLoader(Array(train_set), batchsize=args.batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set,3), shuffle=true, partial=false)

    # initialize encoder and decoder
    encoder = Encoder(input_dim, args.latent_dim, args.hidden_dim, args.rnn_input_dim, args.rnn_output_dim, device)
    decoder = Decoder(input_dim, args.latent_dim, args.hidden_dim, args.hidden_dim_node, seq_len, args.t_max, device)

    # ADAM optimizer
    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(encoder.linear, encoder.rnn, encoder.μ, encoder.logσ²,
                     decoder.neuralODE, decoder.linear)

    # or using IterTools
    # ps = Flux.params(collect(fieldvalues(encoder)), collect(fieldvalues(decoder)))

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader_train))

        for x in loader_train
            loss, back = Flux.pullback(ps) do
                loss_batch(encoder, decoder, args.λ, x |> device, device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)])

            train_steps += 1
        end

        model_path = joinpath(args.save_path, "model_epoch_$(epoch).bson")
        let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
            BSON.@save model_path encoder decoder args
            @info "Model saved: $(model_path)"
        end
    end

    model_path = joinpath(args.save_path, "model.bson")
    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        BSON.@save model_path encoder decoder args
        @info "Model saved: $(model_path)"
    end
end

const seq_len = 100

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
