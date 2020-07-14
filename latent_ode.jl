# Latent ODE
#
# Based on
# https://github.com/FluxML/model-zoo/blob/master/vision/vae_mnist/vae_mnist.jl
# https://arxiv.org/abs/1806.07366
# https://arxiv.org/abs/2003.10775

include("create_data.jl")

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
using Zygote
using Plots
using CuArrays
using Distributions
using CUDAdrv
CuArrays.allowscalar(false)

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
    return Flux.unstack(data_, 2)
end

# # add a method to * so that we can compute the whole batch at once using batchee_mul
# import Base:*
# function *(W::AbstractArray{T,2}, x::AbstractArray{T,3}) where T
#     W_rep = repeat(W, 1, 1, size(x, 3))
#     batched_mul(W_rep, x)
# end

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

function reconstruct(encoder, decoder, x, device)
    μ, logσ² = encoder(x)
    z = μ + device(randn(Float32, size(logσ²))) .* exp.(logσ²/2f0)
    μ, logσ², decoder(z, device)
end

KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0
#the following works better for gpu
CuArrays.@cufunc KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# the calculation via log(var) = log(σ²) is more numerically efficient than through log(σ)
# KL(μ, logσ) = (exp(2f0 * logσ) + μ^2)/2f0 - 0.5f0 - logσ

# function rec_loss(x, pred_x)
#     res = Flux.stack(pred_x - x, 3)
#     sum(mean((res).^2, dims = (2, 3)))
# end

function rec_loss(x, pred_x)
    pred_x_stacked = Flux.stack(pred_x, 3)
    x_stacked = Flux.stack(x, 3)
    res = pred_x_stacked - x_stacked
    res_average = sum(mean((res).^2, dims = (2, 3)))
    res_diff = diff(pred_x_stacked, dims = 3) - diff(x_stacked, dims = 3)
    res_diff_average = sum(mean((res_diff).^2, dims = (2, 3)))
    return res_average + 1000*res_diff_average
end

# The following rec_loss is faster but Zygote has a problem with it.
# We should write a more performant loss function
# function rec_loss(x, pred_x)
#     sum(mean(mean([t.^2 for t in (pred_x - x)]), dims = 2))
# end

function loss_batch(encoder, decoder, λ, x, device)
    μ, logσ², pred_x = reconstruct(encoder, decoder, x, device)
    reconstruction_loss = rec_loss(x, pred_x)
    kl_loss = mean(sum(KL.(μ, logσ²), dims = 1))
    return reconstruction_loss + kl_loss
end

# arguments for the `train` function
@with_kw mutable struct Args
    η = 1e-3                    # learning rate
    λ = 0.05f0                  # regularization paramater
    batch_size = 256            # batch size
    seq_len = 100               # sampling size for output
    epochs = 100                # number of epochs
    seed = 1                    # random seed
    cuda = true                 # use GPU
    hidden_dim = 200            # hidden dimension
    rnn_input_dim = 32          # rnn input dimension
    rnn_output_dim = 32         # rnn output dimension
    latent_dim = 4              # latent dimension
    hidden_dim_node = 200       # hiddend dimension of the neuralODE
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
    loader_train = DataLoader(Array(train_set), batchsize=args.batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set,3), shuffle=true, partial=false)

    # initialize encoder and decoder
    encoder = Encoder(input_dim, args.latent_dim, args.hidden_dim, args.rnn_input_dim, args.rnn_output_dim, device)
    decoder = Decoder(input_dim, args.latent_dim, args.hidden_dim, args.hidden_dim_node, seq_len, args.t_max, device)

    # visualize a random sample from evaluation
    # and its reconstruction from the untrained model
    visualize_training(encoder, decoder, first(loader_val), device)

    # ADAM optimizer
    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(encoder.linear, encoder.rnn, encoder.μ, encoder.logσ²,
                     decoder.neuralODE, decoder.linear)

    # or using IterTools
    # ps = Flux.params(collect(fieldvalues(encoder)), collect(fieldvalues(decoder)))
    mkpath(args.save_path)

    best_val_loss::Float32 = Inf32
    val_loss::Float32 = 0

    # training
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
            val_loss = loss_batch(encoder, decoder, args.λ, first(loader_val) |> device, device)

            # progress meter
            next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])
        end

        visualize_training(encoder, decoder, first(loader_val), device)

        if val_loss < best_val_loss
            best_val_loss = deepcopy(val_loss)

            model_path = joinpath(args.save_path, "best_model.bson")
            let encoder = cpu(encoder),
                decoder = cpu(decoder),
                args=struct2dict(args)

                BSON.@save model_path encoder decoder args
                @info "Model saved: $(model_path)"
            end
        end
    end
end

function visualize_training(encoder, decoder, x, device)

    j = rand(1:size(x[1],2))
    xᵢ = [ x[i][:,j] for i in 1:size(x, 1)]

    μ, logσ², z = reconstruct(encoder, decoder, xᵢ |> device, device)
    xₐ = Flux.stack(xᵢ, 2)
    zₐ = z[1]

    plt = compare_sol(xₐ, cpu(zₐ));
    display(plt) # when displaying in Atom, it prints Plot{Plots.GRBackend() n=4}
    # TODO disable the printing of Plot{Plots.GRBackend() n=4}
    # maybe we could also plot after each epoch

    # png(plt, "Training_sample.png")
end

function import_model(model_path, input_dim, device)

    @load model_path encoder decoder args

    encoder_new = Encoder(input_dim, args[:latent_dim], args[:hidden_dim], args[:rnn_input_dim], args[:rnn_output_dim], device)
    decoder_new = Decoder(input_dim, args[:latent_dim], args[:hidden_dim], args[:hidden_dim_node], args[:seq_len], args[:t_max], device)

    Flux.loadparams!(encoder_new.linear, Flux.params(encoder.linear))
    Flux.loadparams!(encoder_new.rnn, Flux.params(encoder.rnn))
    Flux.loadparams!(encoder_new.μ, Flux.params(encoder.μ))
    Flux.loadparams!(encoder_new.logσ², Flux.params(encoder.logσ²))
    Flux.loadparams!(decoder_new.neuralODE, Flux.params(decoder.neuralODE))
    Flux.loadparams!(decoder_new.linear, Flux.params(decoder.linear))

    encoder_new, decoder_new
end

function predict_from_train()

    #GPU config
    model_pth = "output/model_epoch_100.bson"
    @load model_pth args
    if args[:cuda] && has_cuda_gpu()
        device = gpu
        @info "Evaluating on GPU"
    else
        device = cpu
        @info "Evaluating on CPU"
    end

    # Load a random sample from training set
    @load "lv_data.bson" full_data

    sol = full_data[:,:,rand(1:10000)]

    # Load model
    input_dim = size(sol, 1)
    encoder, decoder = import_model(model_pth, input_dim, device)

    # Predict within time interval given
    x = Flux.unstack(reshape(sol, (size(sol, 1),size(sol, 2), 1)), 2)
    μ, logσ², z = reconstruct(encoder, decoder, x |> device, device)

    # Data dimensions manipulation
    x = dropdims(Flux.stack(x, 2), dims=3)
    z = dropdims(Flux.stack(z, 2), dims=3)

    # Showing in plot panel
    plt = compare_sol(x,z)

    png(plt, "prediction_from_train.png")
end

function compare_sol(x, z)

    plt = plot(x[1,:], color="blue", label="True x")
    plt = plot!(x[2,:], color="red", label="True y")
    plt = plot!(z[1,:], color="blue", linestyle=:dot, label="Model x")
    plt = plot!(z[2,:], color="red", linestyle=:dot, label="Model y")

    plt

end


## Function comparing solution generated from latentODE structure with true solution within the time span
function predict_within()

    # Load model
    model_pth = "output/model_epoch_100.bson"
    @load model_pth args

    if args[:cuda] && has_cuda_gpu()
        device = gpu
        @info "Evaluating on GPU"
    else
        device = cpu
        @info "Evaluating on CPU"
    end

    @load args[:data_name] p

    u0 = rand(Uniform(1.5, 3.0), 2, 1) # [x, y]
    tspan = (0.0, 9.95)
    tstep = 0.1

    sol = Array(solve_prob(u0, p, tspan, tstep))

    input_dim = size(sol, 1)
    encoder, decoder = import_model(model_pth, input_dim, device)

    # Predict within time interval given
    x = Flux.unstack(reshape(sol, (size(sol, 1),size(sol, 2), 1)), 2)
    μ, logσ², z = reconstruct(encoder, decoder, x |> device, device)

    # Data dimensions manipulation
    x = dropdims(Flux.stack(x, 2), dims=3)
    z = dropdims(Flux.stack(z, 2), dims=3)

    # Showing in plot panel
    plt = compare_sol(x, z)

    png(plt, "prediction_outside_train.png")

end

const seq_len = 100

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
