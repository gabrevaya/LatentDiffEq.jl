# GOKU-NET MODEL
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
using ModelingToolkit
using DiffEqGPU

# overload data loader function so that it picks random start times for each
# sample, of size seq_len

# Flux needs to be in v0.11.0 (currently master, which is not compatible with
# DiffEqFlux compatibility, that's why I didn't include it in the Project.toml)

################################################################################
## Problem Definition


# @parameters t α β δ γ
# @variables x(t) y(t)
# @derivatives D'~t
#
# struct ODE_lv
#
#     sys
#     solver
#
#     function ODE_lv(solver)
#
#         eqs = [D(x) ~ α*x - β*x*y,
#                D(y) ~ -δ*y + γ*x*y]
#
#         sys = ODESystem(eqs)
#
#         sys = generate_function(sys, [x, y], [α, β, δ, γ], expression=Val{false})[2]
#
#         new(sys, solver)
#
#     end
#
# end
#
# function (ode_lv::ODE_lv)(z₀, t, p)
#
#     for i in 1:size(z₀,2)
#
#         var_z₀ = [x, y] => z₀[:,i]
#         var_p = [α, β, δ, γ] => p[:,i]
#
#         prob = ODEProblem(ode_lv.sys, var_z₀, t, var_p)
#         pred_z = solve(prob, ode_lv.solver, saveat=0.1)
#         print(size(pred_z))
#     end
# end

## Function parameters:
# z -> [1] : x(t)
#      [2] : y(t)
# p -> [1] : α
#      [2] : β
#      [3] : δ
#      [4] : γ
function lv_func(dz, z, p, t)
    @inbounds begin
        dz[1] = p[1]*z[1] - p[2]*z[1]*z[2]
        dz[2] = -p[3]*z[2] + p[4]*z[1]*z[2]
    end
    nothing
end

################################################################################
## Model definition

struct Encoder

    linear
    rnn
    rnn_μ
    rnn_logσ²
    lstm
    lstm_μ
    lstm_logσ²

    device

    function Encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)

        linear = Chain(Dense(input_dim, hidden_dim, relu),
                       Dense(hidden_dim, rnn_input_dim, relu)) |> device
        rnn = Chain(RNN(rnn_input_dim, rnn_output_dim, relu),
                    RNN(rnn_output_dim, rnn_output_dim, relu)) |> device

        rnn_μ = Dense(rnn_output_dim, latent_dim) |> device
        rnn_logσ² = Dense(rnn_output_dim, latent_dim) |> device

        lstm = Chain(LSTM(rnn_input_dim, rnn_output_dim),
                     LSTM(rnn_output_dim, rnn_output_dim)) |> device

        lstm_μ = Dense(rnn_output_dim, latent_dim) |> device
        lstm_logσ² = Dense(rnn_output_dim, latent_dim) |> device

        new(linear, rnn, rnn_μ, rnn_logσ², lstm, lstm_μ, lstm_logσ², device)
    end
end

function (encoder::Encoder)(x)
    h_rev = encoder.linear.(x)[end:-1:1] # Pass through linear layer and reverse
    rnn_out = encoder.rnn.(h_rev)[end]
    lstm_out = encoder.lstm.(h_rev)[end]
    reset!(encoder.rnn)
    reset!(encoder.lstm)
    encoder.rnn_μ(rnn_out), encoder.rnn_logσ²(rnn_out), encoder.lstm_μ(lstm_out), encoder.lstm_logσ²(lstm_out)
end

struct Decoder

    solver
    ode_func

    z₀_linear
    p_linear
    gen_linear

    device

    function Decoder(input_dim, latent_dim, hidden_dim, ode_dim, p_dim, ode_func, solver, device)

        z₀_linear = Chain(Dense(latent_dim, hidden_dim, relu),
                          Dense(hidden_dim, ode_dim)) |> device
        p_linear = Chain(Dense(latent_dim, hidden_dim, relu),
                         Dense(hidden_dim, p_dim)) |> device
        gen_linear = Chain(Dense(ode_dim, hidden_dim, relu),
                           Dense(hidden_dim, input_dim)) |> device

        new(solver, ode_func, z₀_linear, p_linear, gen_linear, device)

    end

end

function (decoder::Decoder)(latent_z₀, latent_p, t)

    z₀ = decoder.z₀_linear(latent_z₀)
    p = decoder.p_linear(latent_p)

    prob = ODEProblem(lv_func, z₀[:,1], t, p[:,1])

    output_func = (sol,i) -> (Array(sol),false)
    # function output_func(sol, i)
    #     return (Array(sol), false)
    # end
    prob_func = (prob,i,repeat) -> remake(prob, u0=z₀[:,i], p = p[:,i]) ## TODO: verify if the remake really changes the problem parameters, prints seems to say the parameters are not changed
    # function prob_func(prob, i, repeat)
    #     println(z₀[:,i])
    #     println(i)
    #     prob_new = remake(prob; u0=z₀[:,i], p=p[:,i])
    #     println(prob)
    # end

    ens_prob = EnsembleProblem(prob, prob_func = prob_func, output_func = output_func)
    pred_z = solve(ens_prob, decoder.solver, EnsembleGPUArray(), trajectories=size(p, 2), saveat=0.1f0) |> decoder.device
    pred_x = decoder.gen_linear.( Flux.unstack(pred_z, 2) )

    return pred_x, pred_z, p

end

struct Goku

    encoder::Encoder
    decoder::Decoder

    device

    function Goku(input_dim, latent_dim, rnn_input_dim, rnn_output_dim, hidden_dim, ode_dim, p_dim, ode_sys, solver, device)

        encoder = Encoder(input_dim, latent_dim, hidden_dim, rnn_input_dim, rnn_output_dim, device)
        decoder = Decoder(input_dim, latent_dim, hidden_dim, ode_dim, p_dim, ode_sys, solver, device)

        new(encoder, decoder, device)

    end
end

function (goku::Goku)(x, t) # Possible to input a "Variationnal" boolean in order to determine whether the net takes the location latent variables or it samples from the generated distribution

    latent_z₀_μ, latent_z₀_logσ², latent_p_μ, latent_p_logσ² = goku.encoder(x)

    # latent_z₀ = rand(MvNormal(latent_z₀_μ, latent_z₀_logσ²)) |> goku.device
    # latent_p = rand(MvNormal(latent_p_μ, latent_p_logσ²)) |> goku.device

    latent_z₀ = latent_z₀_μ + goku.device(randn(Float32, size(latent_z₀_logσ²))) .* exp.(latent_z₀_logσ²/2f0)
    latent_p = latent_p_μ + goku.device(randn(Float32, size(latent_p_logσ²))) .* exp.(latent_p_logσ²/2f0)

    pred_x, pred_z, pred_p = goku.decoder(latent_z₀, latent_p, t)

    return pred_x, pred_z, pred_p

end

################################################################################
## Training Utils

KL(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0
# the calculation via log(var) = log(σ²) is more numerically efficient than through log(σ)
# KL(μ, logσ) = (exp(2f0 * logσ) + μ^2)/2f0 - 0.5f0 - logσ

function rec_loss(x, pred_x)
    pred_x_stacked = Flux.stack(pred_x, 3)
    x_stacked = Flux.stack(x, 3)
    res = pred_x_stacked - x_stacked
    res_average = sum(mean((res).^2, dims = (2, 3)))
    res_diff = diff(pred_x_stacked, dims = 3) - diff(x_stacked, dims = 3)
    res_diff_average = sum(mean((res_diff).^2, dims = (2, 3)))
    return res_average + 1000*res_diff_average
end

function loss_batch(goku::Goku, λ, x, t)
    pred_x, pred_z, pred_p = goku(x, t)
    reconstruction_loss = rec_loss(x, pred_x)
    # kl_loss = mean(sum(KL.(μ, logσ²), dims = 1)) ## TODO: what to do with KL?
    return reconstruction_loss# + kl_loss
end

################################################################################
## Train

# arguments for the `train` function
@with_kw mutable struct Args
    η = 1e-3                    # learning rate
    λ = 0.05f0                  # regularization paramater
    batch_size = 256            # batch size
    seq_len = 100               # sampling size for output
    epochs = 100                 # number of epochs
    seed = 1                    # random seed
    cuda = true                 # use GPUDecoder(input_dim, latent_dim, hidden_dim, ode_dim, p_dim, ode_sys, solver, device)
    input_dim = 2
    ode_dim = 2
    p_dim = 4
    hidden_dim = 120         # hidden dimension
    rnn_input_dim = 32       # rnn input dimension
    rnn_output_dim = 32      # rnn output dimension
    latent_dim = 4            # latent dimension
    hidden_dim_node = 200   # hiddend dimension of the neuralODE
    t_max = 9.95f0              # edge of time interval for integration
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

    # Initialize ODE problem
    t_span = (0., 10.)

    # initialize Goku-net object
    goku = Goku(args.input_dim, args.latent_dim, args.rnn_input_dim, args.rnn_output_dim, args.hidden_dim, args.ode_dim, args.p_dim, lv_func, Vern7(), device)

    # ADAM optimizer
    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(goku.encoder.linear, goku.encoder.rnn, goku.encoder.rnn_μ, goku.encoder.rnn_logσ², goku.encoder.lstm, goku.encoder.lstm_μ, goku.encoder.lstm_logσ², goku.decoder.z₀_linear, goku.decoder.p_linear, goku.decoder.gen_linear)

    mkpath(args.save_path)

    # training
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader_train))

        for x in loader_train
            loss, back = Flux.pullback(ps) do
                loss_batch(goku, args.λ, x |> device, t_span)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)])

            visualize_training(goku, x, device)

        end

        model_path = joinpath(args.save_path, "model_epoch_$(epoch).bson")
        let encoder = cpu(goku.encoder),
            decoder = cpu(goku.decoder),
            args=struct2dict(args)

            BSON.@save model_path encoder decoder args
            @info "Model saved: $(model_path)"
        end
    end

    model_path = joinpath(args.save_path, "model.bson")
    let encoder = cpu(goku.encoder),
        decoder = cpu(goku.decoder),
        args=struct2dict(args)

        BSON.@save model_path encoder decoder args
        @info "Model saved: $(model_path)"
    end
end

################################################################################
## Forward passing and visualization of results

function visualize_training(goku, x, device)

    j = rand(1:size(x[1],2))
    xᵢ = [ x[i][:,j] for i in 1:size(x, 1)]

    μ, logσ², z = reconstruct(goku, xᵢ |> device, device)
    xₐ = Flux.stack(cpu(xᵢ), 2)
    zₐ = z[1]

    plt = compare_sol(xₐ, zₐ)

    png(plt, "Training_sample.png")
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

################################################################################
## Training help function

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

################################################################################
## Other

const seq_len = 100

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
