
################################################################################
## Julia packages

using OrdinaryDiffEq
using Base.Iterators: partition
using BSON:@save, @load
using BSON
using CUDAapi: has_cuda_gpu ## TODO: use CUDA package instead (device()s)
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
using Distributions
using ModelingToolkit
using CuArrays
CuArrays.allowscalar(false)

# Flux needs to be in v0.11.0 (currently master, which is not compatible with DiffEqFlux compatibility, that's why I didn't include it in the Project.toml)

################################################################################
## Home files and modules

include("utils/utils.jl")
include("utils/visualize.jl")
include("model/model_manager.jl")
include("system/lv_problem.jl")

################################################################################
## Arguments for the train function
@with_kw mutable struct Args

    ## Model and problem definition
    model_name = "latent_ode"
    problem = "lv"

    ## Training params
    η = 1e-3                    # learning rate
    λ = 0.01f0                  # regularization paramater
    batch_size = 256            # minibatch size
    seq_len = 100               # sampling size for output
    epochs = 200                # number of epochs for training
    seed = 1                    # random seed
    cuda = false                # GPU usage
    dt = 0.05                   # timestep for ode solve
    t_span = (0.f0, 4.95f0)     # span of time interval for training
    start_af = 0.00001          # Annealing factor start value
    end_af = 0.00001            # Annealing factor end value
    ae = 200                    # Annealing factor epoch end

    ## Progressive observation training
    progressive_training = false    # progressive training usage
    obs_seg_num = 6             # number of step to progressive training
    start_seq_len = 100         # training sequence length at first step
    full_seq_len = 400          # training sequence length at last step

    ## Model dimensions
    rnn_input_dim = 32          # rnn input dimension
    rnn_output_dim = 32         # rnn output dimension
    latent_dim = 4              # latent dimension
    hidden_dim = 120            # hidden dimension
    hidden_dim_node = 200       # hidden dimension of the neuralODE
    hidden_dim_gen = 10         # hidden dimension of the g function

    ## Save paths and keys
    save_path = "output"        # results path
    data_file_name = "lv_data.bson"  # data file name
    raw_data_name = "raw_data"  # raw data name
    gen_data_name = "gen_data"  # generated data name

end

# TODO : use sciML train function
function train(; kws...)

    ############################################################################
    ## Load hyperparameters and GPU config

    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    if args.cuda && has_cuda_gpu()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ############################################################################
    ## Prepare training data

    # load data from bson
    @load args.data_file_name raw_data
    raw_data = Float32.(raw_data)
    input_dim, time_size, observations = size(raw_data)
    train_set, test_set = splitobs(raw_data, 0.9)
    train_set, val_set = splitobs(train_set, 0.9)

    # Initialize dataloaders
    loader_train = DataLoader(Array(train_set), batchsize=args.batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=true, partial=false)

    ############################################################################
    ## initialize model object and parameter reference

    model, ps = initialize_model(args, device)

    ############################################################################
    ## Define optimizer

    # ADAM optimizer
    opt = ADAM(args.η)

    ############################################################################
    ## Various definitions

    mkpath(args.save_path)

    seq_step = (args.full_seq_len - args.start_seq_len) / args.obs_seg_num
    loss = zeros(Float32, args.epochs)  # TODO : implement loss memorization

    best_val_loss::Float32 = Inf32
    val_loss::Float32 = 0

    ############################################################################
    ## Main train loop
    @info "Start Training of $(args.model_name)-net, total $(args.epochs) epochs"
    for epoch = 1:args.epochs

        ## define seq_len according to training mode (progressive or not)
        if args.progressive_training
            seq_len = Int( args.start_seq_len + seq_step * floor( (epoch-1) / (args.epochs/args.obs_seg_num) ) )
        else
            seq_len = args.seq_len
        end

        ## Select a random sequence of length seq_len for training # TODO : find a way to redefine the const seq_len and use the overload of _getobs()
        start_time = rand(1:args.full_seq_len - seq_len)
        idxs = start_time:start_time+seq_len-1
        t = range(args.t_span[1], step=args.dt, length=seq_len)

        af = 0.     # Annealing factor
        mb_id = 1   # Minibatch id
        @info "Epoch $(epoch) .. (Sequence training length $(seq_len))"
        progress = Progress(length(loader_train))
        for x in loader_train
            loss, back = Flux.pullback(ps) do

                # Use only a random sequence of length seq_len
                x = Flux.unstack(x[:,idxs,:], 2)

                # Comput annealing factor
                af = annealing_factor(args.start_af, args.end_af, args.ae, epoch, mb_id, length(loader_train))

                # Compute loss
                model.loss_batch(model, args.λ, x |> device, t, af)

            end

            # Backpropagate and update
            mb_id += 1
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            val_loss = model.loss_batch(model, args.λ, Flux.unstack(first(loader_val)[:,idxs,:], 2) |> device, t, af)

            # progress meter
            next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])

        end

        visualize_training(model, first(loader_val), device)

        if val_loss < best_val_loss
            best_val_loss = deepcopy(val_loss)

            model_path = joinpath(args.save_path, "best_model.bson")
            let model = cpu(model),
                args=struct2dict(args)

                BSON.@save model_path model args
                @info "Model saved: $(model_path)"
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end
