using .LatentDE
using FileIO
using DrWatson: struct2dict
using Logging: with_logger
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random
using MLDataUtils
using BSON: @save
using Flux.Data: DataLoader
using Flux
using OrdinaryDiffEq
using ModelingToolkit

################################################################################
## Arguments for the train function
@with_kw mutable struct Args
    ## Global model
    LatentDE_model = Goku       # Available: Goku, LatentODE

    ## Latent model
    system = pendulum()         

    ## Training params
    η = 1e-2                        # learning rate
    λ = 0.01f0                      # regularization paramater
    batch_size = 64                 # minibatch size
    seq_len = 50                    # sampling size for output
    epochs = 200                    # number of epochs for training
    seed = 1                        # random seed
    cuda = false                    # GPU usage
    dt = 0.05                       # timestep for ode solve
    start_af = 0.00001f0            # Annealing factor start value
    end_af = 0.00001f0              # Annealing factor end value
    ae = 200                        # Annealing factor epoch end

    ## Progressive observation training
    progressive_training = false    # progressive training usage
    prog_training_duration = 20     # number of eppchs to reach the final seq_len
    start_seq_len = 10              # training sequence length at first step

    ## Visualization
    vis_len = 60                    # number of frames to visualize after each epoch

    ## Model dimensions
    hidden_dim_resnet = 200
    rnn_input_dim = 32              # rnn input dimension
    rnn_output_dim = 16             # rnn output dimension
    latent_dim = 16                 # latent dimension
    hidden_dim_latent_to_ode = 200  # hidden dimension

    ## Model parameters
    variational = true

    ## Save paths and keys
    save_path = "output"            # results path
    
end

################################################################################
################################################################################
## Training done manualy

function train(; kws...)
    ## Load hyperparameters and GPU config
    args = Args(; kws...)
    @unpack_Args args

    seed > 0 && Random.seed!(seed)

    device = cpu
    @info "Training on CPU"

    ############################################################################
    ## Prepare training data

    
    root_dir = @__DIR__
    data_path = "pendulum_friction-less/data/jld/processed_data.jld2"
    # data_path = "pendulum_friction/data/jld/processed_data.jld2"
    data_loaded = load(data_path, "processed_data")
    
    train_data = data_loaded["train"]

    train_data_norm, min_val, max_val = NormalizeToUnitSegment(train_data)
    observations, full_seq_len, h, w = size(train_data_norm)

    train_data = reshape(train_data_norm, observations, full_seq_len, :)
    train_data = permutedims(train_data, [3, 2, 1]) # input_dim, time_size, observations

    train_set, val_set = splitobs(train_data, 0.9)

    loader_train = DataLoader(Array(train_set), batchsize=batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=false, partial=false)

    input_dim = size(train_set,1)

    ############################################################################
    ## initialize model object and parameter reference
    # Create model
    model = LatentDE_model(input_dim, hidden_dim_resnet, rnn_input_dim,
                            rnn_output_dim, latent_dim, hidden_dim_latent_to_ode,
                            length(system.u₀), length(system.p), system.prob,
                            system.transform, Tsit5(), variational, device)

    # hidden_dim = hidden_dim_resnet
    # hidden_dim_node = hidden_dim_resnet
    # model = LatentODE(input_dim, 2, hidden_dim, rnn_input_dim,
    #                         rnn_output_dim, hidden_dim_node, device)

    # Get parameters
    ps = Flux.params(model)
    ############################################################################
    ## Define optimizer

    opt = ADAM(η)

    ############################################################################
    ## Various definitions

    if progressive_training
        prog_seq_lengths = range(start_seq_len, seq_len, step=(seq_len-start_seq_len)/(prog_training_duration-1))
        prog_seq_lengths = Int.(round.(prog_seq_lengths))
    else
        prog_training_duration = 0
    end
    
    loss_mem = zeros(Float32, epochs)  # TODO : implement loss memorization

    best_val_loss::Float32 = Inf32
    val_loss::Float32 = 0

    let
        mkpath(save_path)
        saving_path = joinpath(save_path, "Args.bson")
        args=struct2dict(args)
        @save saving_path args
    end
    
    ############################################################################
    ## Main train loop
    @info "Start Training of $(LatentDE_model.name)-net, total $epochs epochs"
    for epoch = 1:epochs

        ## set a sequence length for training samples
        seq_len = epoch ≤ prog_training_duration ? prog_seq_lengths[epoch] : seq_len

        # Model evaluation length
        t = range(0.f0, step=dt, length=seq_len)

        mb_id = 1   # Minibatch id
        @info "Epoch $epoch .. (Sequence training length $seq_len)"
        progress = Progress(length(loader_train))
        for x in loader_train

            # Comput annealing factor
            af = annealing_factor(start_af, end_af, ae, epoch, mb_id, length(loader_train))
            mb_id += 1

            # Use only a random sequence of length seq_len for all sample in the minibatch
            x = time_loader(x, full_seq_len, seq_len)
            loss, back = Flux.pullback(ps) do
                loss_batch(model, λ, x |> device, t, af)
            end
            # Backpropagate and update
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # Use validation set to get loss and visualisation
            # val_set = time_loader(first(loader_val), full_seq_len, seq_len)
            val_set = Flux.unstack(first(loader_val), 2)
            t_val = range(0.f0, step=dt, length=length(val_set))
            val_loss = loss_batch(model, λ, val_set |> device, t_val, af)

            # progress meter
            next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])
        end

        loss_mem[epoch] = val_loss
        if device != gpu
            val_set = first(loader_val)
            t_val = range(0.f0, step=dt, length=vis_len)
            visualize_val_image(model, val_set[:,1:vis_len,:] |> device, t_val, h, w)
        end
        if val_loss < best_val_loss
            best_val_loss = deepcopy(val_loss)
            model_path = joinpath(save_path, "best_model_$(LatentDE_model.name).bson")

            let
                # model = cpu(model)
                @save model_path model
                @info "Model saved: $model_path"
            end

        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end

# train()
