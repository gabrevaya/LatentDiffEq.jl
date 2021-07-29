using LatentDiffEq
using FileIO
using Parameters: @with_kw
using ProgressMeter
using Random
using Statistics
using MLDataUtils
using BSON: @save
using Flux.Data: DataLoader
using Flux
using DiffEqFlux
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqSensitivity
using ModelingToolkit
using Images
using Plots
import GR

include("pendulum.jl")
include("nODE.jl")
include("create_data.jl")

################################################################################
## Arguments for the train function
@with_kw mutable struct Args
    ## Global model
    # model_type = GOKU()
    model_type = LatentODE()

    ## Latent Differential Equations
    # diffeq = Pendulum_friction()
    # diffeq = Pendulum()
    # diffeq = SPendulum()
    # diffeq = Pendulum_NN_friction()
    # diffeq = NODE(2, augment_dim = 2)
    diffeq = NODE(16)


    ## Training params
    η = 5e-4                        # learning rate
    decay = 0.0001f0                 # decay applied to weights during optimisation
    batch_size = 64                 # minibatch size
    seq_len = 50                    # sequence length for training samples
    epochs = 900                    # number of epochs for training
    seed = 1                        # random seed
    cuda = false                    # GPU usage (not working well yet)
    dt = 0.05                       # timestep for ode solve
    variational = true              # variational or deterministic training

    ## Annealing schedule
    start_β = 0.00001f0                   # start value
    end_β = 0.00001f0                     # end value
    n_cycle = 3                     # number of annealing cycles
    ratio = 0.9                     # proportion used to increase β (and 1-ratio used to fix β)
    
    ## Progressive observation training
    progressive_training = false    # progressive training usage
    prog_training_duration = 200    # number of epochs to reach the final seq_len
    start_seq_len = 5               # training sequence length at first step

    ## Visualization
    vis_len = 60                    # number of frames to visualize after each epoch
    save_figure = false             # true: save visualization figure in save_path folder
                                    # false: display image instead of saving it    
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
    data_path = "$root_dir/data/data.bson"

    if ~isfile(data_path)
        @info "Generating data"
        latent_data, u0s, ps, high_dim_data = generate_dataset(diffeq = diffeq)
        data = (latent_data, u0s, ps, high_dim_data)
        mkpath("$root_dir/data")
        @save data_path data
    end
    
    data_loaded = load(data_path, :data)
    train_data = data_loaded[4]
    latent_data = data_loaded[1]

    # stack time for each sample
    train_data = Flux.stack.(train_data, 3)

    # stack all samples
    train_data = Flux.stack(train_data, 4) # 28x28x400x450
    h, w, full_seq_len, observations = size(train_data)
    latent_data = Flux.stack(latent_data, 3)

    # vectorize frames
    train_data = reshape(train_data, :, full_seq_len, observations) # input_dim, time_size, samples
    train_data = Float32.(train_data)

    train_set, val_set = Array.(splitobs(train_data, 0.9))
    train_set_latent, val_set_latent = Array.(splitobs(latent_data, 0.9))

    # loader_train = DataLoader(Array(train_set), batchsize=batch_size, shuffle=true, partial=false)
    loader_train = DataLoader((train_set, train_set_latent), batchsize=batch_size, shuffle=true, partial=false)
    val_set_time_unstacked = Flux.unstack(val_set, 2)

    input_dim = size(train_set,1)

    ############################################################################
    # Create model

    encoder_layers, decoder_layers = default_layers(model_type, input_dim, diffeq, device = device)
    model = LatentDiffEqModel(model_type, encoder_layers, decoder_layers)

    # Get parameters
    ps = Flux.params(model)

    ############################################################################
    ## Define optimizer
    # opt = AdaBelief(η)
    # opt = ADAM(η)
    opt = ADAMW(η,(0.9,0.999), decay)

    ############################################################################
    ## Various definitions
    if progressive_training
        prog_seq_lengths = range(start_seq_len, seq_len, step=(seq_len-start_seq_len)/(prog_training_duration-1))
        prog_seq_lengths = Int.(round.(prog_seq_lengths))
    else
        prog_training_duration = 0
    end
    
    # KL annealing scheduling
    annealing_schedule = frange_cycle_linear(epochs, start_β, end_β, n_cycle, ratio)

    # Preparation for saving best models weights
    mkpath("$root_dir/output")
    best_val_loss = Inf32
    val_loss = 0f0

    # mkpath("$root_dir/output")
    # args = struct2dict(args)
    # @save "$root_dir/output/args.bson" args

    ## Visualization options
    if save_figure
        mkpath("$root_dir/output/visualization")
        GR.inline("pdf")
    end
    ############################################################################
    ## Main train loop
    @info "Start Training of $(typeof(model_type))-net, total $epochs epochs"
    for epoch = 1:epochs

        # Set annealing factor
        β = annealing_schedule[epoch]

        ## set a sequence length for training samples
        seq_len = epoch ≤ prog_training_duration ? prog_seq_lengths[epoch] : seq_len

        # Model evaluation length
        t = range(0.f0, step=dt, length=seq_len)

        @info "Epoch $epoch .. (Sequence training length $seq_len)"
        progress = Progress(length(loader_train))

        for data in loader_train
            x, latent = data

            # Use only random sequences of length seq_len for the current minibatch
            x = time_loader(x, full_seq_len, seq_len)
            
            # Run the model with the current parameters and compute the loss
            loss, back = Flux.pullback(ps) do
                loss_batch(model, x |> device, t, β, variational)
            end

            # Backpropagate and update
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # Use validation set to get loss and visualisation
            t_val = range(0.f0, step=dt, length=length(val_set_time_unstacked))
            val_loss = loss_batch(model, val_set_time_unstacked |> device, t_val, β, false)

            # progress meter
            next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])
        end

        if device != gpu
            visualize_val_image(model, val_set |> device, val_set_latent, vis_len, dt, h, w, save_figure)
        end
        
        if val_loss < best_val_loss
            best_val_loss = deepcopy(val_loss)
            weights = Flux.params(model)
            @save "$root_dir/output/best_model_weights.bson" weights
            @info "Model saved"
        end
    end
end


################################################################################
## Loss definition
# f(x) = 0.0000000001f0/(x^3)
function loss_batch(model, x, t, β, variational)

    # Make prediction
    X̂, μ, logσ² = model(x, t, variational)
    x̂, ẑ, l̂ = X̂

    # Compute reconstruction loss
    reconstruction_loss = vector_mse(x, x̂)

    # # Compute 1st order difference loss
    # x_diff1 = diff(x)
    # x̂_diff1 = diff(x̂)
    # diff1_loss = vector_mse(x_diff1, x̂_diff1)
    # # Compute 2st order difference loss
    # x_diff2 = diff(x_diff1)
    # x̂_diff2 = diff(x̂_diff1)
    # diff2_loss = vector_mse(x_diff2, x̂_diff2)

    # Compute KL losses from parameter and initial value estimation
    kl_loss = vector_kl(μ, logσ²)

    return reconstruction_loss + β * kl_loss #+ λ*diff1_loss + λ*diff2_loss
end


################################################################################
## Visualization function


function visualize_val_image(model, val_set, val_set_latent, vis_len, dt, h, w, save_figure)
    j = rand(1:size(val_set,3))
    idxs = rand_time(size(val_set,2), vis_len)
    X_test = val_set[:, idxs, j]
    true_latent = val_set_latent[:,idxs,j]

    frames_test = [Gray.(reshape(x,h,w)) for x in eachcol(X_test)]
    X_test = reshape(X_test, Val(3))
    x = Flux.unstack(X_test, 2)
    t_val = range(0.f0, step=dt, length=vis_len)

    X̂, μ, logσ² = model(x, t_val)
    x̂, ẑ, l̂ = X̂

    ẑ = Flux.stack(ẑ, 2)

    plt1 = plot(ẑ[:,:,1]', label="inferred",legend=:topleft, rightmargin = 2.0Plots.cm)
    ylabel!("State variables")
    xlabel!("time")
    plt1 = plot!(twinx(), true_latent[1,:], color=:green, box = :on, xticks=:none, label="ground truth")

    x̂ = Flux.stack(x̂, 2)
    frames_pred = [Gray.(reshape(x,h,w)) for x in eachslice(x̂, dims=2)]

    frames_test = frames_test[1:6:end]
    frames_pred = frames_pred[1:6:end]

    plt2 = mosaicview(frames_test..., frames_pred..., nrow=2, rowmajor=true)
    # plt2 = plot(plt2, leg = false, ticks = nothing, border = :none)
    # plt = plot(plt1, plt2, layout = @layout([a; b]))
    # save_figure ? savefig(plt, "output/visualization/fig.pdf") : display(plt)

    display(plt1)
    display(plt2)
end

train()
