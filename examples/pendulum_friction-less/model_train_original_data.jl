# Example of GOKU-net model on the original friction-less pendulum data
# from the  GOKU-net paper (https://github.com/orilinial/GOKU)

using LatentDiffEq
using FileIO
using Parameters: @with_kw
using ProgressMeter: Progress, next!
using Random
using Statistics
using MLDataUtils
using BSON: @save
using Flux.Data: DataLoader
using Flux
using OrdinaryDiffEq
using ModelingToolkit
using DiffEqSensitivity
using Images
using Plots
import GR

include("pendulum.jl")

################################################################################
## Arguments for the train function
@with_kw mutable struct Args
    ## Global model
    model_type = GOKU()

    ## Latent Differential Equations
    diffeq = Pendulum()

    ## Training params
    η = 1e-3                        # learning rate
    decay = 0.0001f0                # decay applied to weights during optimisation
    batch_size = 64                 # minibatch size
    seq_len = 50                    # sequence length for training samples
    epochs = 900                    # number of epochs for training
    seed = 3                        # random seed
    cuda = false                    # GPU usage (not working well yet)
    dt = 0.05                       # timestep for ode solve
    variational = true              # variational or deterministic training

    ## Annealing schedule
    start_β = 0.00001f0             # start value
    end_β = 0.00001f0               # end value
    n_cycle = 3                     # number of annealing cycles
    ratio = 0.9                     # proportion used to increase β (and 1-ratio used to fix β)    

    ## Progressive observation training
    progressive_training = false    # progressive training usage
    prog_training_duration = 5      # number of epochs to reach the final seq_len
    start_seq_len = 10              # training sequence length at first step

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
    data_path = "$root_dir/data/processed_data.jld2"

    if ~isfile(data_path)
        @info "Downloading pendulum data"
        mkpath("$root_dir/data")
        download("https://ndownloader.figshare.com/files/27986997", data_path)
    end

    data_loaded = load(data_path, "processed_data")
    train_data = data_loaded["train"]

    train_data_norm, min_val, max_val = normalize_to_unit_segment(train_data)
    observations, full_seq_len, h, w = size(train_data_norm)

    train_data = reshape(train_data_norm, observations, full_seq_len, :)
    train_data = permutedims(train_data, [3, 2, 1]) # input_dim, time_size, observations
    train_data = Float32.(train_data)

    train_set, val_set = splitobs(train_data, 0.9) # 28x28x400x450

    loader_train = DataLoader(Array(train_set), batchsize=batch_size, shuffle=true, partial=false)

    val_set = permutedims(val_set, [1,3,2])
    t_val = range(0.f0, step=dt, length = size(val_set, 3))
    input_dim = size(train_set,1)

    ############################################################################
    # Create model

    encoder_layers, decoder_layers = default_layers(model_type, input_dim, diffeq, device = device)
    model = LatentDiffEqModel(model_type, encoder_layers, decoder_layers)

    # Get parameters
    ps = Flux.params(model)

    ############################################################################
    ## Define optimizer
    opt = ADAM(η)
    # opt = ADAMW(η,(0.9,0.999), 0)
    # opt = AdaBelief(η)
    # opt = ADAMW(η,(0.9,0.999), decay)

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

        ## Set a sequence length for training samples
        seq_len = epoch ≤ prog_training_duration ? prog_seq_lengths[epoch] : seq_len

        # Model evaluation length
        t = range(0.f0, step=dt, length=seq_len)

        @info "Epoch $epoch .. (Sequence training length $seq_len)"
        progress = Progress(length(loader_train))
        for x in loader_train

            # Permute dimesions for having (pixels, batch_size, time)
            x = PermutedDimsArray(x, [1,3,2])

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
            val_loss = loss_batch(model, val_set |> device,  t_val, β, false)

            # progress meter
            next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])
        end

        if device != gpu
            visualize_val_image(model, val_set |> device, vis_len, dt, h, w, save_figure)
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

function loss_batch(model, x, t, β, variational)

    # Make prediction
    X̂, μ, logσ² = model(x, t, variational)
    x̂, ẑ, l̂ = X̂

    # Compute reconstruction loss
    reconstruction_loss = sum(mean((x .- x̂).^2, dims=(2,3)))

    # Compute KL losses for parameters and initial values
    kl_loss = vector_kl(μ, logσ²)

    return reconstruction_loss + β * kl_loss
end


################################################################################
## Visualization function

function visualize_val_image(model, val_set, vis_len, dt, h, w, save_figure)
    
    # randomly pick a sample from val_set and a random time interval of length vis_len
    j = rand(1:size(val_set, 2))
    idxs = rand_time(size(val_set, 3), vis_len)
    x = val_set[:, j:j, idxs]

    # create the desired time range for the model diffeq solving
    t_val = range(0.f0, step=dt, length=vis_len)

    # run model with current parameters on the picked sample
    X̂, μ, logσ² = model(x, t_val)
    x̂, ẑ, l̂ = X̂
    ẑ₀, θ̂ = l̂
    θ̂ = θ̂[1]

    plt1 = plot(ẑ[1,1,:], legend = false)
    ylabel!("Angle")
    xlabel!("time")

    # downsample
    x = @view x[:, :, 1:6:end]
    x̂ = @view x̂[:, :, 1:6:end]    

    # build frames vectors
    to_image(x) = Gray{N0f8}.(reshape(x, h, w))
    frames_val = [to_image(xₜ) for xₜ in eachslice(x, dims = 3)]
    frames_pred = [to_image(x̂ₜ) for x̂ₜ in eachslice(x̂, dims = 3)]

    # plot a mosaic view of the frames
    plt2 = mosaicview(frames_val..., frames_pred..., nrow=2, rowmajor=true)
    plt2 = plot(plt2, leg = false, ticks = nothing, border = :none)
    plt = plot(plt1, plt2, layout = @layout([a; b]))
    save_figure ? savefig(plt, "output/visualization/fig.pdf") : display(plt)
    return nothing
end

train()
