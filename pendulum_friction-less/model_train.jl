using Images, FileIO
using JLD2, FileIO


################################################################################
## Arguments for the train function
@with_kw mutable struct Args
    ## Training params
    η = 1e-2                    # learning rate
    λ = 0.01f0                  # regularization paramater
    batch_size = 64             # minibatch size
    seq_len = 50                # sampling size for output
    epochs = 1600               # number of epochs for training
    seed = 1                    # random seed
    cuda = false                # GPU usage
    dt = 0.05                   # timestep for ode solve
    t_span = (0.f0, 2.45f0)     # span of time interval for training
    start_af = 0.0001f0            # Annealing factor start value
    end_af = 0.0001f0           # Annealing factor end value
    ae = 200                    # Annealing factor epoch end

    ## Progressive observation training
    progressive_training = false # progressive training usage
    obs_seg_num = 400           # number of step to progressive training
    start_seq_len = 20          # training sequence length at first step
    full_seq_len = 100          # training sequence length at last step

    ## Visualization
    vis_len = 10                # number of frames to visualize after each epoch

    ## Model dimensions
    # input_dim = 8             # input dimension
    hidden_dim1 = 64
    hidden_dim2 = 32
    hidden_dim3 = 16
    rnn_input_dim = 32          # rnn input dimension
    rnn_output_dim = 32         # rnn output dimension
    latent_dim = 16             # latent dimension
    hidden_dim_latent_ode = 200 # hidden dimension

    ## Model parameters
    variational = true

    ## ILC
    ILC = false                 # train with ILC
    ILC_threshold = 0.1f0       # ILC threshold

    ## SDE
    SDE = false                  # working with SDEs instead of ODEs

    ## Save paths and keys
    save_path = "output"        # results path
    # data_file_name = "kuramoto_data.bson"  # data file name
    raw_data_name = "raw_data"  # raw data name
    transformed_data_name = "transformed_data"
    gen_data_name = "gen_data"  # generated data name

end

################################################################################
################################################################################
## Training done manualy

function train(model_name, system; kws...)
    ## Model and problem definition
    # model_name:               # Available : "latent_ode", "GOKU"
    # system:                   # Available : LV(), vdP_full(k),
                                #             vdP_identical_local(k)
                                #             WC_full(k), WC(k),
                                #             WC_identical_local(k)
                                #             (k → number of oscillators)

    ############################################################################
    ## Load hyperparameters and GPU config

    args = Args(; kws...)
    @unpack_Args args

    seed > 0 && Random.seed!(seed)

    if cuda && has_cuda_gpu()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    ############################################################################
    ## Prepare training data

    
    root_dir = @__DIR__
    data_path = "pendulum_friction-less/data/jld/processed_data.jld2"
    data_loaded = load(data_path, "processed_data")
    
    train_data = data_loaded["train"]

    train_data_norm, min_val, max_val = NormalizeToUnitSegment(train_data)
    observations, time_size, h, w = size(train_data_norm)

    train_data = reshape(train_data_norm, observations, time_size, :)
    train_data = permutedims(train_data, [3, 2, 1]) # input_dim, time_size, observations

    train_set, val_set = splitobs(train_data, 0.9)

    loader_train = DataLoader(Array(train_set), batchsize=batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=false, partial=false)

    input_dim = size(train_set,1)

    ############################################################################
    ## initialize model object and parameter reference
    # Create model
    model = Goku(input_dim, hidden_dim1, hidden_dim2, hidden_dim3,
                rnn_input_dim, rnn_output_dim, latent_dim, hidden_dim_latent_ode,
                length(system.u₀), length(system.p), system.prob, system.transform,
                Tsit5(), variational, SDE, device)

    # Get parameters
    ps = Flux.params(model)
    ############################################################################
    ## Define optimizer

    opt = ADAM(η)

    ############################################################################
    ## Various definitions

    seq_step = (full_seq_len - start_seq_len) / obs_seg_num
    loss_mem = zeros(Float32, epochs)  # TODO : implement loss memorization

    best_val_loss::Float32 = Inf32
    val_loss::Float32 = 0

    let
        mkpath(save_path)
        saving_path = joinpath(save_path, "Args.bson")
        args=struct2dict(args)
        BSON.@save saving_path args
    end
    
    ############################################################################
    ## Main train loop
    @info "Start Training of $(model_name)-net, total $(epochs) epochs"
    @info "ILC: $ILC"
    for epoch = 1:epochs
        seq_len = 50

        ## define seq_len according to training mode (progressive or not)
        if progressive_training
            seq_len = Int( round( start_seq_len + seq_step * floor( (epoch-1) / (epochs/obs_seg_num) ) ) )
        end

        # Model evaluation length
        t = range(t_span[1], step=dt, length=seq_len)

        mb_id = 1   # Minibatch id
        @info "Epoch $(epoch) .. (Sequence training length $(seq_len))"
        progress = Progress(length(loader_train))
        for x in loader_train

            # Comput annealing factor
            af = annealing_factor(start_af, end_af, ae, epoch, mb_id, length(loader_train))
            mb_id += 1

            if ILC
                # Use only a random sequence of length seq_len for all sample in the minibatch
                x = time_loader2(x, full_seq_len, seq_len)
                grad = ILC_train(x, model, λ, t, af, device, ILC_threshold, ps)
            else
                # Use only a random sequence of length seq_len for all sample in the minibatch
                x = time_loader(x, full_seq_len, seq_len)
                loss, back = Flux.pullback(ps) do
                    loss_batch(model, λ, x |> device, t, af)
                end
                # Backpropagate and update
                grad = back(1f0)
            end

            Flux.Optimise.update!(opt, ps, grad)
            # Use validation set to get loss and visualisation
            # val_set = time_loader(first(loader_val), full_seq_len, seq_len)
            val_set = Flux.unstack(first(loader_val), 2)
            t_val = range(t_span[1], step=dt, length=length(val_set))
            val_loss = loss_batch(model, λ, val_set |> device, t_val, af)

            # progress meter
            next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])
            # next!(progress; showvalues=[(:loss, loss)])
            # next!(progress; showvalues=[(:val_loss, val_loss)])

        end

        loss_mem[epoch] = val_loss
        if device != gpu
            val_set = first(loader_val)
            t_val = range(t_span[1], step=dt, length=vis_len)
            visualize_val_image(model, val_set[:,1:vis_len,:] |> device, t_val, h, w)
        end
        if val_loss < best_val_loss
            best_val_loss = deepcopy(val_loss)
            model_path = joinpath(save_path, "best_model_$(model_name).bson")

            let
                # model = cpu(model)
                BSON.@save model_path model
                @info "Model saved: $(model_path)"
            end

        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train("GOKU", pendulum())
end

# train("GOKU", pendulum())
