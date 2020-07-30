

################################################################################
################################################################################
## Training through Flux internal function

function train_flux(; kws...)

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

    # load data from bson
    @load data_file_name raw_data
    raw_data = Float32.(raw_data)
    input_dim, time_size, observations = size(raw_data)
    train_set, test_set = splitobs(raw_data, 0.9)
    train_set, val_set = splitobs(train_set, 0.9)

    # Initialize dataloaders
    loader_train = DataLoader(Array(train_set), batchsize=batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=true, partial=false)

    ############################################################################
    ## initialize model object and parameter reference

    model, ps = initialize_model(args, device)

    ############################################################################
    ## Define optimizer

    opt = ADAM(η)

    ############################################################################
    ## Function definition

    cb = function ()

        val_set = time_loader(first(loader_val), full_seq_len, seq_len)
        visualize_training(model, val_set |> device, t)

    end

    loss = function ()

            # Make prediction
            lat_var, pred_x, pred = model(x, t)

            # Compute reconstruction (and differential) loss
            reconstruction_loss = rec_loss(x, pred_x)

            # Filthy one liner that does the for loop above # lit
            kl_loss = sum( [ mean(sum(KL.(lat_var[i][1], lat_var[i][1]), dims=1)) for i in 1:length(lat_var) ] )

            return reconstruction_loss + kl_loss #TODO : add annealing fator
    end

    ############################################################################
    ## Various definitions

    mkpath(save_path)

    seq_step = (full_seq_len - start_seq_len) / obs_seg_num
    loss_mem = zeros(Float32, epochs)  # TODO : implement loss memorization

    best_val_loss::Float32 = Inf32
    val_loss::Float32 = 0

    ############################################################################
    ## Main train loop
    @info "Start Training of $(model_name)-net, total $(epochs) epochs"
    for epoch = 1:epochs

        ## define seq_len according to training mode (progressive or not)
        if progressive_training
            seq_len = Int( round( start_seq_len + seq_step * floor( (epoch-1) / (epochs/obs_seg_num) ) ) )
        end

        # Model evaluation length
        t = range(t_span[1], step=dt, length=seq_len)

        af = 0.     # Annealing factor
        mb_id = 1   # Minibatch id
        @info "Epoch $(epoch) .. (Sequence training length $(seq_len))"

        ## TODO: Error is here, somehow sciml_train doesn't take Flux ::params as type for its optimized parameters
        DiffEqFlux.sciml_train(loss_batch, ps, opt, loader_train; cb = cb)

    end
end
