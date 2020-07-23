
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
    t_eval = range(args.t_span[1], step=args.dt, length=args.full_seq_len)
    loss_mem = zeros(Float32, args.epochs)  # TODO : implement loss memorization

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

        # Model evaluation length
        t = range(args.t_span[1], step=args.dt, length=seq_len)

        af = 0.     # Annealing factor
        mb_id = 1   # Minibatch id
        @info "Epoch $(epoch) .. (Sequence training length $(seq_len))"
        progress = Progress(length(loader_train))
        for x in loader_train

            # Use only a random sequence of length seq_len for all sample in the minibatch
            x = time_loader(x, args.full_seq_len, seq_len)

            # Comput annealing factor
            af = annealing_factor(args.start_af, args.end_af, args.ae, epoch, mb_id, length(loader_train))

            loss, back = Flux.pullback(ps) do

                # Compute loss
                model.loss_batch(model, args.λ, x |> device, t, af)

            end

            # Backpropagate and update
            mb_id += 1
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # Use validation set to get loss and visualisation
            val_set = time_loader(first(loader_val), args.full_seq_len, seq_len)
            val_loss = model.loss_batch(model, args.λ, val_set |> device, t, af)
            visualize_training(model, val_set, t_eval)

            # progress meter
            next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])

        end

        loss_mem[epoch] = val_loss

        if val_loss < best_val_loss
            best_val_loss = deepcopy(val_loss)

            model_path = joinpath(args.save_path, "best_model_$(args.model_name).bson")
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
