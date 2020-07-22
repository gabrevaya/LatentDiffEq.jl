
include("model/latent_ode_model.jl")
include("utils/utils.jl")
include("utils/visualize.jl")
include("system/problem_definition.jl")

# arguments for the `train` function
@with_kw mutable struct Args

    ## Training params
    η = 1e-3                    # learning rate
    λ = 0.01f0                  # regularization paramater
    batch_size = 256            # minibatch size
    seq_len = 100               # sampling size for output
    epochs = 200                # number of epochs for training
    seed = 1                    # random seed
    cuda = false                # GPU usage
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
    input_dim = 2               # model input size
    ode_dim = 2                 # ode solve size
    p_dim = 4                   # number of parameter of system
    rnn_input_dim = 32          # rnn input dimension
    rnn_output_dim = 32         # rnn output dimension
    latent_dim = 4              # latent dimension
    hidden_dim = 120            # hidden dimension
    hidden_dim_node = 200       # hidden dimension of the neuralODE
    hidden_dim_gen = 10         # hidden dimension of the g function

    ## Save paths and keys
    full_t_span = (0.0, 19.95)  # full time span of training exemple (un-sequenced)
    dt = 0.05                   # timestep for ode solve
    u₀_range = (1.5, 3.0)       # initial value range
    p₀_range = (1.0, 2.0)       # parameter value range
    save_path = "output"        # results path
    data_file_name = "lv_data.bson"  # data file name
    raw_data_name = "raw_data"  # raw data name
    gen_data_name = "gen_data"  # generated data name

end

function train(; kws...)

    ############################################################################
    # load hyperparameters and GPU config
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

    ############################################################################
    ## Prepare training data

    # load dataset from BSON
    @load args.data_name full_data
    full_data = Float32.(full_data)
    input_dim, time_size, observations = size(full_data)
    train_set, test_set = splitobs(full_data, 0.9)
    train_set, val_set = splitobs(train_set, 0.9)

    # Initialize dataloaders
    loader_train = DataLoader(Array(train_set), batchsize=args.batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set,3), shuffle=true, partial=false)


    ############################################################################
    ## Initialize encoder and decoder
    encoder = Encoder(input_dim, args.latent_dim, args.hidden_dim, args.rnn_input_dim, args.rnn_output_dim, device)
    decoder = Decoder(input_dim, args.latent_dim, args.hidden_dim, args.hidden_dim_node, seq_len, args.t_max, device)

    # visualize a random sample from evaluation
    # and its reconstruction from the untrained model
    visualize_training(encoder, decoder, first(loader_val), device)

    ############################################################################
    ## Define training parameters and optimizer

    # ADAM optimizer
    opt = ADAM(args.η)

    # parameters

    ############################################################################
    ## Various definitions

    mkpath(args.save_path)

    seq_step = (args.full_seq_len - args.start_seq_len) / args.obs_seg_num
    loss = zeros(Float32, args.epochs) # TODO : implement loss memorization

    best_val_loss::Float32 = Inf32
    val_loss::Float32 = 0

    ############################################################################
    ## Main train loop
    @info "Start Training, total $(args.epochs) epochs"
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

        mb_id = 1 # Minibatch id
        @info "Epoch $(epoch) .. (Sequence training length $(seq_len))"
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
