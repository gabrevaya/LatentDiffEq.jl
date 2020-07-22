
include("model/GOKU_model.jl")
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

# TODO : use sciML train function
function train(; kws...)

    ############################################################################
    # load hyperparameters and GPU config
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
    @load args.data_file_name raw_data # gen_data
    raw_data = Float32.(raw_data)
    input_dim, time_size, observations = size(raw_data)
    train_set, test_set = splitobs(raw_data, 0.9)
    train_set, val_set = splitobs(train_set, 0.9)

    # Initialize dataloaders
    loader_train = DataLoader(Array(train_set), batchsize=args.batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=true, partial=false)

    ############################################################################
    ## initialize Goku-net object
    goku = Goku(args.input_dim, args.latent_dim, args.rnn_input_dim, args.rnn_output_dim, args.hidden_dim, args.ode_dim, args.p_dim, lv_func, Tsit5(), device)

    ############################################################################
    ## Define training parameters and optimizer

    # ADAM optimizer
    opt = ADAM(args.η)

    # parameters
    ps = Flux.params(goku.encoder.linear, goku.encoder.rnn, goku.encoder.rnn_μ, goku.encoder.rnn_logσ², goku.encoder.lstm, goku.encoder.lstm_μ, goku.encoder.lstm_logσ², goku.decoder.z₀_linear, goku.decoder.p_linear, goku.decoder.gen_linear)

    ############################################################################
    ## Various definitions

    mkpath(args.save_path)
    seq_step = (args.full_seq_len - args.start_seq_len) / args.obs_seg_num
    loss = zeros(Float32, args.epochs) # TODO : implement loss memorization

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

                # Use only a random sequence of length seq_len
                x = Flux.unstack(x[:,idxs,:], 2)

                # Comput annealing factor
                af = annealing_factor(args.start_af, args.end_af, args.ae, epoch, mb_id, length(loader_train))

                # Compute loss
                loss_batch(goku, args.λ, x |> device, t, af)

            end

            # Backpropagate and update
            mb_id += 1
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # progress meter
            next!(progress; showvalues=[(:loss, loss)])

        end


        # # Plot a random sample
        # visualize_training(goku, x, t)

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


const seq_len = 100

if abspath(PROGRAM_FILE) == @__FILE__
    train()
end










# function train(; kws...)
#
#     # load hyperparamters
#     args = Args(; kws...)
#     args.seed > 0 && Random.seed!(args.seed)
#
#     # GPU config
#     if args.cuda && has_cuda_gpu()
#         device = gpu
#         @info "Training on GPU"
#     else
#         device = cpu
#         @info "Training on CPU"
#     end
#
#     # load data from bson
#     @load args.data_file_name raw_data # gen_data
#     raw_data = Float32.(raw_data)
#     input_dim, time_size, observations = size(raw_data)
#     train_set, test_set = splitobs(raw_data, 0.9)
#     train_set, val_set = splitobs(train_set, 0.9)
#
#     # Initialize dataloaders
#     loader_train = DataLoader(Array(train_set), batchsize=args.batch_size, shuffle=true, partial=false)
#     loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=true, partial=false)
#
#     # Define saving time steps
#     t = range(args.t_span[1], args.t_span[2], length=args.seq_len)
#
#     # initialize Goku-net object
#     goku = Goku(args.input_dim, args.latent_dim, args.rnn_input_dim, args.rnn_output_dim, args.hidden_dim, args.ode_dim, args.p_dim, lv_func, Tsit5(), device)
#
#     # ADAM optimizer
#     opt = ADAM(args.η)
#
#     # parameters
#     ps = Flux.params(goku.encoder.linear, goku.encoder.rnn, goku.encoder.rnn_μ, goku.encoder.rnn_logσ², goku.encoder.lstm, goku.encoder.lstm_μ, goku.encoder.lstm_logσ², goku.decoder.z₀_linear, goku.decoder.p_linear, goku.decoder.gen_linear)
#
#     mkpath(args.save_path)
#
#     # training
#     @info "Start Training, total $(args.epochs) epochs"
#     for epoch = 1:args.epochs
#         @info "Epoch $(epoch)"
#         progress = Progress(length(loader_train))
#         mb_id = 1 # TODO : implement a cleaner way of knowing minibatch ID, there must be some way to do it from the dataloader, but didn't find anything
#         for x in loader_train
#             loss, back = Flux.pullback(ps) do
#                 af = annealing_factor(args.start_af, args.end_af, args.ae, epoch, mb_id, length(loader_train))
#                 loss_batch(goku, args.λ, x |> device, t, af)
#             end
#             grad = back(1f0)
#             Flux.Optimise.update!(opt, ps, grad)
#
#             # progress meter
#             next!(progress; showvalues=[(:loss, loss)])
#
#             visualize_training(goku, x, t)
#             mb_id += 1
#
#         end
#
#         model_path = joinpath(args.save_path, "model_epoch_$(epoch).bson")
#         let encoder = cpu(goku.encoder),
#             decoder = cpu(goku.decoder),
#             args=struct2dict(args)
#
#             BSON.@save model_path encoder decoder args
#             @info "Model saved: $(model_path)"
#         end
#     end
#
#     model_path = joinpath(args.save_path, "model.bson")
#     let encoder = cpu(goku.encoder),
#         decoder = cpu(goku.decoder),
#         args=struct2dict(args)
#
#         BSON.@save model_path encoder decoder args
#         @info "Model saved: $(model_path)"
#     end
# end
