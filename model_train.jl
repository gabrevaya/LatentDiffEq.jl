

################################################################################
## Arguments for the train function
@with_kw mutable struct Args
    ## Training params
    η = 1e-2                    # learning rate
    λ = 0.01f0                  # regularization paramater
    batch_size = 256            # minibatch size
    seq_len = 30               # sampling size for output
    epochs = 200                # number of epochs for training
    seed = 1                    # random seed
    cuda = false                # GPU usage
    dt = 0.05                   # timestep for ode solve
    t_span = (0.f0, 4.95f0)     # span of time interval for training
    start_af = 0.0f0            # Annealing factor start value
    end_af = 1.f0               # Annealing factor end value
    ae = 1000                   # Annealing factor epoch end

    ## Model dimension    ## Model dimensions
    # input_dim = 8             # input dimension
    hidden_dim1 = 200
    hidden_dim2 = 200
    hidden_dim3 = 200
    rnn_input_dim = 32          # rnn input dimension
    rnn_output_dim = 16         # rnn output dimension
    latent_dim = 16             # latent dimension
    hidden_dim_latent_ode = 200 # hidden dimension

    ## Model parameters
    variational = true

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

function train(model_name, system, data_file_name, input_dim=2; kws...)
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

    # load data from bson
    @load data_file_name raw_data
    # @load data_file_name transformed_data
    # raw_data = transformed_data
    raw_data = Float32.(raw_data)
    input_dim, time_size, observations = size(raw_data)
    # raw_data = raw_data[:,:,1:1000]
    train_set, test_set = splitobs(raw_data, 0.9)
    train_set, val_set = splitobs(train_set, 0.9)
    
    # Initialize dataloaders
    loader_train = DataLoader(Array(train_set), batchsize=batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=true, partial=false)

    ############################################################################
    ## initialize model object and parameter reference
    model = Goku(input_dim, hidden_dim1, hidden_dim2, hidden_dim3,
        rnn_input_dim, rnn_output_dim, latent_dim, hidden_dim_latent_ode,
        length(system.u₀), length(system.p), system.prob, system.transform,
        Tsit5(), variational, device)

    # Get parameters
    ps = Flux.params(model)

    ############################################################################
    ## Define optimizer

    opt = ADAM(η)

    ############################################################################
    ## Various definitions

    mkpath(save_path)

    loss_mem = zeros(Float32, epochs)  # TODO : implement loss memorization

    best_val_loss::Float32 = Inf32
    val_loss::Float32 = 0

    ############################################################################
    ## Main train loop
    @info "Start Training of $(model_name)-net, total $(epochs) epochs"
    for epoch = 1:epochs

        # Model evaluation length
        t = range(t_span[1], step=dt, length=seq_len)

        mb_id = 1   # Minibatch id
        @info "Epoch $(epoch) .. (Sequence training length $(seq_len))"
        progress = Progress(length(loader_train))
        for x in loader_train

            # Comput annealing factor
            af = annealing_factor(start_af, end_af, ae, epoch, mb_id, length(loader_train))
            mb_id += 1

            # Use only a random sequence of length seq_len for all sample in the minibatch
            x = time_loader(x, time_size, seq_len)
            loss, back = Flux.pullback(ps) do
                loss_batch(model, λ, x |> device, t, af)
            end
            # Backpropagate and update
            grad = back(1f0)

            Flux.Optimise.update!(opt, ps, grad)

            # Use validation set to get loss and visualisation
            val_set = time_loader(first(loader_val), time_size, seq_len)
            val_loss = loss_batch(model, λ, val_set |> device, t, af)

            # progress meter
            # next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])
            next!(progress; showvalues=[(:val_loss, val_loss)])

        end

        loss_mem[epoch] = val_loss
        if device != gpu
            val_set = time_loader(first(loader_val), time_size, seq_len)
            visualize_training(model, val_set |> device, t)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    train("GOKU", LV(), "lv_data.bson")
end

train("GOKU", LV(), "lv_data.bson")
# train("GOKU", vdP_full(6), "vdP6_data.bson", 12)
# train("GOKU", SLV(), "SLV_data.bson", 2)
# train("GOKU", SvdP_full(1), "SvdP_data.bson", 2)