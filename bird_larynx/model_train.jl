using Images, FileIO
# using Statistics
# using BSON:@save, @load
# using MLDataUtils
# using Flux
# using Flux.Data: DataLoader



# root_dir = @__DIR__
# A = readdir("$root_dir/GIF")
# filter!(e->e≠".DS_Store",A)

# # img = load("$root_dir/GIF/$(A[5])")

# data = []
# for p in A
#     push!(data, load("$root_dir/GIF/$p"))
# end

# data = [Float32.(Gray.(frame)) for frame in data]
# mean_data = mean(vcat(vec.(data)...))

# data = [d .- mean_data for d in data]
# data_vec = vec.(data)


# @load "$root_dir/../SvdP_data.bson" raw_data

# raw_data = hcat(data_vec...)
# train_set, val_set = splitobs(raw_data, 0.8)

# seq_len = 40
# train_set = [train_set[:,k:k+seq_len-1] for k in 1:(size(train_set,2)-seq_len)]
# train_set = Flux.stack(train_set, 3)

# seq_len = 5
# val_set = [val_set[:,k:k+seq_len-1] for k in 1:(size(val_set,2)-seq_len)]
# val_set = Flux.stack(val_set, 3)

# batch_size = 12
# loader_train = DataLoader(Array(train_set), batchsize=batch_size, shuffle=true, partial=false)
# loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=true, partial=false)

# model_name = "GOKU"
# system = z_switch()
# input_dim = size(raw_data,1)

# X_test = first(loader_val)[:,:,1]
# frames_test = [Gray.(reshape(x,h,w)) for x in eachcol(X_test)]
# mosaicview(frames_test..., nrow=1)

# args = Args()
# @unpack_Args args

# seed > 0 && Random.seed!(seed)

# if cuda && has_cuda_gpu()
#     device = gpu
#     @info "Training on GPU"
# else
#     device = cpu
#     @info "Training on CPU"
# end

# ############################################################################
# ## initialize model object and parameter reference
# model, ps = initialize_model(args, input_dim, model_name, system, variational, SDE, device)

# ############################################################################
# ## Define optimizer

# opt = ADAM(η)



function visualize_val_image(model, val_set, t_val, h, w)
    # @show size(val_set)
    # @show size(val_set[:,:,1])
    # val_set = Flux.unstack(val_set, 2)
    # @show size(val_set)
    # @show size(val_set[:,:,1])

    j = rand(1:size(val_set,3))
    X_test = val_set[:,:,j]
    frames_test = [Gray.(reshape(x,h,w)) for x in eachcol(X_test)]
    x = Flux.unstack(X_test, 2)

    lat_var, pred_x, pred = model(x, t_val)
    pred_x = Flux.stack(pred_x, 2)

    frames_pred = [Gray.(reshape(x,h,w)) for x in eachcol(pred_x)]

    plt = mosaicview(frames_test..., frames_pred..., nrow=2, rowmajor=true)
    display(plt)
end


################################################################################
## Arguments for the train function
@with_kw mutable struct Args
    ## Training params
    η = 1e-2                    # learning rate
    λ = 0.01f0                  # regularization paramater
    batch_size = 256            # minibatch size
    seq_len = 40               # sampling size for output
    epochs = 200                # number of epochs for training
    seed = 1                    # random seed
    cuda = false                # GPU usage
    dt = 0.05                   # timestep for ode solve
    t_span = (0.f0, 4.95f0)     # span of time interval for training
    start_af = 0.0f0            # Annealing factor start value
    end_af = 1.f0               # Annealing factor end value
    ae = 1000                   # Annealing factor epoch end

    ## Progressive observation training
    progressive_training = false # progressive training usage
    obs_seg_num = 400           # number of step to progressive training
    start_seq_len = 20          # training sequence length at first step
    full_seq_len = 400          # training sequence length at last step

    ## Model dimensions
    # input_dim = 8             # input dimension
    rnn_input_dim = 32          # rnn input dimension
    rnn_output_dim = 32         # rnn output dimension
    latent_dim = 4              # latent dimension
    hidden_dim = 120            # hidden dimension
    hidden_dim_node = 200       # hidden dimension of the neuralODE
    hidden_dim_gen = 10         # hidden dimension of the g function

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

    
    root_dir = @__DIR__
    A = readdir("$root_dir/GIF")
    filter!(e->e≠".DS_Store",A)

    # img = load("$root_dir/GIF/$(A[5])")

    data = []
    for p in A
        push!(data, load("$root_dir/GIF/$p"))
    end

    data = [Float32.(Gray.(frame)) for frame in data]
    h,w = size(data[1])

    mean_data = mean(vcat(vec.(data)...))

    data = [d .- mean_data for d in data]
    data_vec = vec.(data)

    @load "$root_dir/../SvdP_data.bson" raw_data

    raw_data = hcat(data_vec...)
    train_set, val_set = splitobs(raw_data, 0.8)

    seq_len = 40
    train_set = [train_set[:,k:k+seq_len-1] for k in 1:(size(train_set,2)-seq_len)]
    train_set = Flux.stack(train_set, 3)

    seq_len = 5
    val_set = [val_set[:,k:k+seq_len-1] for k in 1:(size(val_set,2)-seq_len)]
    val_set = Flux.stack(val_set, 3)

    batch_size = 12
    loader_train = DataLoader(Array(train_set), batchsize=batch_size, shuffle=true, partial=false)
    loader_val = DataLoader(Array(val_set), batchsize=size(val_set, 3), shuffle=true, partial=false)

    input_dim = size(raw_data,1)

    ############################################################################
    ## initialize model object and parameter reference
    model, ps = initialize_model(args, input_dim, model_name, system, variational, SDE, device)

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
        seq_len = 40

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
                # x = time_loader(x, full_seq_len, seq_len)
                x = Flux.unstack(x, 2)
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
            t_val = range(t_span[1], step=dt, length=size(val_set,2))
            visualize_val_image(model, val_set |> device, t_val, h, w)
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
    train("GOKU", vdP_full(6), "vdP6_data.bson", 12)
end
# train("GOKU", vdP_full(6), "vdP6_data.bson", 12)
# train("GOKU", SLV(), "SLV_data.bson", 2)
# train("GOKU", SvdP_full(1), "SvdP_data.bson", 2)
# train("GOKU", z_switch(), "SvdP_data.bson", 2)