### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 9bc2a4a0-eddb-4229-a07b-8c7ece31b4ac
begin
	using Flux: length
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
	using DiffEqFlux
	using OrdinaryDiffEq
	using StochasticDiffEq
	using DiffEqSensitivity
	using ModelingToolkit
	using Images
	using Plots
	import GR
end

# ╔═╡ 1779a842-e98a-11eb-2ae5-b33c8ae64674
md"Welcome to the introductory tutorial for the LatentDiffEq.jl package!  In the tutorial, you will be guided through the training script as well as possible changes you can make."

# ╔═╡ a17607a0-43f4-437c-9806-aeafda77118c
begin
	path_to_local_env = "/Users/marktakken/.julia/environments/LatentDiffEqEnv/"
	path_to_package = "/Users/marktakken/.julia/dev/LatentDiffEq"
end

# ╔═╡ 90a07467-db9b-4e18-bb27-6ba48f9a04e5
begin
	import Pkg
	Pkg.activate(path_to_local_env)
end

# ╔═╡ 96fad2de-57d8-45d5-8019-669021a587da
include(path_to_package * "/examples/pendulum_friction-less/create_data.jl")

# ╔═╡ a4df6882-6841-4436-abc9-0fa510f31687
md"First, activate your local environment with LatentFiffEq.jl installed."

# ╔═╡ c0fe3684-9938-4f4b-bd30-be494954e420
md"We import the necessary packages."

# ╔═╡ ec34d402-b3e0-4b91-bbdd-5628d5def6aa
md"We define the model type.  Current options are `GOKU()` and `LatentODE().`"

# ╔═╡ 208ad135-eb21-4238-8fa6-37c3a0459dcd
model_type = GOKU()

# ╔═╡ eda7ec50-ea8e-417d-b155-be96ca682754
md"And now we define the differential equation type.  For `model_type = GOKU()`, choose the ODE of your choice (`Pendulum()`,`SPendulum()`,`Pendulum_friction()`).  For `model_type = LatentODE()`, choose instead the number of latent dimensions D (e.g. 16) and set `diffeq = NODE(D)`."

# ╔═╡ 35bf672a-65a9-4db7-adc7-e6689a030e97
diffeq = Pendulum()

# ╔═╡ fda5f9cb-0622-4d26-a048-6f88f47614b2
md"We define find some hyperparameters."

# ╔═╡ 59853dc7-d167-4020-89fa-2b921ac36c97
begin
	η = 5e-4            # learning rate
	λ = 0.001f0         # regularization paramater
	batch_size = 64     # minibatch size
	seq_len = 50        # sequence length for training samples
	epochs = 900        # number of epochs for training
	seed = 1            # random seed, set to negative value for none
	dt = 0.05           # timestep for ode solve
end

# ╔═╡ 8afdeeb1-540e-403e-90ee-c1c15d27a717
md"We let the annealing factor vary periodically from `start_β` to `end_β`."

# ╔═╡ d69e8626-32c0-4d5e-943f-d8c20be0f74b
begin
	start_β = 0f0       # start value
	end_β = 1f0         # end value
	n_cycle = 3         # number of annealing cycles
	ratio = 0.9         # (number of epochs per cycle with annealing factor growing)/(number of epochs per cycle with annealing factor fixed value)
	
end

# ╔═╡ 1100944e-aadb-4fd7-8f48-56e8017115e2
md"Optionally, we can first train on samples with a smaller sequenece length and gradually work our way up to `seq_len`.  By default, this feature is disabled."

# ╔═╡ 3b9dfc6b-93b3-4830-adba-eb21464e7b2d
begin
	progressive_training = false  # progressive training usage
	prog_training_duration = 200  # number of eppchs to reach the final seq_len
	start_seq_len = 10            # training sequence length at first step
end

# ╔═╡ 1b77273a-f1c7-47c7-a3c6-299beae6adc8
md"And finally, a parameter concerning the visualization of the model's performance at the end of each epoch."

# ╔═╡ a5402647-327f-4ed4-8c20-27cc53362834
vis_len = 60    # number of test frames to visualize after each epoch

# ╔═╡ 292a9984-81bd-4cea-b57e-5a62f64a3ff3
md"Before beginning training, let's define a loss-calculating function.  `x` is the minibatch, `t` is the time range of the minibatch, and `β` is the annealing factor."

# ╔═╡ 62450f77-f14a-408f-a712-fcdfd92dee7d
function loss_batch(model, x, t, β)

    # Make prediction
    X̂, μ, logσ² = model(x, t)
    x̂, ẑ, l̂ = X̂

    # Compute reconstruction loss
    reconstruction_loss = vector_mse(x, x̂)

    # Compute KL losses from parameter and initial value estimation
    kl_loss = vector_kl(μ, logσ²)

    return reconstruction_loss + β*kl_loss
end

# ╔═╡ c6b4997b-cf75-4acb-b5b9-2cc3f25d74b4
md"And we define a function for visualizing the performance of the model on the validation set (only applicable when training on pendulum data)."

# ╔═╡ c722cff9-3dbd-4a3a-8c64-a53b65b738dd
function visualize_val_image(model, val_set, val_set_latent, val_set_params, vis_len, dt, h, w)
    j = rand(1:size(val_set,3))                 # Pick a random batch to display
    idxs = rand_time(size(val_set,2), vis_len)  # Pick random time interval
    X_test = val_set[:, idxs, j]
    true_latent = val_set_latent[:,idxs,j]
	true_params = Float32(val_set_params[1,1,j])

    frames_test = [Gray.(reshape(x,h,w)) for x in eachcol(X_test)]
    X_test = reshape(X_test, Val(3))
    x = Flux.unstack(X_test, 2)
    t_val = range(0.f0, step=dt, length=vis_len)

    X̂, μ, logσ² = model(x, t_val)
    x̂, ẑ, l̂ = X̂
    ẑ₀, θ̂ = l̂

	println("True Pendulum Length = $true_params")
    println("Inferred Pendulum Length = $θ̂")

    ẑ = Flux.stack(ẑ, 2)
	
	
    plt1 = plot(ẑ[1,:,1], legend=false, ylabel="inferred angle", color=:indigo, yforeground_color_axis=:indigo, yforeground_color_text=:indigo, yguidefontcolor=:indigo, rightmargin = 2.0Plots.cm)
	
    xlabel!("time")
	
    plt1 = plot!(twinx(), true_latent[1,:], color=:darkorange1, box = :on, xticks=:none, legend=false, ylabel="true angle", yforeground_color_axis=:darkorange1, yforeground_color_text=:darkorange1, yguidefontcolor=:darkorange1)

    x̂ = Flux.stack(x̂, 2)
    frames_pred = [Gray.(reshape(x,h,w)) for x in eachslice(x̂, dims=2)]

    frames_test = frames_test[1:6:end]
    frames_pred = frames_pred[1:6:end]

    plt2 = mosaicview(frames_test..., frames_pred..., nrow=2, rowmajor=true)
    plt2 = plot(plt2, leg = false, ticks = nothing, border = :none)
    plt = plot(plt1, plt2, layout = @layout([a; b]))
    display(plt)
end

# ╔═╡ ca49fd89-eea6-40f5-9ab2-236a2ebca5ec
md"And now we begin training!  First, we set the random seed and generate the data if needed."

# ╔═╡ 8b10850f-ae46-44b2-b0b2-789c97050959
seed > 0 && Random.seed!(seed)

# ╔═╡ c4852032-ebbd-482d-9389-2233860d6566
begin
	root_dir = path_to_package * "/examples/pendulum_friction-less"
	data_path = "$root_dir/data/data.bson"
	if ~isfile(data_path)
        @info "Generating data"
        latent_data, u0s, p, high_dim_data = generate_dataset(diffeq = diffeq)
        data = (latent_data, u0s, p, high_dim_data)
        mkpath("$root_dir/data")
        @save data_path data
    end
end

# ╔═╡ 135b9c18-5392-4d3f-857d-9086b046fd67
md"We load the data and perform necessary manipulations."

# ╔═╡ 7296a5a5-2653-4e9a-a3b5-ab5f61b4251b
function loadData()
	data_loaded = load(data_path, :data)
    train_data = data_loaded[4]
    latent_data = data_loaded[1]
	latent_params = data_loaded[3]

    # Stack time for each sample
    train_data = Flux.stack.(train_data, 3)

    # Stack all samples
    train_data = Flux.stack(train_data, 4) # 28x28x400x450
    h, w, full_seq_len, observations = size(train_data)
    latent_data = Flux.stack(latent_data, 3)
	latent_params = Flux.stack(latent_params, 3)

    # Vectorize frames
    train_data = reshape(train_data, :, full_seq_len, observations) # input_dim, time_size, samples
	
    train_data = Float32.(train_data)

    train_set, val_set = Array.(splitobs(train_data, 0.9))
    train_set_latent, val_set_latent = Array.(splitobs(latent_data, 0.9))
	train_set_params, val_set_params = Array.(splitobs(latent_params, 0.9))

    # loader_train = DataLoader(train_set, batchsize=batch_size, shuffle=true, partial=false)
	
    loader_train = DataLoader((train_set, train_set_latent), batchsize=batch_size, shuffle=true, partial=false)
	
    val_set_time_unstacked = Flux.unstack(val_set, 2)

    input_dim = size(train_set,1)
	
	return h,w,full_seq_len,train_set,val_set,train_set_latent,val_set_latent,train_set_params, val_set_params,loader_train,val_set_time_unstacked,input_dim
	
end

# ╔═╡ 9d243a51-f7dd-408f-8648-505d83225f92
begin
h,w,full_seq_len,train_set,val_set,train_set_latent,val_set_latent,train_set_params, val_set_params,loader_train,val_set_time_unstacked,input_dim = loadData()
nothing
end

# ╔═╡ fd86f340-a595-4946-b69a-d6a898c07da7
md"We create the model, access its parameters and define the optimizer."

# ╔═╡ cbb0c3bd-aaae-4c79-94e8-7f68031f350b
begin
	encoder_layers, decoder_layers = default_layers(model_type, input_dim, diffeq, cpu)
	model = LatentDiffEqModel(model_type, encoder_layers, decoder_layers)
	ps = Flux.params(model)
	opt = ADAMW(η,(0.9,0.999),λ)
end

# ╔═╡ 08926fa6-1838-412c-ac40-544232538aa7
md"If progressive training is enabled, we define the progressive sequence lengths and the progressive training duration"

# ╔═╡ 0cde3115-951d-408a-8e5d-b98d114028f6
begin
	if progressive_training
        temp = range(start_seq_len, seq_len, step=(seq_len-start_seq_len)/(prog_training_duration-1))
        prog_seq_lengths = Int.(round.(temp))
		prog_training_duration2 = prog_training_duration 
    else
        prog_training_duration2 = 0
    end
end

# ╔═╡ c5902ab2-0ba4-4b96-8632-39f6d43b0318
md"We set up the annealing factor schedule."

# ╔═╡ 12fac29d-cd5d-4fc1-8c52-734230d1d19d
annealing_schedule = frange_cycle_linear(epochs, start_β, end_β, n_cycle, ratio)

# ╔═╡ 4f4ecfaa-96f9-4512-acbb-6cff32757798
md"And we train!"

# ╔═╡ da8b7096-d705-4234-ace8-1c6adac3c8f8
begin
@info "Start Training of $(typeof(model_type))-net, total $epochs epochs"
    for epoch = 1:epochs
		
		global seq_len

        # Set annealing factor
        β = annealing_schedule[epoch]

        # Set a sequence length for training samples
        seq_len2 = epoch ≤ prog_training_duration2 ? prog_seq_lengths[epoch] : seq_len

        # Model evaluation length
        t = range(0.f0, step=dt, length=seq_len2)

        @info "Epoch $epoch .. (Sequence training length $seq_len)"
        progress = Progress(length(loader_train))

        for data in loader_train
            x, latent = data

            # Use only random sequences of length seq_len for the current minibatch
            x = time_loader(x, full_seq_len, seq_len2)
            
			# We calculate the loss using the loss-calculating function above.
            loss, back = Flux.pullback(ps) do
                loss_batch(model, x, t, β)
            end
			
            # Backpropagate and update
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)

            # Use validation set to get loss and visualisation
            t_val = range(0.f0, step=dt, length=length(val_set_time_unstacked))
            val_loss = loss_batch(model, val_set_time_unstacked, t_val, β)

            # Progress meter
            next!(progress; showvalues=[(:loss, loss),(:val_loss, val_loss)])
        end
		
            visualize_val_image(model, val_set, val_set_latent, val_set_params, vis_len, dt, h, w)
    end
end

# ╔═╡ Cell order:
# ╟─1779a842-e98a-11eb-2ae5-b33c8ae64674
# ╠═a17607a0-43f4-437c-9806-aeafda77118c
# ╟─a4df6882-6841-4436-abc9-0fa510f31687
# ╠═90a07467-db9b-4e18-bb27-6ba48f9a04e5
# ╟─c0fe3684-9938-4f4b-bd30-be494954e420
# ╠═9bc2a4a0-eddb-4229-a07b-8c7ece31b4ac
# ╠═96fad2de-57d8-45d5-8019-669021a587da
# ╟─ec34d402-b3e0-4b91-bbdd-5628d5def6aa
# ╠═208ad135-eb21-4238-8fa6-37c3a0459dcd
# ╟─eda7ec50-ea8e-417d-b155-be96ca682754
# ╠═35bf672a-65a9-4db7-adc7-e6689a030e97
# ╟─fda5f9cb-0622-4d26-a048-6f88f47614b2
# ╠═59853dc7-d167-4020-89fa-2b921ac36c97
# ╟─8afdeeb1-540e-403e-90ee-c1c15d27a717
# ╠═d69e8626-32c0-4d5e-943f-d8c20be0f74b
# ╟─1100944e-aadb-4fd7-8f48-56e8017115e2
# ╠═3b9dfc6b-93b3-4830-adba-eb21464e7b2d
# ╟─1b77273a-f1c7-47c7-a3c6-299beae6adc8
# ╠═a5402647-327f-4ed4-8c20-27cc53362834
# ╟─292a9984-81bd-4cea-b57e-5a62f64a3ff3
# ╠═62450f77-f14a-408f-a712-fcdfd92dee7d
# ╟─c6b4997b-cf75-4acb-b5b9-2cc3f25d74b4
# ╠═c722cff9-3dbd-4a3a-8c64-a53b65b738dd
# ╟─ca49fd89-eea6-40f5-9ab2-236a2ebca5ec
# ╠═8b10850f-ae46-44b2-b0b2-789c97050959
# ╠═c4852032-ebbd-482d-9389-2233860d6566
# ╟─135b9c18-5392-4d3f-857d-9086b046fd67
# ╟─7296a5a5-2653-4e9a-a3b5-ab5f61b4251b
# ╠═9d243a51-f7dd-408f-8648-505d83225f92
# ╟─fd86f340-a595-4946-b69a-d6a898c07da7
# ╠═cbb0c3bd-aaae-4c79-94e8-7f68031f350b
# ╟─08926fa6-1838-412c-ac40-544232538aa7
# ╠═0cde3115-951d-408a-8e5d-b98d114028f6
# ╟─c5902ab2-0ba4-4b96-8632-39f6d43b0318
# ╠═12fac29d-cd5d-4fc1-8c52-734230d1d19d
# ╟─4f4ecfaa-96f9-4512-acbb-6cff32757798
# ╠═da8b7096-d705-4234-ace8-1c6adac3c8f8
