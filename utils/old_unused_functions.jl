function rec_ini_loss(x, x̂)
    # Data prep
    x_stacked = Flux.stack(x, 3)
    res = mean((x̂[1] - x_stacked[:,:,1]).^2)
end

function time_loader2(x, full_seq_len, seq_len)

    x_ = Array{Float32, 3}(undef, (size(x,1), seq_len, size(x,3)))

    for i in 1:size(x,3)
        x_[:,:,i] = x[:,rand_time(full_seq_len, seq_len),i]
    end

    x_samples_unstacked = Flux.unstack(x_, 3)
    return x_samples_unstacked

end


# make it better for gpu
CUDA.@cufunc kl(μ, logσ²) = -logσ²/2f0 + ((exp(logσ²) + μ^2)/2f0) - 0.5f0

# the calculation via log(var) = log(σ²) is more numerically efficient than through log(σ)
kl(μ, logσ) = (exp(2f0 * logσ) + μ^2)/2f0 - 0.5f0 - logσ



function create_prob(sys_name, k, sys, u₀, tspan, p)

    func_folder = mkpath(joinpath("precomputed_systems", sys_name))
    osc_folder = mkpath(joinpath(func_folder, "oscillators_"*string(k)))
    # f_file = joinpath(osc_folder, "generated_f.jld2")
    # jac_file = joinpath(osc_folder, "generated_jac.jld2")
    # tgrad_file = joinpath(osc_folder, "generated_tgrad.jld2")
    f_file = joinpath(osc_folder, "generated_f.bson")
    jac_file = joinpath(osc_folder, "generated_jac.bson")
    tgrad_file = joinpath(osc_folder, "generated_tgrad.bson")

    generate_functions = ~(isfile(f_file) && isfile(jac_file) && isfile(tgrad_file))

    if generate_functions
        computed_f = generate_function(sys, sparse = true)[2]
        computed_jac = generate_jacobian(sys, sparse = true)[2]
        computed_tgrad = generate_tgrad(sys, sparse = true)[2]

        f = eval(computed_f)
        jac = eval(computed_jac)
        tgrad = eval(computed_tgrad)

        @save(f_file, f)
        @save(jac_file, jac)
        @save(tgrad_file, tgrad)
        
    else
        @info "Precomputed functions found for the system"
        @load(f_file, f)
        @load(jac_file, jac)
        @load(tgrad_file, tgrad)

        # print(f)
    end
    prob = ODEProblem(f, u₀, tspan, p, jac = jac, tgrad = tgrad)
    return prob
end



# normalize raw data passed as a 3D array (input_dim, time, trajectories)
function normalize_Z(data)

    data = Flux.unstack(data, 3)

    μ = 0.
    σ = 0.
    for i in 1:size(data,1)
          for j in 1:size(data[1],1)
                μ = mean(data[i][j,:])
                σ = std(data[i][j,:])
                data[i][j,:] = ( data[i][j,:] .- μ ) ./ σ
          end
    end

    data = Flux.stack(data, 3)

    return norm_data

end



## for ILC


function rec_loss(x::Array{T,2}, pred_x) where T
    pred_stacked = Flux.stack(pred_x, 2)
    # Residual loss
    res = x - pred_stacked
    res_average = mean((res).^2, dims = (1, 2))
    return res_average[1]
end

function ilc_train(x, model, λ, t, af, device, ILC_threshold, ps)
    grads = Zygote.Grads[]
    for sample in x
        loss, back = Flux.pullback(ps) do
            # Compute loss
            loss_batch(model, λ, sample |> device, t, af)
        end
        # Backpropagate
        grad = back(1f0)
        push!(grads, grad)
    end

    masking!.(ps, Ref(grads), ILC_threshold)
    return grads[1]
end

function masking!(p, grads, threshold)
    if ~(grads[1][p] == nothing)
        mean_signs = mean([sign.(el[p]) for el in grads])
        mask = mean_signs .< threshold
        mean_grads = mean([el[p] for el in grads])
        mean_grads[mask] .= 0.f0
        grads[1][p][:] = mean_grads[:]
    end
end

rec_ini_loss(x::Array{T,2}, pred) where T = mean((pred[1] - x[:,1]).^2)


function visualize_training(model::AbstractModel, x, t)

    j = rand(1:size(x[1],2))
    xᵢ = [ x[i][:,j] for i in 1:size(x, 1)]

    lat_var, pred_x, pred = model(xᵢ, t)

    x = Flux.stack(xᵢ, 2)
    pred_x = Flux.stack(pred_x, 2)

    plt = compare_sol(x, dropdims(pred_x, dims = tuple(findall(size(pred_x) .== 1)...)))
    display(plt)
    # png(plt, "figure/Training_sample.png")
end


## Loading and using trained models



function import_model_ode(model_path, input_dim, device)

    @load model_path encoder decoder args

    encoder_new = Encoder(input_dim, args[:latent_dim], args[:hidden_dim], args[:rnn_input_dim], args[:rnn_output_dim], device)
    decoder_new = Decoder(input_dim, args[:latent_dim], args[:hidden_dim], args[:hidden_dim_node], args[:seq_len], args[:t_max], device)

    Flux.loadparams!(encoder_new.linear, Flux.params(encoder.linear))
    Flux.loadparams!(encoder_new.rnn, Flux.params(encoder.rnn))
    Flux.loadparams!(encoder_new.μ, Flux.params(encoder.μ))
    Flux.loadparams!(encoder_new.logσ², Flux.params(encoder.logσ²))
    Flux.loadparams!(decoder_new.neuralODE, Flux.params(decoder.neuralODE))
    Flux.loadparams!(decoder_new.linear, Flux.params(decoder.linear))

    encoder_new, decoder_new
end

function import_model_goku(model_path, ode_func, device)

    @load model_path encoder decoder args

    goku = Goku(args[:input_dim], args[:latent_dim], args[:rnn_input_dim], args[:rnn_output_dim], args[:hidden_dim], args[:ode_dim], args[:p_dim], ode_func, Tsit5(), device)

    Flux.loadparams!(goku.encoder.linear, Flux.params(encoder.linear))
    Flux.loadparams!(goku.encoder.rnn, Flux.params(encoder.rnn))
    Flux.loadparams!(goku.encoder.rnn_μ, Flux.params(encoder.rnn_μ))
    Flux.loadparams!(goku.encoder.rnn_logσ², Flux.params(encoder.rnn_logσ²))
    Flux.loadparams!(goku.encoder.lstm, Flux.params(encoder.lstm))
    Flux.loadparams!(goku.encoder.lstm_μ, Flux.params(encoder.lstm_μ))
    Flux.loadparams!(goku.encoder.lstm_logσ², Flux.params(encoder.lstm_logσ²))
    Flux.loadparams!(goku.decoder.z₀_linear, Flux.params(decoder.z₀_linear))
    Flux.loadparams!(goku.decoder.p_linear, Flux.params(decoder.p_linear))
    Flux.loadparams!(goku.decoder.gen_linear, Flux.params(decoder.gen_linear))

    return goku

end

function predict_from_train()

    #GPU config
    model_pth = "output/model_epoch_129.bson"

    # Load a random sample from training set
    @load "lv_data.bson" raw_data

    sol = raw_data[:,:,rand(1:10000)]

    # Load model
    input_dim = size(sol, 1)
    goku = import_model_goku(model_pth, lv_func, cpu)

    # define time
    t = range(0., step = 0.05, length = 400)


    # Predict within time interval given
    x = Flux.unstack(reshape(sol, (size(sol, 1),size(sol, 2), 1)), 2)

    z₀_μ, z₀_logσ², p_μ, p_logσ², pred_x, pred_z₀, pred_p = goku(x, t)

    # Data dimensions manipulation
    x = dropdims(Flux.stack(x, 2), dims=3)
    pred_x = dropdims(Flux.stack(pred_x, 2), dims=3)

    # Showing in plot panel
    plt = compare_sol(x,pred_x)

    png(plt, "figure/prediction_from_train.png")
end

function compare_sol(x, z)

    plt = plot(x[1,:], label=string("True ",1))
    plt = plot!(z[1,:], linestyle=:dot, label=string("Model ",1))
    for i = 2:size(x,1)
        plt = plot!(x[i,:], label=string("True ",i))
        plt = plot!(z[i,:], linestyle=:dot, label=string("Model ",i))
    end
    plt

end


## Function comparing solution generated from latentODE structure with true solution within the time span
function predict_within()

    # Load model
    model_pth = "output/model_epoch_100.bson"
    @load model_pth args

    if args[:cuda] && has_cuda_gpu()
        device = gpu
        @info "Evaluating on GPU"
    else
        device = cpu
        @info "Evaluating on CPU"
    end

    @load args[:data_name] p

    u0 = rand(Uniform(1.5, 3.0), 2, 1) # [x, y]
    tspan = (0.0, 9.95)
    tstep = 0.1

    sol = Array(solve_prob(u0, p, tspan, tstep))

    input_dim = size(sol, 1)
    encoder, decoder = import_model(model_pth, input_dim, device)

    # Predict within time interval given
    x = Flux.unstack(reshape(sol, (size(sol, 1),size(sol, 2), 1)), 2)
    μ, logσ², z = reconstruct(encoder, decoder, x |> device, device)

    # Data dimensions manipulation
    x = dropdims(Flux.stack(x, 2), dims=3)
    z = dropdims(Flux.stack(z, 2), dims=3)

    # Showing in plot panel
    plt = compare_sol(x, z)

    png(plt, "figure/prediction_outside_train.png")

end
