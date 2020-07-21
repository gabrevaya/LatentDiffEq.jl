

################################################################################
## Forward passing and visualization of results

function visualize_training(goku, x, t)

    j = rand(1:size(x[1],2))
    xᵢ = [ x[i][:,j] for i in 1:size(x, 1)]

    z₀_μ, z₀_logσ², p_μ, p_logσ², pred_x, pred_z₀, pred_p = goku(xᵢ, t)

    x = Flux.stack(xᵢ, 2)
    pred_x = Flux.stack(pred_x, 2)

    println("Predicted parameters : ", pred_p)

    plt = compare_sol(x, pred_x)

    png(plt, "Training_sample.png")
end

function import_model(model_path, input_dim, device)

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

function predict_from_train()

    #GPU config
    model_pth = "output/model_epoch_100.bson"
    @load model_pth args
    if args[:cuda] && has_cuda_gpu()
        device = gpu
        @info "Evaluating on GPU"
    else
        device = cpu
        @info "Evaluating on CPU"
    end

    # Load a random sample from training set
    @load "lv_data.bson" full_data

    sol = full_data[:,:,rand(1:10000)]

    # Load model
    input_dim = size(sol, 1)
    encoder, decoder = import_model(model_pth, input_dim, device)

    # Predict within time interval given
    x = Flux.unstack(reshape(sol, (size(sol, 1),size(sol, 2), 1)), 2)
    μ, logσ², z = reconstruct(encoder, decoder, x |> device, device)

    # Data dimensions manipulation
    x = dropdims(Flux.stack(x, 2), dims=3)
    z = dropdims(Flux.stack(z, 2), dims=3)

    # Showing in plot panel
    plt = compare_sol(x,z)

    png(plt, "prediction_from_train.png")
end

function compare_sol(x, z)

    plt = plot(x[1,:], color="blue", label="True x")
    plt = plot!(x[2,:], color="red", label="True y")
    plt = plot!(z[1,:], color="blue", linestyle=:dot, label="Model x")
    plt = plot!(z[2,:], color="red", linestyle=:dot, label="Model y")

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

    png(plt, "prediction_outside_train.png")

end
