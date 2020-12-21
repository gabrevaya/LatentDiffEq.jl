
################################################################################
## Forward passing and visualization of results

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



function visualize_val_image(model, val_set, t_val, h, w)
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


function visualize_val_image(model, val_set, t_val, h, w, color_scheme)
    j = rand(1:size(val_set,3))
    X_test = val_set[:,:,j]
    frames_test = [get.(Ref(color_scheme), reshape(x,h,w)) for x in eachcol(X_test)]

    x = Flux.unstack(X_test, 2)

    lat_var, pred_x, pred = model(x, t_val)
    pred_x = Flux.stack(pred_x, 2)

    frames_pred = [get.(Ref(color_scheme), reshape(x,h,w)) for x in eachcol(pred_x)]

    plt = mosaicview(frames_test..., frames_pred..., nrow=2, rowmajor=true)
    display(plt)
end