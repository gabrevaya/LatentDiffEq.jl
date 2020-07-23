

function initialize_model(args, device)

    input_dim = Int(0)      # model input size
    ode_dim = Int(0)        # ode solve size
    p_dim = Int(0)          # number of parameter of system

    if args.problem == "lv"

        ## Problem defined dimension definition
        input_dim = 2
        ode_dim = 2
        p_dim = 4

        ## ODE function
        func = lv_func
    end

    if args.model_name == "GOKU"

        # Create model
        model = Goku(input_dim, args.latent_dim, args.rnn_input_dim, args.rnn_output_dim, args.hidden_dim, ode_dim, p_dim, func, Tsit5(), device)

        # Get parameters
        ps = Flux.params(model.encoder.linear, model.encoder.rnn, model.encoder.rnn_μ, model.encoder.rnn_logσ², model.encoder.lstm, model.encoder.lstm_μ, model.encoder.lstm_logσ², model.decoder.z₀_linear, model.decoder.p_linear, model.decoder.gen_linear)

    elseif args.model_name == "latent_ode"

        # Create model
        model = Latent_ODE(input_dim, args.latent_dim, args.hidden_dim, args.rnn_input_dim, args.rnn_output_dim, args.hidden_dim_node, args.seq_len, args.t_span[2], device)

        # Get parameters
        ps = Flux.params(model.encoder.linear, model.encoder.rnn, model.encoder.μ, model.encoder.logσ²,
                         model.decoder.neuralODE, model.decoder.linear)

    end

    model, ps
end
