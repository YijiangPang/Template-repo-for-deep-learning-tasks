def solver_loader(logger, model_args, data_args, training_args, proj_args, seed_c):
    if proj_args.solver == "solver_LanCla":
        from Solvers.solver_LanCla import solver_LanCla
        s =  solver_LanCla(logger, model_args, data_args, training_args, proj_args, seed_c)
    elif proj_args.solver == "solver_ImgCla":
        from Solvers.solver_ImgCla import solver_ImgCla
        s =  solver_ImgCla(logger, model_args, data_args, training_args, proj_args, seed_c)
    elif proj_args.solver == "solver_CL":
        from Solvers.solver_CL import solver_CL
        s =  solver_CL(logger, model_args, data_args, training_args, proj_args, seed_c)
    return s

