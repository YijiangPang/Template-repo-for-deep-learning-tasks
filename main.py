from Solvers.Solver_loader import solver_loader
from configs.cfg_loader import cfg_loader
import os


if __name__ == "__main__":
    (logger, model_args, data_args, training_args, proj_args), index_cfg = cfg_loader()
    
    #single-setting run 
    for i in range(proj_args.num_run):
        seed_c = proj_args.proj_seed + i
        training_args.output_dir = os.path.join(proj_args.proj_dir, "seed-%d"%(seed_c))
        solver = solver_loader(logger, model_args, data_args, training_args, proj_args, seed_c)
        solver.run()

    # #multi-settings run
    # #diff LR experiments
    # lr_dic = {"ImgCla":[5e-3, 1e-3, 5e-4, 1e-4], "LanCla":[1e-4, 5e-5, 2.5e-5, 1e-5]}
    # for lr  in lr_dic[index_cfg]:
    #     training_args.learning_rate = lr
    #     for i in range(proj_args.num_run):
    #         seed_c = proj_args.proj_seed + i
    #         training_args.output_dir = os.path.join(proj_args.proj_dir, "lr-%.1e_seed-%d"%(lr, seed_c))
    #         solver = solver_loader(logger, model_args, data_args, training_args, proj_args, seed_c)
    #         solver.run()