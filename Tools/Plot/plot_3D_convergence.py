import numpy as np
from matplotlib import pyplot as plt 
import os
from Utils.utils import LoadFromPickleFile
from Utils.Surface_3D_plot import Surface_3D_plot


def getFilesInPath_img(folder, flag_time_list, suffix):
    name_list = []
    f_list = sorted(os.listdir(folder))
    folder_subs = [f_n for f_n in f_list]
    for folder_sub in folder_subs:
        folder_sub = os.path.join(folder, folder_sub)
        f_sub_list = sorted(os.listdir(folder_sub))

        for f_n in f_sub_list:
            if suffix in os.path.splitext(f_n)[1] and sum([1 if f in folder_sub else 0 for f in flag_time_list]) >=1 :
                pathName = os.path.join(folder_sub, f_n)
                name_list.append(pathName)

    return name_list, folder_subs


if __name__ == "__main__":
    flag_time_list = ["2024-04-27_17-20-59", "2024-04-27_17-21-28", "2024-04-27_17-21-57", "2024-04-27_17-22-27"]
    file_path_list, folder_sub = getFilesInPath_img(folder = "checkpoints", flag_time_list = flag_time_list, suffix = "pickle")

    para_range_list, resolution_list, zs_list, traj_dic = [], [], [], {}
    for file_path in file_path_list:
        dic_surface_plot = LoadFromPickleFile(file_path)
        para_range, resolution, zs, traj = dic_surface_plot["para_range"], dic_surface_plot["resolution"], dic_surface_plot["zs"], dic_surface_plot["traj"]
        traj_dic.update(traj)
        para_range_list.append(para_range)
        resolution_list.append(resolution)
        zs_list.append(zs)

    assert all(i == para_range_list[0] for i in para_range_list)
    assert all(i == resolution_list[0] for i in resolution_list)
    assert all(sum(i) == sum(zs_list[0]) for i in zs_list)

    loss_plot = Surface_3D_plot(x_range = [-para_range, para_range], y_range = [-para_range, para_range], resolution = resolution)
    loss_plot.plot(os.getcwd(), "param_convergence" ,zs_list[0], traj_dic)