import os
from matplotlib import pyplot as plt 
import numpy as np
from Utils.plot_utils import LogProcess
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


backbone_dic = {"densenet121":"DenseNet121", "resnet18":"ResNet18", "vit_B_16":"ViT-B/16", "vgg11": "VGG11"}

class LogProcess_C(LogProcess):
    @staticmethod
    def plot(record_dic, folder_path, plot_curve_name, plot_cvrve_final_p):
        #plot fig
        num_b_max, num_l_max = [], []
        for d in record_dic:
            num_b = []
            for b in record_dic[d]:
                num_b.append(b)
                for o in record_dic[d][b]:
                    num_l = []
                    for l in record_dic[d][b][o]:
                        num_l.append(l)
                    num_l_max = num_l_max if len(num_l_max) >= len(num_l) else num_l
            num_b_max = num_b_max if len(num_b_max) >= len(num_b) else num_b
            
        title, x_label, y_label = "", "Number of iteration", "Loss values"
        fig, ax = plt.subplots(len(num_b_max), len(num_l_max), figsize=(16, 16))#figsize=(16, 4)

        for d in record_dic:
            for b_id, b in enumerate(record_dic[d]):
                for o in record_dic[d][b]:
                    for _, l in enumerate(record_dic[d][b][o]):
                        l_id = num_l_max.index(l)
                        record = record_dic[d][b][o][l]
                        loss_v = list(record[plot_curve_name].values())
                        acc_avg = list(record[plot_cvrve_final_p].values())
                        v_mean, v_std = np.mean(loss_v, axis = 0), np.std(loss_v, axis = 0)
                        ax[b_id][l_id].plot(range(len(v_mean)), v_mean, label = "%s_%.1e"%(o, l))
                        ax[b_id][l_id].fill_between(range(len(v_mean)), v_mean - v_std, v_mean + v_std, alpha=.2)
                        ax[b_id][l_id].grid()
                        if b_id + 1 == len(record_dic[d]): ax[b_id][l_id].set_xlabel(x_label, fontsize = "small")
                        if l_id == 0: ax[b_id][l_id].set_ylabel(y_label, fontsize = "small")
                        ax[b_id][l_id].legend(fontsize = "small")
                        ax[b_id][l_id].set_title("%s"%(b), fontsize = "small")

        img_name = os.path.join(folder_path, "%s.pdf"%(plot_curve_name))
        plt.show()
        plt.savefig(img_name, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_best(record_dic, folder_path, plot_curve_name, plot_cvrve_final_p):
        #plot fig
        num_b_max, num_l_max = [], []
        for d in record_dic:
            num_b = []
            for b in record_dic[d]:
                num_b.append(b)
                for o in record_dic[d][b]:
                    num_l = []
                    for l in record_dic[d][b][o]:
                        num_l.append(l)
                    num_l_max = num_l_max if len(num_l_max) >= len(num_l) else num_l
            num_b_max = num_b_max if len(num_b_max) >= len(num_b) else num_b
        
        title, x_label, y_label = "", "Epoch", "Evaluation accuracy"
        fig, ax = plt.subplots(1, len(num_b_max), figsize=(16, 3))#figsize=(16, 4)

        ylim = [[0.84, 0.89], [0.9, 0.95], [0.82, 0.87], [0.6, 0.65]] if "random" in folder_path else [[0.87, 0.91], [0.91, 0.96], [0.86, 0.89], [0.79, 0.84]]
        for d in record_dic:
            for b_id, b in enumerate(record_dic[d]):
                axins = zoomed_inset_axes(ax[b_id], 3, loc = 'center')
                # ax[b_id].set_ylim(0, 1.0)
                for o in record_dic[d][b]:

                    v_mean_list, v_std_list, v_acc_list, l_list = [], [] ,[], []
                    v_acc_test_list = []
                    for _, l in enumerate(record_dic[d][b][o]):
                        l_id = num_l_max.index(l)
                        record = record_dic[d][b][o][l]
                        loss_v = list(record[plot_curve_name].values())
                        acc_test = record[plot_cvrve_final_p]
                        acc_eval = {i:j[-1] for i,j in record["acc_eval"].items()}
                        v_mean, v_std = np.mean(loss_v, axis = 0), np.std(loss_v, axis = 0)
                        v_mean_list.append(v_mean)
                        v_std_list.append(v_std)
                        v_acc_list.append(acc_eval)
                        l_list.append(l)
                        v_acc_test_list.append(list(acc_test.values()))
                    v_acc_list = [list(i.values()) for i in v_acc_list]
                    v_acc_mean_list = [np.mean(i) for i in v_acc_list]
                    id_best = v_acc_mean_list.index(max(v_acc_mean_list))
                    v_mean, v_std, acc_eval = v_mean_list[id_best], v_std_list[id_best], v_acc_list[id_best]
                    # ax[b_id].plot(range(len(v_mean)), v_mean, label = "%s_%.3f±%.3f"%(o, np.mean(v_acc_list[id_best]), np.std(v_acc_list[id_best])))
                    ax[b_id].plot(range(len(v_mean)), v_mean, label = "%s_%.1e_%.1f±%.1f%%"%(o, l_list[id_best], np.mean(v_acc_test_list[id_best])*100, np.std(v_acc_test_list[id_best])*100))
                    ax[b_id].fill_between(range(len(v_mean)), v_mean - v_std, v_mean + v_std, alpha=.2)
                    ax[b_id].set_xlabel(x_label)
                    # if "random" not in folder_path: ax[b_id].set_xticks(np.arange(0, len(v_mean) + 4.2*4, 4.2*4), labels = ["%d"%i for i in np.arange(0, 24, 4)])
                    if b_id == 0: ax[b_id].set_ylabel(y_label)
                    ax[b_id].legend(fontsize = "xx-small")   #
                    ax[b_id].set_title("%s"%(backbone_dic[b]))
                    ax[b_id].grid()

                    axins.plot(range(len(v_mean)), v_mean, label = "%s_%.1e"%(o, l_list[id_best]))
                    axins.fill_between(range(len(v_mean)), v_mean - v_std, v_mean + v_std, alpha=.2)
                
                if "random" in folder_path:
                    axins.set_xlim(150, 200)
                else:
                    axins.set_xlim(70, 84)
                axins.set_ylim(ylim[b_id][0], ylim[b_id][1])
                axins.xaxis.set_visible(False)
                axins.yaxis.set_visible(False)
                mark_inset(ax[b_id], axins, loc1=2, loc2=4, fc="none", ec="0.5")

        img_name = os.path.join(folder_path, "%s_best.png"%(plot_curve_name))
        plt.show()
        plt.savefig(img_name, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    time_list = None #["2024-07-03_17-42-34"]
    folder_path = os.path.join("checkpoints", "main_n")
    # folder_path = os.path.join("checkpoints", "main_cifar10_pretrained_NN_20")
    # filter_settings = {"backbone": ["bert-base-uncased"],
    #                    "task_name": ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte", "wnli"],
    #                    "optmizer": [],
    #                    "lr": [1e-4, 5e-5, 2.5e-5, 1e-5]}   
    filter_settings = {"dataset": ["cifar10"],
                       "backbone": ["densenet121", "resnet18", "vgg11", "vit_B_16"],
                       "optmizer": ["AdamW", "AdamOL"],
                       "lr": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]}   
    logP = LogProcess_C()
    file_path_list = logP.getFilesInPath(folder_path, time_list, suffix = "json", target_file_name = [["training_args", "data_args", "model_args"], ["proj_args", "trainer_state", "all_results"]])
    record_dic = logP.getPerf(file_path_list, filter_settings)
    record_dic = logP.filter(record_dic, filter_settings)
    # for n in ["loss_t", "loss_e", "acc_eval"]:
    #     logP.plot(record_dic, folder_path, plot_curve_name = n, plot_cvrve_final_p = "performance_criterion")
    logP.plot_best(record_dic, folder_path, plot_curve_name = "acc_eval", plot_cvrve_final_p = "performance_criterion")


    # time_list = None #["2024-07-03_17-42-34"]
    # folder_path = os.path.join("checkpoints", "adamOL_record")
    # dataset_list = ["CIFAR10"]
    # backbone_list = ["densenet121", "resnet18", "vit_B_16", "vgg11"]
    # optmizer_list = ["AdamW", "AdamOL", "Adam"]
    # lr_list = [5e-3, 1e-3, 5e-4, 1e-4]
    # filter_settings = {"dataset": dataset_list,
    #                    "backbone": backbone_list,
    #                    "optmizer": optmizer_list,
    #                    "lr": lr_list}   
    # logP = LogProcess_C()
    # file_path_list = logP.getFilesInPath(folder_path, time_list, suffix = "log")
    # record_dic = logP.getPerf(file_path_list)
    # record_dic = logP.filter(record_dic, filter_settings)
    # logP.plot(record_dic, folder_path, img_name = "loss_convergence", plot_curve_name = "loss", plot_cvrve_final_p = "acc_avg")
