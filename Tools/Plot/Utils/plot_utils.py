from matplotlib import pyplot as plt 
import numpy as np
import os
import json
import copy


class LogProcess():
    def __init__(self):
        pass

    def getPerf(self, file_path_list, filter_settings):
        try:
            _, file_suffix = os.path.splitext(file_path_list[0])
        except:
            _, file_suffix = os.path.splitext(file_path_list[0][0][0])
        if "log" in file_suffix:
            return self.getPerf_log(file_path_list)
        elif "json" in file_suffix:
            return self.getPerf_json(file_path_list, filter_settings)
    
    @staticmethod
    def getPerf_json(file_path_list, filter_settings):
        record_dic = {}
        # task_criterion = {"cola": ["eval_matthews_correlation"], "sst2": ["eval_accuracy"], \
        #           "mrpc": ["eval_f1", "eval_accuracy"], "stsb": ["eval_pearson", "eval_spearmanr"], \
        #           "qqp": ["eval_f1", "eval_accuracy"], "mnli": ["eval_accuracy", "eval_accuracy_mm"], \
        #           "qnli": ["eval_accuracy"], "rte": ["eval_accuracy"], \
        #           "wnli": ["eval_accuracy"]}
        task_test_criterion = {"cola": "eval_matthews_correlation", "sst2": "eval_accuracy", \
                  "mrpc": "eval_f1", "stsb": "eval_pearson", "qqp": "eval_f1", "mnli": "eval_accuracy", \
                  "qnli": "eval_accuracy", "rte": "eval_accuracy", "wnli": "eval_accuracy", \
                    "densenet121": "test_accuracy", "resnet18": "test_accuracy", "vgg11": "test_accuracy",
                    "vit_B_16": "test_accuracy"}
        task_eval_criterion = {"cola": "eval_matthews_correlation", "sst2": "eval_accuracy", \
                  "mrpc": "eval_f1", "stsb": "eval_pearson", "qqp": "eval_f1", "mnli": "eval_accuracy", \
                  "qnli": "eval_accuracy", "rte": "eval_accuracy", "wnli": "eval_accuracy", \
                    "densenet121": "eval_accuracy", "resnet18": "eval_accuracy", "vgg11": "eval_accuracy",
                    "vit_B_16": "eval_accuracy"}
        for file_setting, file_perf, file_final in zip(file_path_list[1][0], file_path_list[1][1], file_path_list[1][2]):
            flag_folder = file_setting.split("/")[2]
            file_training = [i for i in file_path_list[0][0] if flag_folder in i]
            assert len(file_training) == 1
            file_training = file_training[0]
            with open(file_training, 'r') as openfile_t, open(file_setting, 'r') as openfile_s, open(file_perf, 'r') as openfile_p, open(file_final, 'r') as openfile_f:
                json_t, json_s, json_p, json_f = json.load(openfile_t), json.load(openfile_s), json.load(openfile_p), json.load(openfile_f)
                optmizer_name = json_s["opt_name"]
                if optmizer_name not in filter_settings["optmizer"]: continue
                try:
                    #lan
                    backbone = json_s["model_name_or_path"]
                    task_name = json_s["task_name"]
                except:
                    #img
                    backbone = json_s["dataset_name"]
                    task_name = json_s["model_name_or_path"]
                
                if "weight_decay" in json_s:
                    gamma = json_s["weight_decay"]
                    if gamma is not None:
                        optmizer_name = optmizer_name + "_%.1e"%gamma
                lr = json_s["learning_rate"]
                seed = json_s["seed_c"]
                performance_criterion = json_f[task_test_criterion[task_name]]

                loss_t, loss_e, acc_eval = [], [], []
                loss_t_avg = None
                for r in json_p["log_history"]:
                    if "eval_loss" in r:
                        loss_e.append(r["eval_loss"])
                        acc_eval.append(r[task_eval_criterion[task_name]])
                    elif "train_loss" in r:
                        loss_t_avg = r["train_loss"]
                    else:
                        loss_t.append(r["loss"])
                    step = r["step"]
                if backbone not in record_dic: record_dic[backbone] = {}
                if task_name not in record_dic[backbone]: record_dic[backbone][task_name] = {}
                if optmizer_name not in record_dic[backbone][task_name]: record_dic[backbone][task_name][optmizer_name] = {}
                if lr not in record_dic[backbone][task_name][optmizer_name]: record_dic[backbone][task_name][optmizer_name][lr] = {}
                record_dic[backbone][task_name][optmizer_name][lr]["loss_t"] = {seed:loss_t} | record_dic[backbone][task_name][optmizer_name][lr]["loss_t"] if "loss_t" in record_dic[backbone][task_name][optmizer_name][lr] else {seed:loss_t}
                record_dic[backbone][task_name][optmizer_name][lr]["loss_e"] = {seed:loss_e} | record_dic[backbone][task_name][optmizer_name][lr]["loss_e"] if "loss_e" in record_dic[backbone][task_name][optmizer_name][lr] else {seed:loss_e}
                record_dic[backbone][task_name][optmizer_name][lr]["acc_eval"] = {seed:acc_eval} | record_dic[backbone][task_name][optmizer_name][lr]["acc_eval"] if "acc_eval" in record_dic[backbone][task_name][optmizer_name][lr] else {seed:acc_eval}
                record_dic[backbone][task_name][optmizer_name][lr]["performance_criterion"] = {seed:performance_criterion} | record_dic[backbone][task_name][optmizer_name][lr]["performance_criterion"] if "performance_criterion" in record_dic[backbone][task_name][optmizer_name][lr] else {seed:performance_criterion}#"%.3f"%performance_criterion
        # record_dic_temp = copy.deepcopy(record_dic)
        # for backbone in record_dic:
        #     for task_name in record_dic[backbone]:
        #         for optmizer_name in record_dic[backbone][task_name]:
        #             for lr in record_dic[backbone][task_name][optmizer_name]:
        #                 perf_seed = record_dic[backbone][task_name][optmizer_name][lr]
        #                 dic_perf = {k:[] for k in list((list(perf_seed.values())[0]).keys())}
        #                 for p in perf_seed:
        #                     for n in dic_perf:
        #                         dic_perf[n].append(perf_seed[p][n])
        #                 record_dic_temp[backbone][task_name][optmizer_name][lr] = dic_perf
        return record_dic
    @staticmethod
    def getPerf_log(file_path_list):
        record_dic = {}
        for log_file in file_path_list:
            with open(log_file, mode = 'r') as f:
                lines_log = f.readlines()
                for l in lines_log:
                    if "----Setting----" in l:
                        data_name, epochs, m_pretrained, backbone, optmizer_name, acc_avg, lr = None, None, None, None, None, None, None
                        seed, iteration = None, None
                        # flag_setting_end = False
                    elif "data_name - " in l:
                        data_name = l.split("data_name - ")[-1][:-1]
                    elif "training - epochs - " in l:
                        epochs = l.split("training - epochs - ")[-1][:-1]
                    elif "training - m_pretrained - " in l:
                        m_pretrained = l.split("training - m_pretrained - ")[-1][:-1]
                    elif "backbone - " in l:
                        backbone = l.split("backbone - ")[-1][:-1]
                    elif "optmizer_name - " in l:
                        optmizer_name = l.split("optmizer_name - ")[-1][:-1]
                    elif "training - lr_init - " in l:
                        lr = float(l.split("training - lr_init - ")[-1][:-1])
                    elif "seed - " in l:
                        seed = int(l.split("seed - ")[-1])
                    elif "iteration - " in l:
                        iteration = int(l.split("iteration - ")[-1])
                    elif "----log----" in l:
                        seed_c = seed
                        loss_v = {seed + i:[] for i in range(iteration)}
                        acc_v = {seed + i:[] for i in range(iteration)}
                    elif ",seed\epoch=" in l:
                        l_v = (l.split("loss=")[-1]).split(",")[0]
                        l_v = float(l_v) if "nan" not in l_v else "nan"
                        loss_v[seed_c].append(l_v)
                        a_v = (l.split("acc_clean=")[-1]).split(",")[0]
                        a_v = float(a_v)
                        acc_v[seed_c].append(a_v)
                    elif "load the full trained model to test" in l:
                        seed_c = seed_c + 1
                    elif "acc_avg = " in l:
                        acc_avg = l.split("acc_avg = ")[-1][:-1]
                        if data_name not in record_dic: record_dic[data_name] = {}
                        if backbone not in record_dic[data_name]: record_dic[data_name][backbone] = {}
                        if optmizer_name not in record_dic[data_name][backbone]: record_dic[data_name][backbone][optmizer_name] = {}
                        if lr not in record_dic[data_name][backbone][optmizer_name]: record_dic[data_name][backbone][optmizer_name][lr] = {}
                        record_dic[data_name][backbone][optmizer_name][lr]["loss"] = loss_v
                        record_dic[data_name][backbone][optmizer_name][lr]["acc"] = acc_v
                        record_dic[data_name][backbone][optmizer_name][lr]["acc_avg"] = acc_avg
        return record_dic
    
    def getFilesInPath(self, folder, time_list, suffix, target_file_name = None):
        if target_file_name is None:
            return self.getFilesInPath_log(folder, time_list, suffix)
        else:
            return self.getFilesInPath_json(folder, time_list, suffix, target_file_name)

    @staticmethod
    def getFilesInPath_log(folder, time_list, suffix):
        name_list = []
        f_list = sorted(os.listdir(folder))
        folder_subs = [f_n for f_n in f_list]
        for folder_sub in folder_subs:
            folder_sub = os.path.join(folder, folder_sub)
            if os.path.isdir(folder_sub):
                f_sub_list = sorted(os.listdir(folder_sub))
                for f_n in f_sub_list:
                    if suffix in os.path.splitext(f_n)[1]:
                        if time_list is None:
                            pathName = os.path.join(folder_sub, f_n)
                            name_list.append(pathName)
                        else: 
                            if sum([1 if f in folder_sub else 0 for f in time_list]) >=1:
                                pathName = os.path.join(folder_sub, f_n)
                                name_list.append(pathName)
        return name_list
    
    @staticmethod
    def getFilesInPath_json(folder, time_list, suffix, target_name_list):
        target_name_list_sub, target_name_list_sub2 = target_name_list
        file_path_list_sub = [[] for i in range(len(target_name_list_sub))]
        file_path_list_sub2 = [[] for i in range(len(target_name_list_sub2))]
        for folder_sub in sorted(os.listdir(folder)):
            if not os.path.isdir(os.path.join(folder, folder_sub)): continue
            folder_sub_list = sorted(os.listdir(os.path.join(folder, folder_sub)))
            folder_sub_list = [os.path.join(folder, folder_sub, i) for i in folder_sub_list]
            file_sub_list = [i for i in folder_sub_list if not os.path.isdir(i)]
            folder_sub_list = [i for i in folder_sub_list if os.path.isdir(i)]
            #files in sub foler
            for file_sub in file_sub_list:
                file_name, file_suffix = os.path.splitext(file_sub)
                for id, target_name in enumerate(target_name_list_sub):
                    if suffix in file_suffix and target_name in file_name:
                        if time_list is None:
                            file_path_list_sub[id].append(file_sub)
                        else: 
                            if sum([1 if f in folder_sub else 0 for f in time_list]) >=1:
                                file_path_list_sub[id].append(file_sub)
            #files in sub2 foler
            for folder_sub2 in folder_sub_list:
                file_sub2_list = sorted(os.listdir(folder_sub2))
                for file_sub2 in file_sub2_list:
                    file_name, file_suffix = os.path.splitext(file_sub2)
                    for id, target_name in enumerate(target_name_list_sub2):
                        if suffix in file_suffix and target_name in file_name:
                            if time_list is None:
                                pathName = os.path.join(folder_sub2, file_sub2)
                                file_path_list_sub2[id].append(pathName)
                            else: 
                                if sum([1 if f in folder_sub else 0 for f in time_list]) >=1:
                                    pathName = os.path.join(folder_sub2, file_sub2)
                                    file_path_list_sub2[id].append(pathName)
        path_len_list = [len(j) for j in file_path_list_sub2]
        assert all(i == path_len_list[0] for i in path_len_list)
        path_len_list = [len(j) for j in file_path_list_sub]
        assert all(i == path_len_list[0] for i in path_len_list)
        return [file_path_list_sub, file_path_list_sub2]

    @staticmethod
    def filter(record_dic, filter_settings):
        record_dic_temp = {}
        keys = list(filter_settings.keys())
        d_exist = list(record_dic.keys())
        if len(filter_settings[keys[0]]) == 0: filter_settings[keys[0]] = d_exist
        filter_dataset = list(set([i for i in filter_settings[keys[0]] if i in d_exist]) & set(d_exist))
        for d in sorted(filter_dataset):
            record_dic_temp[d] = {}
            b_exist = list(record_dic[d].keys())
            if len(filter_settings[keys[1]]) == 0: filter_settings[keys[1]] = b_exist
            filter_backbone = list(set([i for i in filter_settings[keys[1]] if i in b_exist]) & set(b_exist))
            for b in sorted(filter_backbone):
                record_dic_temp[d][b] = {}
                o_exist = list(record_dic[d][b].keys())
                # if len(filter_settings[keys[2]]) == 0: filter_settings[keys[2]] = o_exist
                filter_settings[keys[2]] = o_exist #disable opt filter
                filter_optmizer = list(set([i for i in filter_settings[keys[2]] if i in o_exist]) & set(o_exist))
                for o in sorted(filter_optmizer):
                    record_dic_temp[d][b][o] = {}
                    l_exist = list(record_dic[d][b][o].keys())
                    if len(filter_settings[keys[3]]) == 0: filter_settings[keys[3]] = l_exist
                    filter_lr = list(set([i for i in filter_settings[keys[3]] if i in l_exist]) & set(l_exist))
                    for l in sorted(filter_lr, reverse = True):
                        record_dic_temp[d][b][o][l] = record_dic[d][b][o][l]
        return record_dic_temp
    
    @staticmethod
    def plot(record_dic, folder_path, img_name):
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
                        loss_v = list(record["loss"].values())
                        acc_avg = record["acc_avg"]
                        v_mean, v_std = np.mean(loss_v, axis = 0), np.std(loss_v, axis = 0)
                        ax[b_id][l_id].plot(range(len(v_mean)), v_mean, label = "%s_%.1e_%s"%(o, l, acc_avg))
                        ax[b_id][l_id].fill_between(range(len(v_mean)), v_mean - v_std, v_mean + v_std, alpha=.2)
                        ax[b_id][l_id].grid()
                        if b_id + 1 == len(record_dic[d]): ax[b_id][l_id].set_xlabel(x_label, fontsize = "small")
                        if l_id == 0: ax[b_id][l_id].set_ylabel(y_label, fontsize = "small")
                        ax[b_id][l_id].legend(fontsize = "small")
                        ax[b_id][l_id].set_title("%s"%(b), fontsize = "small")

        img_name = os.path.join(folder_path, "%s.png"%(img_name))
        plt.show()
        plt.savefig(img_name, bbox_inches="tight")
        plt.close()


def plot_curves(curves, path, file_name):
    fig, axs = plt.subplots(1, len(curves), figsize=(8*len(curves), 8))#
    for id, c_name in enumerate(curves):
        curve = list(curves[c_name].values())
        c_v = np.array(curve)
        v_mean, v_std = np.mean(c_v, axis = 0), np.std(c_v, axis = 0)
        axs[id].plot(range(len(v_mean)), v_mean, label = "%s"%(c_name))
        axs[id].fill_between(range(len(v_mean)), v_mean - v_std, v_mean + v_std, alpha=.2)
        axs[id].grid()
        # axs[id].set_xlabel(x_label, fontsize = 15)
        axs[id].set_ylabel("%s"%(c_name), fontsize = 15)
        # ax.set_title(title)
        # ax.legend(fontsize = 15)
    # plt.show()
    # plt.savefig(os.path.join(path, "%s.pdf"%(file_name)), bbox_inches="tight")
    plt.savefig(os.path.join(path, "%s.png"%(file_name)), bbox_inches="tight")
    plt.close()
