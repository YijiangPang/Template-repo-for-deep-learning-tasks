from matplotlib import pyplot as plt 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pickle
import errno
import os


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def insert(self, key, value):
        self[key] = value


def visualize_img_tensor(images, name, index = 0):
    img = images[index].data.detach().cpu().numpy()
    img = (img*255).astype(np.int)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(np.transpose(img, (1,2,0)))
    plt.show()
    plt.savefig('%s.png'%(name))
    plt.close()


def visualize_imglist_tensor(images, f_name, title = None, num = 10):
    imgs = images[:num].data.detach().cpu().numpy()
    imgs = (imgs*255).astype(np.int)

    fig = plt.figure()
    for i, im in enumerate(imgs):
        ax = fig.add_subplot(1, len(imgs), i+1)
        if title is not None: ax.title.set_text("%s"%(title[i]))
        ax.imshow(np.transpose(im, (1,2,0)))
    plt.show()
    plt.savefig('%s.png'%(f_name))
    plt.close()


func_count_paras_all = lambda model : sum(p.numel() for p in model.parameters())
func_count_paras_tuneable = lambda model : sum(p.numel() for p in model.parameters() if p.requires_grad)
def Count_Paras_Func(model):
    num_paras_all = func_count_paras_all(model)
    num_paras_tuneable = func_count_paras_tuneable(model)
    print("number of paras / tuneable paras = %d / %d" % (num_paras_all, num_paras_tuneable))
    
    
def tsne(model, dataloader, device, n_components, name = "tsne"):
    #step 1
    image_features= []
    labels = []
    for idx, (x, y) in enumerate(dataloader):
        x = x.to(device).float()
        img_f, _ = model(x)
        img_f = img_f.data.detach().cpu().numpy()
        image_features.append(img_f)
        labels.append(y.data.detach().cpu().numpy())

    image_features = np.vstack(image_features)
    labels = np.concatenate(labels, axis = -1)

    #step 2
    standarized_data = StandardScaler().fit_transform(image_features)
    model = TSNE(n_components = n_components)
    tsne_data = model.fit_transform(standarized_data)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # fig, axs = plt.subplots(ncols= 3, figsize=(9,3))
    v_color = ["blue", "brown", "pink", "gray", "olive", "cyan", "purple", "orange", "green", "red"]

    for id, l in enumerate(np.unique(labels)):
        data_sub = tsne_data[np.where(labels == l)]
        ax.scatter(data_sub[:, 0], data_sub[:, 1], c = v_color[id], alpha=0.2)    #, alpha=0.2

    plt.savefig("%s_out.png"%(name))


def cal_grad_nrom(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def SaveToPickleFile(myValue, localFileName):
    path = os.path.dirname(localFileName)
    try:
        os.makedirs(path)
    except OSError as exc: 
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        elif path == "":
            pass
        else: raise
    try:
        f = open(localFileName,'wb') #Overwrite the original file
    except OSError:
        print('Error! Writing context to local pickle file: %s' %(localFileName))
        return

    pickle.dump(myValue, f)
    f.close()


def LoadFromPickleFile(localFile):
    try:
            f = open(localFile,'rb')
    except OSError:
            print('Error! Reading local pickle file: %s' %(localFile))
            return -1
    pickle_file = pickle.load(f)
    f.close()
    #print('Success! Reading local pickle file：%s' %(localFile))
    return pickle_file