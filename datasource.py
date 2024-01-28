# encoding=utf-8
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
from torch.utils.data import DataLoader, Dataset ,TensorDataset
from torchvision import datasets, transforms
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import random
import torchvision
import os
import cv2
import datetime
import sys

CUDA = torch.cuda.is_available()

def logging(string):
    print(str(datetime.datetime.now())+' '+str(string))
    sys.stdout.flush()

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label

# Process SVM Points dataset
class SVMconstructor(Dataset):
    def __init__(self, transform):
        X, Y = make_blobs(n_samples=50000, centers=2, random_state=0, cluster_std=5.0)
        X = (X - X.mean()) / X.std()
        Y[np.where(Y == 0)] = -1
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)

        data = []
        for i in range(len(X)):
            data.append((X[i], Y[i]))
        self.data = data
        self.transform = transform
        self.targets = Y

    def __getitem__(self, index):
        point, label = self.data[index]
        return point, label

    def __len__(self):
        return len(self.data)

# Process chest x-ray dataset		
class xrayconstructor(Dataset):
    def __init__(self, image_address, labels, t):
        self.image_address = image_address
        self.targets = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = np.array(cv2.imread(self.image_address[index])), np.array(self.targets[index])
        sample = sample.astype(np.float32) / 255.
        target = target.astype(np.int64)
        #print('sample type: ' + str(sample.dtype))
        #print('target type: ' + str(target.dtype))
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.image_address)

# Process KWS dataset
class KWSconstructor(Dataset):
    def __init__(self, root, transform=None):
        f = open(root, 'r')
        data = []
        self.targets = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append( (info[0], int(info[1])) )
            self.targets.append(int(info[1]))
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        f, label = self.data[index]
        feature = np.loadtxt(f)
        feature = np.reshape(feature, (50, 10))
        feature = feature.astype(np.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label
 
    def __len__(self):
        return len(self.data)
		
# Process chest HAR dataset		
class HARconstructor(Dataset):
    def __init__(self, samples, labels, t):
        self.data = samples
        self.targets = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = self.data[index], self.targets[index]
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.data)
		
def xray2_load_data():
    train_data = pd.read_csv('/data/chestxray14_2class_resize/train_new_2class.txt', header=None, index_col=None)[0].str.split(' ', 1)
    train_labels = np.vstack(train_data.apply(lambda x: max(x[1].split())).values).astype(np.int8)[:32233]
    #print('train labels: ' + str(train_labels))
    train_images = train_data.apply(lambda x: '/data/chestxray14_2class_resize/images-2class/' + x[0]).values[:32233]
    val_data = pd.read_csv('/data/chestxray14_2class_resize/val_new_2class.txt', header=None, index_col=None)[0].str.split(' ', 1)
    val_labels = np.vstack(val_data.apply(lambda x: max(x[1].split())).values).astype(np.int8)[:3572]
    val_images = val_data.apply(lambda x: '/data/chestxray14_2class_resize/images-2class/' + x[0]).values[:3572]
    train_labels = train_labels.reshape(len(train_labels),)
    val_labels = val_labels.reshape(len(val_labels),)
    return train_images, train_labels, val_images, val_labels
	


# Process ag_news dataset		
class ag_newsconstructor(Dataset):
    def __init__(self, root, transform=None):
        f = open(root, 'r')
        data = []
        self.targets = []
        for line in f:
            s = line.split('\n')
            info = s[0].split(' ')
            data.append( (info[0], int(info[1])) )
            self.targets.append(int(info[1]))
        self.data = data
        self.transform = transform
        
    def __getitem__(self, index):
        f, label = self.data[index]
        feature = np.loadtxt(f)
        #feature = np.reshape(feature, (50, 10))
        feature = feature.astype(np.float32)
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label
 
    def __len__(self):
        return len(self.data)
		
def HAR_load_data():
    data = np.load('./HAR/data_har.npz')
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    return X_train, onehot_to_label(Y_train), X_test, onehot_to_label(Y_test)

def onehot_to_label(y_onehot):
    a = np.argwhere(y_onehot == 1)
    return a[:, -1]
	
class organaconstructor(Dataset):
    def __init__(self, image, labels, t):
        self.image = image
        self.targets = labels
        self.T = t

    def __getitem__(self, index):
        sample, target = np.array(self.image[index]), np.array(self.targets[index])
        sample = sample.astype(np.float32) / 255.
        target = target.astype(np.int64)
        #print('sample type: ' + str(sample.dtype))
        #print('target type: ' + str(target.dtype))
        if self.T:
            return self.T(sample), target
        else:
            return sample, target

    def __len__(self):
        return self.image.shape[0]
		
def organa_load_data():
    all_data = np.load("/data/organa/organamnist.npz")
    train_x = np.concatenate((all_data['train_images'], all_data['val_images']), axis=0)
    train_y = np.concatenate((all_data['train_labels'], all_data['val_labels']), axis=0)
    test_x = all_data['test_images']
    test_y = all_data['test_labels']
    train_y = train_y.reshape(train_y.shape[0], )
    test_y = test_y.reshape(test_y.shape[0], )
    return train_x, train_y, test_x, test_y

class ML_Dataset(): 
    # Initialization
    def __init__(self, dataset_name, world_size, rank, batch_size, labeled_ratio, d_alpha=100.0, is_independent=True):
        self.dataset_name = dataset_name
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.labeled_ratio = labeled_ratio
        self.d_alpha = d_alpha
        self.is_independent = is_independent
        self.train_data, self.test_data, self.class_num, self.mediator_length = self.get_datasets(self.dataset_name)
        self.local_size = len(self.train_data) / self.mediator_length
        self.idxs = self.get_idxs()
    
    def get_composition(self):
        tp_list = []
        for i in range(self.mediator_length):
            tp_list.append(self.d_alpha)
        tp_list[self.rank % self.mediator_length] = 10.0 # 0.1
        tp_list = tuple(tp_list)
        composition_ratio = np.random.dirichlet(tp_list)
        return (composition_ratio*self.local_size).astype(int)

    def get_idxs(self):
        local_idxs = []
        self.set_seed(0)
        if self.is_independent == True: # for large scale exp: samples on each client is independently sampled from respective class pools
            labels = np.array(self.train_data.targets)
            sorted_idxs = np.argsort(labels)
            composition = self.get_composition()
            print('local dataset composition: ' + str(composition))
            class_pool_size = len(self.train_data) / self.mediator_length
            for i in range(len(composition)):
                temp = random.sample(list(sorted_idxs[int(class_pool_size)*i : int(class_pool_size)*(i+1)]),composition[i])
                #temp = np.argsort(np.array(temp))
                for j in range(composition[i]):
                    #sample_index = sorted_idxs[int(class_pool_size*random.random()) + int(class_pool_size)*i] # randomly sampling
                
                    # sample_index = sorted_idxs[(class_pool_size/self.world_size*self.rank+j) % class_pool_size + class_pool_size*i]
                    local_idxs.append(temp[j])
            logging('local idxs: ' + str(len(local_idxs)))
        else:
            labels = np.random.rand(len(self.train_data)) if self.d_alpha >= 1 else np.array(self.train_data.targets) # alpha>1:IID; alpha<1:non-IID
            #print('labels: ' + str(labels))
            sorted_idxs = np.argsort(labels)
            #print('sorted_ids: ' + str(sorted_idxs))
            if self.rank%self.mediator_length != self.mediator_length-1:
                local_idxs = sorted_idxs[int(self.local_size*(self.rank%self.mediator_length)) : int(self.local_size*((self.rank+1)%self.mediator_length))]
            else:
                local_idxs = sorted_idxs[int(self.local_size*(self.rank%self.mediator_length)) : ]
            logging('local idxs: ' + str(len(local_idxs)))
        return local_idxs

    def get_datasets(self, dataset_name):
	    # Load dataset based on dataset name
        if dataset_name == 'Mnist':
            train_dataset = datasets.MNIST(root='/home/yisu/data/mnist/', train=True, transform=transforms.ToTensor(), download=True)
            test_dataset = datasets.MNIST(root='/home/yisu/data/mnist/', train=False, transform=transforms.ToTensor())
            class_num = 10
            mediator_length = 3
        if dataset_name == 'chestxray2':
            x_train_address, y_train, x_test_address, y_test = xray2_load_data()
            transform = transforms.ToTensor()
            train_dataset = xrayconstructor(x_train_address, y_train, transform)
            test_dataset = xrayconstructor(x_test_address, y_test, transform)
            class_num = 2
            mediator_length = 3
        if dataset_name == "organa":
            train_x, train_y, test_x, test_y = organa_load_data()
            transform = transforms.ToTensor()
            train_dataset = organaconstructor(train_x, train_y, transform)
            test_dataset = organaconstructor(test_x, test_y, transform)
            class_num = 11
            mediator_length = 3
        return train_dataset, test_dataset, class_num, mediator_length

    def set_seed(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)
        if CUDA:
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def get_dataloaders(self):
        dataset={}
		
        dataset['data'] = []
        dataset['label'] = []
        dataset['test_data']=[]
        dataset['test_label']=[]
		
        train_size = len(DatasetSplit(self.train_data, self.idxs))
        test_size = len(self.test_data)
		
        for i in range(train_size):
            sample, target = DatasetSplit(self.train_data, self.idxs)[i]
            sample = sample[0,:,:]
            sample = sample.reshape(28,28)
            dataset['data'].append(sample)
            #dataset['label'].append(target)
            '''
            if target == 0:
                dataset['label'].append(np.array([1.,0.]))
            else:
                dataset['label'].append(np.array([0.,1.]))
            '''
            temp = np.zeros(11)
            temp[target] = 1.0
            dataset['label'].append(temp)
				
        datas = torch.cat(dataset['data'],0)
        datas = datas.reshape(train_size,28*28)
        labels = torch.tensor(dataset['label']).float()
        #labels =torch.tensor(np.array(dataset['label']))
		
        for i in range(test_size):
            sample, target = self.test_data[i]
            sample = sample[0,:,:]
            sample = sample.reshape(28,28)
            dataset['test_data'].append(sample)
            #dataset['test_label'].append(target)
            '''
            if target == 0:
                dataset['label'].append(np.array([1.,0.]))
            else:
                dataset['label'].append(np.array([0.,1.]))
            '''
            temp = np.zeros(11)
            temp[target] = 1.0
            dataset['test_label'].append(temp)
			
        tmp = random.sample(list(range(len(datas))), int(train_size*self.labeled_ratio))
        dataset['labeled_data'] = []
        dataset['labeled_label'] = []
        for i in range(len(tmp)):
            dataset['labeled_data'].append(datas[tmp[i]])
            dataset['labeled_label'].append(labels[tmp[i]])
			
        dataset['labeled_data'] = torch.stack(dataset['labeled_data'],0)
        dataset['labeled_label'] = torch.stack(dataset['labeled_label'],0)
        
        #dataset['labeled_data'] = datas[:int(train_size*self.labeled_ratio)]
        #dataset['labeled_label'] = labels[:int(train_size*self.labeled_ratio)]
        dataset['unlabeled_data'] = datas
        dataset['unlabeled_label'] = labels
		
        dataset['test_data'] = torch.cat(dataset['test_data'],0)
        dataset['test_data'] = dataset['test_data'].reshape(test_size,28*28)
        #dataset['test_label']=torch.tensor(np.array(dataset['test_label']))
        dataset['test_label']=torch.tensor(dataset['test_label']).float()
		
        dataloader={}
		
        dataloader['labeled'] = DataLoader(TensorDataset(dataset['labeled_data'], dataset['labeled_label']),
                                           batch_size=self.batch_size, shuffle=True,
                                           drop_last=True)
        dataloader['unlabeled'] = DataLoader(TensorDataset(dataset['unlabeled_data'], dataset['unlabeled_label']),
                                             batch_size=self.batch_size, shuffle=True,
                                             drop_last=True)
        dataloader['test'] = DataLoader(TensorDataset(dataset['test_data'],dataset['test_label']),
                                        batch_size=self.batch_size,shuffle=True,drop_last=True)
        return dataloader['labeled'], dataloader['unlabeled'], dataloader['test']

