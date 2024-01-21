import torch
import torchvision
import numpy as np
import datasource
import time
import datetime
import torch.nn.functional as F
from as_manager import *
import copy, argparse, time, sys, os, random
import gc
from itertools import repeat, cycle
from torch.autograd import Variable
from models import dgm, vae_pretrain
from inference import SVI, DeterministicWarmup, ImportanceWeightedSampler

parser = argparse.ArgumentParser()
parser.add_argument('--master_address', '-m', type=str, default='127.0.0.1')
parser.add_argument('--world_size', '-w', type=int, default=5)
parser.add_argument('--rank', '-r', type=int, default=0)
parser.add_argument('--trial_no', type=int, default=0)
parser.add_argument('--remarks', type=str, default='Remarks Missing...')

args = parser.parse_args()
MASTER_ADDRESS = args.master_address
WORLD_SIZE = args.world_size
RANK = args.rank
# master_address: the address of cloud, world_size: the size of clients with a cloud, rank: the rank of current client/cloud, trial_no, the num of trial, remarks: remark the current trial.

# Initialization

## Suggested hyper-parameter value of (LEARNING_RATE, WEIGHT_DECAY) for each model
HyperParams = { 
        'CNN': ('Mnist', 0.01, 0.01),
        'CNN1': ('organa', 0.01, 0.01),
        'LogisticRegression': ('Mnist', 0.01, 0.0001),
        'SVM': ('Points', 0.01, 0.0000),
        'FixupResNet': ('Cifar10', 0.1, 0.0001),
        'ResNet': ('Cifar10', 0.1, 0.001),
        'VGG16': ('chestxray2', 0.1, 0.0005),
        'DenseNet121': ('chestxray2', 0.1, 0.0001),
        'AlexNet': ('ImageNet', 0.1, 0.0001),
        'LSTM': ('KWS', 0.05, 0.01),
        'LSTM_NLP': ('ag_news', 0.05, 0.01),
        'SqueezeNet': ('chestxray2', 0.04, 0.0001)
        }

MODEL, D_ALPHA, IS_INDEPENDENT = 'CNN1', 0.1, True
INIT_SYNC_FREQ = 10

BATCH_SIZE = 50
LABELED_RATIO = 0.05
image_size = 28*28
n_labels = 11

DATASET, LEARNING_RATE, WEIGHT_DECAY = HyperParams[MODEL]
VAE_LEARNING_RATE = 0.001
INIT_LEARNING_RATE = VAE_LEARNING_RATE
MAX_ROUND = 3000
MAX_PRETRAIN_ROUND = 20

CHECKPOINT_ENABLED = False
#NOT_MEDIATOR = True
CUDA = torch.cuda.is_available()
if CUDA:
    torch.cuda.set_device(RANK % torch.cuda.device_count())

#Write logs in folder Logs
def logging(string):
    print(str(datetime.datetime.now())+' '+str(string))
    sys.stdout.flush()
	
def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)
	
#Test accuracy (supervised learning)
def test(test_loader, model):
    accuracy = 0
    positive_test_number = 0
    total_test_number = 0
    for step, (test_x, test_y) in enumerate(test_loader):
        if CUDA:
            test_x = test_x.cuda()
            test_y = test_y.cuda()
        with torch.no_grad():
            test_output = model(test_x)
        #print('mokaiwei')
        if MODEL != 'SVM':
            pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        else:
            test_output = test_output.data.cpu().numpy()
            pred_y = np.where(test_output>0, np.ones(test_output.shape).astype(int), -1*np.ones(test_output.shape).astype(int)).reshape(-1)
        if MODEL == 'DenseNet121':
            test_y_temp = test_y.data.cpu().numpy()
            for i in range(len(test_y_temp)):
                if test_y_temp[i] < 100:
                    if pred_y[i] == test_y_temp[i]:
                        positive_test_number += 1
                else:
                    tmp = []
                    tmp_value = test_y_temp[i]
                    while tmp_value > 0:
                        tmp.append(tmp_value%100)
                        tmp_value = tmp_value // 100
                    if pred_y[i] in test_y_temp[i]:
                        positive_test_number += 1
        else:
            positive_test_number += (pred_y == test_y.data.cpu().numpy()).astype(int).sum()
        total_test_number += float(test_y.size(0))
    accuracy = positive_test_number / total_test_number
    return accuracy

#train model, including VAE model and model
def train():
    labelled, unlabelled, validation = datasource.ML_Dataset(DATASET, WORLD_SIZE, RANK, BATCH_SIZE, LABELED_RATIO, D_ALPHA, IS_INDEPENDENT).get_dataloaders() #Load dataset
    model = dgm.AuxiliaryDeepGenerativeModel([28*28, n_labels, 100, 100, [500, 500]])
    vae_model = vae_pretrain.VAE_Mnist()
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=VAE_LEARNING_RATE)   
    beta = DeterministicWarmup(n=4*len(unlabelled)*100)
    sampler = ImportanceWeightedSampler(mc=1, iw=1)	
    elbo = SVI(model, likelihood=binary_cross_entropy, beta=beta, sampler=sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999))
    loss_func = torch.nn.CrossEntropyLoss() #loss function (supervised learning)

    logging('initial model parameters: ')
    logging('\n\n ----- start training -----')

    as_manager = AS_Manager(vae_model, model, MASTER_ADDRESS, WORLD_SIZE, RANK, INIT_SYNC_FREQ)
    
		
    iter_id = 0
    epoch_id = 0
    pretrain_epoch_id = 0
    pretrain_iter_id = 0
    best = 0.0
    alpha = 0.1 * len(unlabelled) / len(labelled)
	
    client_mu_mu = torch.zeros(30)
    client_sigma_mu = torch.zeros(30)
    client_mu_sigma = torch.zeros(30)
    client_sigma_sigma = torch.zeros(30)
	
    logging('\n\n ----- start pre-training -----')

    # Add before FL train
    while pretrain_epoch_id < MAX_PRETRAIN_ROUND:
        logging('\n\n--- start pretrain epoch '+ str(pretrain_epoch_id) + ' ---')

        for step, (b_x, b_y) in enumerate(unlabelled):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            #print('b_y: ' + str(b_y))
            b_x = b_x.view(-1, image_size)
            x_reconst, mu, log_var = vae_model(b_x) # Unsupervised learning

            # Compute reconstruction loss and kl divergence
            reconst_loss = F.mse_loss(x_reconst, b_x, size_average=False)
            kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Backprop and optimize
            loss = reconst_loss + kl_div
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()


        logging('pretrain Reconst Loss:' + str(reconst_loss.item()) + '; pretrain epoch id: ' + str(pretrain_epoch_id))
        as_manager.last_test_pretrain_epoch_id = pretrain_epoch_id

        pretrain_epoch_id += 1
        as_manager.pretrain_epoch_id = pretrain_epoch_id

    logging("local pre-train finishes.")
	
    logging("start calculate mu and sigma.")

    # Calculate mu for each client
    for step, (b_x, b_y) in enumerate(unlabelled):
        b_x = b_x.cuda()
        b_x = b_x.view(-1, image_size)
        with torch.no_grad():
            x_reconst, mu, log_var = vae_model(b_x) # Unsupervised learning
        log_sigma = 0.5 * log_var
        #print('pretrain iter id: ' + str(pretrain_iter_id))

        client_mu_mu += torch.sum(mu,dim=0).cpu() # Calculate client mu
        #print('client mu mu: ' + str(client_mu_mu))
        client_sigma_mu += torch.sum(log_sigma,dim=0).cpu()

        pretrain_iter_id += 1

    num_of_samples = pretrain_iter_id * BATCH_SIZE
    client_mu_mu = client_mu_mu / num_of_samples
    client_sigma_mu = client_sigma_mu / num_of_samples

    # Calculate sigma for each client
    for step, (b_x, b_y) in enumerate(unlabelled):
        b_x = b_x.cuda()
        b_x = b_x.view(-1, image_size)
        with torch.no_grad():
            x_reconst, mu, log_var = vae_model(b_x) # Unsupervised learning
        log_sigma = 0.5 * log_var

        for i in range(BATCH_SIZE):
            mu_mu_temp = mu[i].cpu()
            sigma_mu_temp = log_sigma[i].cpu()
            client_mu_sigma += (mu_mu_temp - client_mu_mu).pow(2)
            client_sigma_sigma += (sigma_mu_temp - client_sigma_mu).pow(2)

    client_mu_sigma = client_mu_sigma / (num_of_samples - 1) # Calculate client sigma
    client_sigma_sigma = client_sigma_sigma / (num_of_samples - 1)
    num_of_samples = torch.tensor(num_of_samples)
    total_num_samples = as_manager.total_samples(num_of_samples)

    logging("finishing calculating mu and sigma.")
	
    as_manager.sync_mu_and_sigma(client_mu_mu, client_mu_sigma, client_sigma_mu, client_sigma_sigma)
	
    logging('mediator num: ' + str(as_manager.mediators_num))
    

    torch.autograd.set_detect_anomaly(True)
    while epoch_id < MAX_ROUND:   
        model.train()	
        logging('\n\n--- start epoch '+ str(epoch_id) + ' ---')
        logging('\nrank id: ' + str(as_manager.rank))     
        #count = [0]*200
        total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
        #for step, (b_x, b_y) in enumerate(train_loader):
        for (x, y), (u, _) in zip(cycle(labelled), unlabelled):
            #mediator_id, rank_id, mediator_length = as_manager.sync_within_mediator(model, iter_id)
            
            #if NOT_MEDIATOR == True or as_manager.rank == rank_id[as_manager.current_transferred_client % mediator_length]:	
            '''
                #if as_manager.rank == 0 or as_manager.rank == 1:			
                size_original = b_x[int(b_x.size()[0]*0.8):b_x.size()[0]].size() # Supervised learning size
                #size_original = b_x.size()
                b_y = b_y[int(b_y.size()[0]*0.8):b_y.size()[0]] # Only use 10 percent samples for supervised learning
                b_x = b_x.cuda()
                b_y = b_y.cuda()
                b_x = b_x.view(-1, image_size)
                x_reconst, mu, log_var = vae_model(b_x) # Unsupervised learning
            

                # Compute reconstruction loss and kl divergence
                # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
                reconst_loss = F.mse_loss(x_reconst, b_x, size_average=False)
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                # Backprop and optimize
                loss = reconst_loss + kl_div
                vae_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                vae_optimizer.step()

                x_reconst_supervised = x_reconst[int(x_reconst.size()[0]*0.8):x_reconst.size()[0]].detach() # 10 percent for supervised training          
                #ypred = model(x_reconst_supervised.reshape(size_original)) #Supervised learning
                ypred = model(b_x[int(b_x.size()[0]*0.8):b_x.size()[0]].reshape(size_original))
                #ypred = model(b_x.reshape(size_original))
                added_loss = loss_func(ypred, b_y)
                optimizer.zero_grad()
                added_loss.backward()
                optimizer.step()
                #if iter_id % INIT_SYNC_FREQ == 10:
                #    print('iter id: ' + str(iter_id))
            '''
            x, y, u = Variable(x), Variable(y), Variable(u)
            x, y = x.cuda(), y.cuda()
            u = u.cuda()
            #logging('y: ' + str(y))
				
            L = -elbo(x, y)
            U = -elbo(u)
				
            # Add auxiliary classification loss q(y|x)
            logits = model.classify(x)
            classication_loss = torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
                
            J_alpha = L - alpha * classication_loss + U
				
            J_alpha.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += J_alpha.data
            labelled_loss += L.data
            unlabelled_loss += U.data
				
            _, pred_idx = torch.max(logits, 1)
            _, lab_idx = torch.max(y, 1)
            accuracy += torch.mean((pred_idx.data == lab_idx.data).float())
				
            
            iter_id += 1
        
            #logging('\niter_id: ' + str(iter_id))

                #print('current_transferred_client: ' + str(as_manager.current_transferred_client))
            			
           
            # sync(): synchronization function 
            #if as_manager.sync(model, iter_id):
            if as_manager.sync_within_mediator(model, iter_id):
            #if as_manager.rank == 0 or as_manager.rank == 1:
                '''
                accuracy = test(test_loader, model) 
                #print('mokaiwei')				
                if epoch_id != as_manager.last_test_epoch_id and epoch_id != 0:
                    logging('\n - test - accuracy:' + str(accuracy) + ';Reconst Loss:' + str(reconst_loss.item()) + '; round_id:' + str(as_manager.round_id) + '; epoch_id:' + str(epoch_id) + '; iter_id:' + str(iter_id) + '; sync_frequency:' + str(as_manager.sync_frequency) + '; time: ' + str((as_manager.round_id + 0.05*iter_id)/3600.0))
                   # print('loss: ' + str(loss))				
                as_manager.last_test_epoch_id = epoch_id
                '''
                if as_manager.last_test_epoch_id != epoch_id:
                    model.eval()
                    m = len(unlabelled)
                    logging("Epoch: {}".format(epoch_id))
                    logging('round_id: ' + str(as_manager.round_id))
                    logging("[Train]\t\t J_a: {:.2f}, L: {:.2f}, U: {:.2f}, accuracy: {:.2f}".format(total_loss / m,
                                                                                              labelled_loss / m,
                                                                                              unlabelled_loss / m,
                                                                                              accuracy / m))
																							  
                    total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
                    for x, y in validation:
                        x, y = Variable(x), Variable(y)

                        x, y = x.cuda(), y.cuda()
               
                        L = -elbo(x, y)
                        U = -elbo(x)

                        logits = model.classify(x)
                        classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                        J_alpha = L + alpha * classication_loss + U

                        total_loss += J_alpha.data
                        labelled_loss += L.data
                        unlabelled_loss += U.data

                        _, pred_idx = torch.max(logits, 1)
                        _, lab_idx = torch.max(y, 1)
                        accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

                    m = len(validation)
                    logging("within_mediator_Validation\t J_a: {:.3f}, L: {:.3f}, U: {:.3f}, accuracy: {:.3f}".format(total_loss / m,
                                                                                                  labelled_loss / m,
                                                                                                  unlabelled_loss / m,
                                                                                                  accuracy / m))
                    if as_manager.if_stable == 0:
                        if accuracy / m > best:
                            best = accuracy / m
                            as_manager.last_best_epoch_id = epoch_id
                        else:
                            if epoch_id - as_manager.last_best_epoch_id > as_manager.stable_window_size:
                                as_manager.if_stable = torch.tensor(1)
                    as_manager.last_test_epoch_id = epoch_id
					
            if as_manager.sync(model, iter_id):	
                for (i, p) in enumerate(model.parameters()): 
                    dist.broadcast(p.data, as_manager.rank_id[0], group=as_manager.within_mediator_group[as_manager.mediator_id-1])				
                if as_manager.last_mediator_test_epoch_id != epoch_id:
                    model.eval()
                    m = len(unlabelled)
                    logging("Epoch: {}".format(epoch_id))
                    logging('mediators_round_id: ' + str(as_manager.mediators_round_id))
																							  
                    total_loss, labelled_loss, unlabelled_loss, accuracy = (0, 0, 0, 0)
                    for x, y in validation:
                        x, y = Variable(x), Variable(y)

                        x, y = x.cuda(), y.cuda()
               
                        L = -elbo(x, y)
                        U = -elbo(x)

                        logits = model.classify(x)
                        classication_loss = -torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()

                        J_alpha = L + alpha * classication_loss + U

                        total_loss += J_alpha.data
                        labelled_loss += L.data
                        unlabelled_loss += U.data

                        _, pred_idx = torch.max(logits, 1)
                        _, lab_idx = torch.max(y, 1)
                        accuracy += torch.mean((pred_idx.data == lab_idx.data).float())

                    m = len(validation)
                    logging("mediators_Validation\t J_a: {:.3f}, L: {:.3f}, U: {:.3f}, accuracy: {:.3f}".format(total_loss / m,
                                                                                                  labelled_loss / m,
                                                                                                  unlabelled_loss / m,
                                                                                                  accuracy / m))
                    as_manager.last_mediator_test_epoch_id = epoch_id					


        epoch_id += 1
        as_manager.epoch_id = epoch_id

if __name__ == "__main__":

    logging('Trial ID: ' + str(args.trial_no) + '; Exp Setup Remarks: ' + args.remarks)
    logging('\nInitialization:\n\t model: ' + MODEL + '; dataset: ' + DATASET + '; batch_size: ' + str(BATCH_SIZE) + '; d_alpha: ' + str(D_ALPHA) 
        + '\n\t master_address: ' + str(MASTER_ADDRESS) + '; world_size: '+str(WORLD_SIZE) + '; rank: '+ str(RANK) 
        + '; weight_decay: ' + str(WEIGHT_DECAY)
        + ';\n\t mediators_sync_frequency: 1/'+str(INIT_SYNC_FREQ*10) + '.\n')

    train()
    