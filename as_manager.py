import numpy as np
import torch
import copy
from datetime import datetime
import datetime
import sys
try:
    import torch.distributed.deprecated as dist
except ImportError:
    import torch.distributed as dist
import numpy as np

CUDA = torch.cuda.is_available()
print('cuda' + str(CUDA))

def logging(string):
    print(str(datetime.datetime.now())+' '+str(string))
    sys.stdout.flush()
			

class AS_Manager:
    def __init__(self, pretrain_model, model, master_address, world_size, rank, sync_frequency):
        # Initialization
        dist.init_process_group(backend='gloo', init_method=master_address, timeout=datetime.timedelta(0, 7200), world_size=world_size, rank=rank)
        group = dist.new_group([i for i in range(world_size)])
        
		
        # send tensor to 1
        a = torch.tensor([3,5,7,2])
        if rank == 0:
            a += 1
            dist.send(tensor=a, dst=1)
        elif rank == 1:
            dist.recv(tensor=a, src=0)
        logging('a: ' + str(a))
        self.sync_frequency = sync_frequency
        self.group = group
        self.world_size = world_size # Number of clients
        self.rank = rank # Order number of clients
        self.model = model #  model
        self.pretrain_model = pretrain_model # pretrain model
        self.round_id = 0
        self.last_best_epoch_id = 0
        self.stable_window_size = 10
        self.if_stable = torch.tensor(0)
        self.if_global_stable = torch.tensor(0)
        self.mediators_round_id = 0
        self.epoch_id = 0
        self.pretrain_epoch_id = 0		
        self.pretrain_iter_id = 0  		
        self.last_test_epoch_id = 0
        self.last_mediator_test_epoch_id = 0
        self.last_test_pretrain_epoch_id = 0
        self.next_receiver = -1
        
        self.total_num_samples = 0
        self.h_size = 30
        self.mediators_allocate = np.zeros(self.world_size)
        self.mediators_allocate = self.mediators_allocate.astype(np.int64)
        self.mediators_client = np.zeros(self.world_size, dtype=np.int32)-1
        self.mediators_num = 0
        self.alpha = 0.5
        if self.rank == 0:
            self.model_delivered = torch.tensor(1)
        else:
            self.model_delivered = torch.tensor(0)
		
        # Deliver model from the server to the clients.
        for param in model.parameters():
            dist.broadcast(param.data, src=0, group=group)
        

        self.sync_mediators_freq = 10 # set a multiplier of sync_freq within mediators compared to that for clients
        self.next_sync_iter_id = self.sync_frequency
        self.next_sync_within_mediators = self.sync_frequency*self.sync_mediators_freq
        self.gathered_parameters = []
        self.model_size = 0
        self.gathered_parameters_added = []
        self.gathered_parameters_within_mediators = []
        self.model_size_added = 0
		
        self.global_mu_mu = torch.zeros(self.h_size)
        self.global_mu_sigma = torch.zeros(self.h_size)
        self.global_sigma_mu = torch.zeros(self.h_size)
        self.global_sigma_sigma = torch.zeros(self.h_size)
		
        self.gathered_num_samples = []
        self.gathered_mu_mu = []
        self.gathered_mu_sigma = []
        self.gathered_sigma_mu = []
        self.gathered_sigma_sigma = []
		
        self.mediator_id = 0
        self.rank_id = []
        self.mediator_length = 0
        self.within_mediator_group = []
        self.group_to = dist.new_group([0])
        
        # Calculate the number of model parameters
        for p in model.parameters():
            if rank == 0:
                self.gathered_parameters.append([copy.deepcopy(p.data) for i in range(world_size)])
                self.model_size += p.data.numel()
            else:
                self.gathered_parameters.append([])
                self.model_size += p.data.numel()
              
        for p in pretrain_model.parameters():
            if rank == 0:
                self.gathered_parameters_added.append([copy.deepcopy(p.data) for i in range(world_size)])
                self.model_size_added += p.data.numel()
            else:
                self.gathered_parameters_added.append([])
                self.model_size_added += p.data.numel()
        	
        if rank == 0:
            self.gathered_num_samples.append([copy.deepcopy(torch.tensor(world_size)) for i in range(world_size)])
            self.gathered_mu_mu.append([copy.deepcopy(torch.zeros(self.h_size)) for i in range(world_size)])
            self.gathered_mu_sigma.append([copy.deepcopy(torch.zeros(self.h_size)) for i in range(world_size)])
            self.gathered_sigma_mu.append([copy.deepcopy(torch.zeros(self.h_size)) for i in range(world_size)])
            self.gathered_sigma_sigma.append([copy.deepcopy(torch.zeros(self.h_size)) for i in range(world_size)])
        else:
            self.gathered_num_samples.append([])
            self.gathered_mu_mu.append([])
            self.gathered_mu_sigma.append([])
            self.gathered_sigma_mu.append([])
            self.gathered_sigma_sigma.append([])			
        if CUDA:
            torch.cuda.set_device(self.rank % torch.cuda.device_count())
            self.pretrain_model.cuda()
            self.model.cuda()
            #self.model_added.cuda()
        logging('mokaiwei5')
        logging('vae_model size: ' + str(self.model_size) + '; model size: ' + str(self.model_size_added))
		
        
		
	# Calculate the number of samples, add in as_manager.py
    def total_samples(self, num_of_samples):
        dist.gather(num_of_samples, gather_list=self.gathered_num_samples[0], dst=0, group=self.group)
        if self.rank == 0:
            self.total_num_samples = sum(self.gathered_num_samples[0])
        print("total number of samples: " + str(self.total_num_samples))
        print("num sample list: " + str(self.gathered_num_samples))	

    # Get global mu and sigma
    def sync_mu_and_sigma(self, client_mu_mu, client_mu_sigma, client_sigma_mu, client_sigma_sigma):
        # Calculate global_mu_mu
        dist.gather(client_mu_mu, gather_list=self.gathered_mu_mu[0], dst=0, group=self.group)
        if self.rank == 0:
            for i in range(self.world_size):
                self.global_mu_mu += self.gathered_num_samples[0][i] * self.gathered_mu_mu[0][i]
            self.global_mu_mu = self.global_mu_mu / self.total_num_samples
        dist.broadcast(self.global_mu_mu, 0, group=self.group) # Broadcast global mu_mu to every client

        # Calculate global_mu_sigma
        dist.gather(client_mu_sigma, gather_list=self.gathered_mu_sigma[0], dst=0, group=self.group)
        if self.rank == 0:
            for i in range(self.world_size):
                self.global_mu_sigma += (self.gathered_num_samples[0][i] - 1) * self.gathered_mu_sigma[0][i] + self.gathered_num_samples[0][i] * (self.gathered_mu_mu[0][i] - self.global_mu_mu).pow(2)
            self.global_mu_sigma = self.global_mu_sigma / (self.total_num_samples - 1) # Calculate global mu_sigma
        dist.broadcast(self.global_mu_sigma, 0, group=self.group) # Broadcast global mu to every client

        # Calculate global_sigma_mu
        dist.gather(client_sigma_mu, gather_list=self.gathered_sigma_mu[0], dst=0, group=self.group)
        if self.rank == 0:
            for i in range(self.world_size):
                self.global_sigma_mu += self.gathered_num_samples[0][i] * self.gathered_sigma_mu[0][i]
            self.global_sigma_mu = self.global_sigma_mu / self.total_num_samples
        dist.broadcast(self.global_sigma_mu, 0, group=self.group) # Broadcast global mu to every client

        # Calculate global_sigma_sigma
        dist.gather(client_sigma_sigma, gather_list=self.gathered_sigma_sigma[0], dst=0, group=self.group)
        if self.rank == 0:
            for i in range(self.world_size):
                self.global_sigma_sigma += (self.gathered_num_samples[0][i] - 1) * self.gathered_sigma_sigma[0][i] + self.gathered_num_samples[0][i] * (self.gathered_sigma_mu[0][i] - self.global_sigma_mu).pow(2)
            self.global_sigma_sigma = self.global_sigma_sigma / (self.total_num_samples - 1)
        dist.broadcast(self.global_sigma_sigma, 0, group=self.group)

        logging("finish synchronizing mu and sigma.")	
		
        if self.rank == 0:
            self.set_mediator()
			
        #test
        b = torch.tensor([3,5,7,2])
        if self.rank == 0:
            b = b + 1
        dist.broadcast(tensor=b, src=0, group=self.group)
        logging('b: ' + str(b))
			
        mediators_num_transfered = torch.tensor(self.mediators_num)
        dist.broadcast(mediators_num_transfered, src=0, group=self.group)
        self.mediators_num = int(mediators_num_transfered)
        mediators_allocate_transfered = torch.tensor(self.mediators_allocate)
        dist.broadcast(mediators_allocate_transfered, src=0, group=self.group)
        self.mediators_allocate = mediators_allocate_transfered.detach().numpy()
        mediators_client_transfered = torch.tensor(self.mediators_client)
        dist.broadcast(mediators_client_transfered, 0, group=self.group)
        self.mediators_client = mediators_client_transfered.detach().numpy()
		
		
        # sync within mediator		
        self.mediator_id = self.mediators_allocate[self.rank]
        self.rank_id = list(np.arange(self.world_size)[np.array(self.mediators_allocate) == self.mediator_id])
        self.mediator_length = len(self.rank_id)
        logging('mediator_id: ' + str(self.mediator_id))
        logging('rank_id ' + str(self.rank_id))
        logging('mediator_length: ' + str(self.mediator_length))
        logging('mediators_num: ' + str(self.mediators_num))
        logging('mediators_allocate: ' + str(self.mediators_allocate))
        '''
        for i in range(self.mediator_length):
            self.rank_id[i] = int(self.rank_id[i])
            #logging(str(rank_id[i]))
        self.within_mediator_group = dist.new_group(self.rank_id, timeout=datetime.timedelta(0, 7200))
        '''
        for i in range(self.mediators_num):
            temp = list(np.arange(self.world_size)[np.array(self.mediators_allocate) == (i+1)])
            for j in range(len(temp)):
                temp[j] = int(temp[j])
            logging('temp: ' + str(temp))
            group_temp = dist.new_group(temp)
            self.within_mediator_group.append(group_temp)

		
        #sync
        mediators_client_temp = self.mediators_client.tolist()
        temp_index = mediators_client_temp.index(-1)
        mediators_client_temp = mediators_client_temp[:temp_index]		
        self.group_to = dist.new_group(mediators_client_temp)
        
        
			

    # Calculate kl divergence
    def kl_divergence(self, mu1, mu2, sigma1, sigma2):
        return (torch.sum(0.5*torch.log(sigma2/sigma1) + ((mu1-mu2).pow(2) + sigma1) / (2*sigma2) -0.5))

    # Calculate mediator mu and sigma
    def calculate_mu_and_sigma(self, mu_list, mu1, sigma_list, sigma1, num_samples_list, num_samples1):
        total_num_samples = sum(num_samples_list) + num_samples1
        total_mu = mu1 * num_samples1
        for i in range(len(mu_list)):
            total_mu += mu_list[i] * num_samples_list[i]
        current_mu = total_mu / total_num_samples

        total_sigma = (num_samples1 - 1) * sigma1 + num_samples1 * ((mu1 - current_mu).pow(2))
        for i in range(len(sigma_list)):
            total_sigma += (num_samples_list[i] - 1) * sigma_list[i] + num_samples_list[i] * ((mu_list[i] - current_mu).pow(2))
        current_sigma = total_sigma / (total_num_samples - 1)
        return current_mu, current_sigma

    # Allocate clients for mediator after pre-train, add in as_manager.py
    def set_mediator(self):
        clients_no_allocate = self.world_size
        allocate_flag = np.ones(self.world_size)
        num_mediator = 1
        #print('gathered mu mu: ' + str(self.gathered_mu_mu[0]))
        #print('gathered mu mu: ' + str(self.gathered_mu_sigma[0]))
        #print('gathered mu mu: ' + str(self.gathered_sigma_mu[0]))
        #print('gathered mu mu: ' + str(self.gathered_sigma_sigma[0]))
		

        while clients_no_allocate > 0:
            # xxx_in_mediator record the information(mu,sigma,and num_sample) used in the current mediator
            num_samples_in_mediator = []
            mu_mu_in_mediator = []
            mu_sigma_in_mediator = []
            sigma_mu_in_mediator = []
            sigma_sigma_in_mediator = []

            # Select a client for a mediator
            for i in range(self.world_size):
                if allocate_flag[i] > 0:
                    allocate_flag[i] = 0
                    clients_no_allocate = clients_no_allocate - 1
                    self.mediators_allocate[i] = num_mediator
                    num_samples_in_mediator.append(self.gathered_num_samples[0][i])
                    mu_mu_in_mediator.append(self.gathered_mu_mu[0][i])
                    mu_sigma_in_mediator.append(self.gathered_mu_sigma[0][i])
                    sigma_mu_in_mediator.append(self.gathered_sigma_mu[0][i])
                    sigma_sigma_in_mediator.append(self.gathered_sigma_sigma[0][i])
                    break

            # The first last kl divergence is one client with the global
            #logging('self.gathered_mu_mu[0][i]: ' + str(self.gathered_mu_mu[0][i]) + ' ;self.global_mu_mu: ' + str(self.global_mu_mu) + ' ;self.gathered_mu_sigma[0][i]: ' + str(self.gathered_mu_sigma[0][i]) + ' ;self.global_mu_sigma: ' + str(self.global_mu_sigma))
            #logging('self.gathered_sigma_mu[0][i]: ' + str(self.gathered_sigma_mu[0][i]) + ' ;self.global_sigma_mu: ' + str(self.global_sigma_mu) + ' ;self.gathered_sigma_sigma[0][i]: ' + str(self.gathered_sigma_sigma[0][i]) + ' ;self.global_sigma_sigma: ' + str(self.global_sigma_sigma))
            last_mu_kl_div = self.kl_divergence(self.gathered_mu_mu[0][i], self.global_mu_mu, self.gathered_mu_sigma[0][i], self.global_mu_sigma)
            last_sigma_kl_div = self.kl_divergence(self.gathered_sigma_mu[0][i], self.global_sigma_mu, self.gathered_sigma_sigma[0][i], self.global_sigma_sigma)
            #print('last mu kl div: ' + str(last_mu_kl_div))
            #print('last sigma kl div: ' + str(last_sigma_kl_div))
            last_kl_div = self.alpha * last_mu_kl_div + (1 - self.alpha) * last_sigma_kl_div

            # Calculate the most suitable clients for current mediator
            while True:
                if clients_no_allocate < 1:
                    break

                # Calculate minimum kl divergence
                min_kl_div = torch.tensor(99999.0)
                client_index = -1
                for i in range(self.world_size):
                    if allocate_flag[i] > 0:
                        current_mu_mu, current_mu_sigma =  self.calculate_mu_and_sigma(mu_mu_in_mediator, self.gathered_mu_mu[0][i], mu_sigma_in_mediator, self.gathered_mu_sigma[0][i], num_samples_in_mediator, self.gathered_num_samples[0][i])
                        current_mu_kl_div = self.kl_divergence(current_mu_mu, self.global_mu_mu, current_mu_sigma, self.global_mu_sigma)
                        current_sigma_mu, current_sigma_sigma =  self.calculate_mu_and_sigma(sigma_mu_in_mediator, self.gathered_sigma_mu[0][i], sigma_sigma_in_mediator, self.gathered_sigma_sigma[0][i], num_samples_in_mediator, self.gathered_num_samples[0][i])
                        current_sigma_kl_div = self.kl_divergence(current_sigma_mu, self.global_sigma_mu, current_sigma_sigma, self.global_sigma_sigma)
                        current_kl_div = self.alpha * current_mu_kl_div + (1 - self.alpha) * current_sigma_kl_div
						
                        #print("current_kl_div type: " + str(type(current_kl_div)))
                        #print("min_kl_div type: " + str(type(min_kl_div)))						

                        if current_kl_div < min_kl_div:
                            min_kl_div = current_kl_div
                            client_index = i

                current_kl_div = min_kl_div
                logging("last kl div: " + str(last_kl_div))
                logging("current kl div: " + str(current_kl_div))

                # allocate the client for the current mediator
                if current_kl_div < last_kl_div:
                    #print('mokaiwei')
                    last_kl_div = current_kl_div
                    self.mediators_allocate[client_index] = num_mediator
                    allocate_flag[client_index] = 0
                    num_samples_in_mediator.append(self.gathered_num_samples[0][client_index])
                    mu_mu_in_mediator.append(self.gathered_mu_mu[0][client_index])
                    mu_sigma_in_mediator.append(self.gathered_mu_sigma[0][client_index])
                    sigma_mu_in_mediator.append(self.gathered_sigma_mu[0][client_index])
                    sigma_sigma_in_mediator.append(self.gathered_sigma_sigma[0][client_index])
                    clients_no_allocate = clients_no_allocate - 1
                    if clients_no_allocate < 1:
                        break
                else:
                    #print('abb')
                    num_mediator = num_mediator + 1
                    break         # end and save current mediator		

        self.mediators_allocate = np.array([1,2,3,1,2,3,1,2,3,4,5,6,4,5,6,4,5,6,7,8,8,7,7,8])					
        unique_id = np.unique(self.mediators_allocate)
        mediators_client_temp = []
        for j in unique_id:
            mediators_client_temp.append(int((np.arange(self.world_size)[self.mediators_allocate == j])[0]))
        self.mediators_num = len(mediators_client_temp)
        mediators_client_temp = np.array(mediators_client_temp)
        for i in range(len(mediators_client_temp)):
            self.mediators_client[i] = mediators_client_temp[i]
		

        logging("mediator allocate: " + str(self.mediators_allocate))
        logging("finish setting mediators.")
        
    def sync(self, model, iter_id):
        if self.if_global_stable == 0:		
            if self.rank == 0:
                sign_list = [torch.zeros_like(self.if_stable) for _ in range(self.mediators_num)]
            else:
                sign_list = []
            dist.gather(self.if_stable, gather_list=sign_list, dst = 0, group=self.group_to)
            #logging('sign_list: ' + str(sign_list))
            for i in range(len(sign_list)):
                if sign_list[i] == 0:
                    break
                if i == len(sign_list)-1 and sign_list[i] == 1:
                    self.next_sync_within_mediators = iter_id
                    self.if_global_stable = torch.tensor(1)
            dist.broadcast(self.if_global_stable, 0, group=self.group)
            next_sync_within_mediators_transfered = torch.tensor(self.next_sync_within_mediators)
            dist.broadcast(next_sync_within_mediators_transfered, 0, group=self.group)
            self.next_sync_within_mediators = int(next_sync_within_mediators_transfered)
            #logging('if_global_stable: ' + str(self.if_global_stable))
        if iter_id == self.next_sync_within_mediators and self.if_global_stable == 1:
            if self.rank in self.mediators_client:
                if self.mediators_num > 1:    
                    if CUDA:
                        model.cpu()
                        #model_added.cpu()
                    # Aggregation
            

                    for (i, p) in enumerate(model.parameters()):
                        if self.rank == 0:
                            grad_list = [torch.zeros_like(p.data) for _ in range(self.mediators_num)]
                        else:
                            grad_list = []
                        dist.gather(p.data, gather_list=grad_list, dst = 0, group=self.group_to)
                        if self.rank == 0:
                            p.data = sum(grad_list) / self.mediators_num # reduce to average
                        dist.broadcast(p.data, 0, group=self.group_to)
					
                   
                    
                
                    if CUDA:
                        model.cuda()
                        #model_added.cuda()
				
            self.next_sync_within_mediators = iter_id + self.sync_frequency*self.sync_mediators_freq
            self.mediators_round_id += 1
            #self.last_model = copy.deepcopy(model)
            #self.last_model_added = copy.deepcopy(model_added)

            return True
        return False
		
    def sync_without_mediator(self, model, iter_id):
        if iter_id == self.next_sync_iter_id:
            #if self.mediators_num > 1:    
            if CUDA:
                model.cpu()
                #model_added.cpu()
            # Aggregation
            
            		
            
            for (i, p) in enumerate(model.parameters()):
                dist.gather(p.data, gather_list=self.gathered_parameters[i], dst=0, group=self.group)
                if self.rank == 0:
                    p.data = sum(self.gathered_parameters[i]) / self.world_size # reduce to average
                dist.broadcast(p.data, 0, group=self.group)
          
            if CUDA:
                model.cuda()
               # model_added.cuda()
				
            self.next_sync_iter_id = iter_id + self.sync_frequency
            self.round_id += 1
            #self.last_model = copy.deepcopy(model)
            #self.last_model_added = copy.deepcopy(model_added)

            return True
        return False
		
		
    
    def sync_within_mediator(self, model, iter_id):

        if iter_id == self.next_sync_iter_id:
            if CUDA:
                model.cpu()
                #model_added.cpu()
            '''
            if self.rank == self.rank_id[self.current_transferred_client % self.mediator_length]:
                for (i, p) in enumerate(model.parameters()):
                    p_data = torch.zeros_like(p.data)
                    dist.send(tensor=p_data, dst=int(self.rank_id[(self.current_transferred_client+1) % self.mediator_length]))
            elif self.rank == self.rank_id[(self.current_transferred_client+1) % self.mediator_length]:
                for (i, p) in enumerate(model.parameters()):
                    p_data = torch.zeros_like(p.data)
                    dist.recv(tensor=p_data, src=int(self.rank_id[self.current_transferred_client % self.mediator_length]))
            self.next_sync_iter_id = iter_id + self.sync_frequency
            self.current_transferred_client += 1
            '''
			
            for (i, p) in enumerate(model.parameters()):
                if self.rank == self.rank_id[0]:
                    grad_list = [torch.zeros_like(p.data) for _ in range(self.mediator_length)]
                else:
                    grad_list = []				
                dist.gather(p.data, gather_list=grad_list, dst=self.rank_id[0], group=self.within_mediator_group[self.mediator_id-1])
                if self.rank == self.rank_id[0]:
                    p.data = sum(grad_list) / self.mediator_length # reduce to average
                dist.broadcast(p.data, self.rank_id[0], group=self.within_mediator_group[self.mediator_id-1])
            	
            if CUDA:
                model.cuda()
                #model_added.cuda()
				
            self.next_sync_iter_id = iter_id + self.sync_frequency
            self.round_id += 1
            
            return True
        #dist.broadcast(tensor=self.current_transferred_client, src=int(self.rank_id[self.current_transferred_client % self.mediator_length]), group=self.within_mediator_group[self.mediator_id-1])	
        return False
        