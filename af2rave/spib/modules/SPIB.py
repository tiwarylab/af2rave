"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import torch
from torch import nn
import numpy as np
import os
import torch.nn.functional as F
        
# --------------------
# Model
# --------------------   

class SPIB(nn.Module):

    def __init__(self, encoder_type, z_dim, output_dim, data_shape, device, UpdateLabel= False, neuron_num1=128, 
                 neuron_num2=128):
        
        super(SPIB, self).__init__()
        if encoder_type == 'Nonlinear':
            self.encoder_type = 'Nonlinear'
        else:
            self.encoder_type = 'Linear'

        self.z_dim = z_dim
        self.output_dim = output_dim
        
        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2
        
        self.data_shape = data_shape
        
        self.UpdateLabel = UpdateLabel
        
        self.eps = 1e-10
        self.device = device
        
        

        # representative-inputs
        self.representative_dim = output_dim

        # torch buffer, these variables will not be trained
        self.representative_inputs = torch.eye(self.output_dim, np.prod(self.data_shape), device=device, requires_grad=False)
        
        # create an idle input for calling representative-weights
        # torch buffer, these variables will not be trained
        self.idle_input = torch.eye(self.output_dim, self.output_dim, device=device, requires_grad=False)

        # representative weights
        self.representative_weights = nn.Sequential(
            nn.Linear(self.output_dim, 1, bias=False),
            nn.Softmax(dim=0))
        
        self.encoder = self._encoder_init()

        if self.encoder_type == 'Nonlinear': 
            self.encoder_mean = nn.Linear(self.neuron_num1, self.z_dim)
        else:
            self.encoder_mean = nn.Linear(np.prod(self.data_shape), self.z_dim)
        
        # Note: encoder_type = 'Linear' only means that z_mean is a linear combination of the input OPs, 
        # the log_var is always obtained through a nonlinear NN

        # enforce log_var in the range of [-10, 0]
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.neuron_num1, self.z_dim),
            nn.Sigmoid())
        
        self.decoder = self._decoder_init()
        
    def _encoder_init(self):
        
        modules = [nn.Linear(np.prod(self.data_shape), self.neuron_num1)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num1, self.neuron_num1)]
            modules += [nn.ReLU()]
        
        return nn.Sequential(*modules)
    
    def _decoder_init(self):
        # cross-entropy MLP decoder
        # output the probability of future state
        modules = [nn.Linear(self.z_dim, self.neuron_num2)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num2, self.neuron_num2)]
            modules += [nn.ReLU()]
        
        modules += [nn.Linear(self.neuron_num2, self.output_dim)]
        modules += [nn.LogSoftmax(dim=1)]
        
        return nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def encode(self, inputs):
        enc = self.encoder(inputs)
        
        if self.encoder_type == 'Nonlinear': 
            z_mean = self.encoder_mean(enc)
        else:
            z_mean = self.encoder_mean(inputs)

        # Note: encoder_type = 'Linear' only means that z_mean is a linear combination of the input OPs, 
        # the log_var is always obtained through a nonlinear NN
        
        # enforce log_var in the range of [-10, 0]
        z_logvar = -10*self.encoder_logvar(enc)
        
        return z_mean, z_logvar
    
    def forward(self, data):
        inputs = torch.flatten(data, start_dim=1)
        
        z_mean, z_logvar = self.encode(inputs)
        
        z_sample = self.reparameterize(z_mean, z_logvar)
        
        outputs = self.decoder(z_sample)
        
        return outputs, z_sample, z_mean, z_logvar
    
    def log_p (self, z, sum_up=True):
        # get representative_z - representative_dim * z_dim
        representative_z_mean, representative_z_logvar = self.get_representative_z()
        # get representative weights - representative_dim * 1
        w = self.representative_weights(self.idle_input)
        # w = 0.5*torch.ones((2,1)).to(self.device)
        
        # expand z - batch_size * z_dim
        z_expand = z.unsqueeze(1)
        
        representative_mean = representative_z_mean.unsqueeze(0)
        representative_logvar = representative_z_logvar.unsqueeze(0)
        
        # representative log_q
        representative_log_q = -0.5 * torch.sum(representative_logvar + torch.pow(z_expand-representative_mean, 2)
                                        / torch.exp(representative_logvar), dim=2 )
        
        if sum_up:
            log_p = torch.sum(torch.log(torch.exp(representative_log_q)@w + self.eps), dim=1)
        else:
            log_p = torch.log(torch.exp(representative_log_q)*w.T + self.eps)  
            
        return log_p
        
    # the prior
    def get_representative_z(self):
        # calculate representative_means
        # with torch.no_grad():
        X = self.representative_inputs

        # calculate representative_z
        representative_z_mean, representative_z_logvar = self.encode(X)  # C x M

        return representative_z_mean, representative_z_logvar

    def reset_representative(self, representative_inputs):
        
        # reset the nuber of representative inputs   
        self.representative_dim = representative_inputs.shape[0]        
        
        # reset representative weights
        self.idle_input = torch.eye(self.representative_dim, self.representative_dim, device=self.device, requires_grad=False)

        self.representative_weights = nn.Sequential(
            nn.Linear(self.representative_dim, 1, bias=False),
            nn.Softmax(dim=0))
        self.representative_weights[0].weight = nn.Parameter(torch.ones([1, self.representative_dim], device=self.device))
        
        # reset representative inputs
        self.representative_inputs = representative_inputs.clone().detach()
        
    @torch.no_grad()
    def init_representative_inputs(self, inputs, labels):
        state_population = labels.sum(dim=0).cpu()
        
        # randomly pick up one sample from each initlal state as the initial guess of representative-inputs
        representative_inputs=[]
        
        for i in range(state_population.shape[-1]):
            if state_population[i]>0:
                index = np.random.randint(0,state_population[i])
                representative_inputs+=[inputs[labels[:,i].bool()][index].reshape(1,-1)]
                # print(index)
        
        representative_inputs = torch.cat(representative_inputs, dim=0)

        self.reset_representative(representative_inputs.to(self.device))
            
        return representative_inputs

    @torch.no_grad()
    def estimatate_representative_inputs(self, inputs, bias, batch_size):
        prediction = []
        mean_rep = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size].to(self.device)
        
            # pass through VAE
            z_mean, z_logvar = self.encode(batch_inputs)        
            log_prediction = self.decoder(z_mean)
            
            # label = p/Z
            prediction += [log_prediction.exp()]
            
            mean_rep += [z_mean]
        
        prediction = torch.cat(prediction, dim=0)
        mean_rep = torch.cat(mean_rep, dim=0)
        
        max_pos = prediction.argmax(1)
        labels = F.one_hot(max_pos, num_classes=self.output_dim)
        
        state_population = labels.sum(dim=0)
        
        # save new guess of representative-inputs
        representative_inputs=[]
        
        for i in range(state_population.shape[-1]):
            if state_population[i]>0:
                if bias == None:
                    center_z = ((mean_rep[labels[:,i].bool()]).mean(dim=0)).reshape(1,-1)
                else:
                    weights = bias[labels[:,i].bool()].reshape(-1,1)
                    center_z = ((weights*mean_rep[labels[:,i].bool()]).sum(dim=0)/weights.sum()).reshape(1,-1)
                
                # find the one cloest to center_z as representative-inputs
                dist=torch.square(mean_rep-center_z).sum(dim=-1)                
                index = torch.argmin(dist)
                representative_inputs+=[inputs[index].reshape(1,-1)]
                # print(index)
        
        representative_inputs = torch.cat(representative_inputs, dim=0)
            
        return representative_inputs
            
    @torch.no_grad()
    def update_labels(self, inputs, batch_size):
        if self.UpdateLabel:
            labels = []
            
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i+batch_size].to(self.device)
            
                # pass through VAE
                z_mean, z_logvar = self.encode(batch_inputs)        
                log_prediction = self.decoder(z_mean)
                
                # label = p/Z
                labels += [log_prediction.exp()]
            
            labels = torch.cat(labels, dim=0)
            max_pos = labels.argmax(1)
            labels = F.one_hot(max_pos, num_classes=self.output_dim)
            
            return labels
    
    @torch.no_grad()
    def save_representative_parameters(self, path, index=0):
        
        # output representative centers
        representative_path = path + '_representative_inputs' + str(index) + '.npy'
        representative_weight_path = path + '_representative_weight' + str(index) + '.npy'
        representative_z_mean_path = path + '_representative_z_mean' + str(index) + '.npy'
        representative_z_logvar_path = path + '_representative_z_logvar' + str(index) + '.npy'
        os.makedirs(os.path.dirname(representative_path), exist_ok=True)
        
        np.save(representative_path, self.representative_inputs.cpu().data.numpy())
        np.save(representative_weight_path, self.representative_weights(self.idle_input).cpu().data.numpy())
        
        representative_z_mean, representative_z_logvar = self.get_representative_z()
        np.save(representative_z_mean_path, representative_z_mean.cpu().data.numpy())
        np.save(representative_z_logvar_path, representative_z_logvar.cpu().data.numpy())
        
    @torch.no_grad()
    def save_traj_results(self, inputs, batch_size, path, SaveTrajResults, traj_index=0, index=1):
        all_prediction=[] 
        all_z_sample=[] 
        all_z_mean=[] 
        
        for i in range(0, len(inputs), batch_size):
            
            batch_inputs = inputs[i:i+batch_size].to(self.device)
        
            # pass through VAE
            z_mean, z_logvar = self.encode(batch_inputs)
            z_sample = self.reparameterize(z_mean, z_logvar)
        
            log_prediction = self.decoder(z_mean)
            
            all_prediction+=[log_prediction.exp().cpu()]
            all_z_sample+=[z_sample.cpu()]
            all_z_mean+=[z_mean.cpu()]
            
        all_prediction = torch.cat(all_prediction, dim=0)
        all_z_sample = torch.cat(all_z_sample, dim=0)
        all_z_mean = torch.cat(all_z_mean, dim=0)
        
        max_pos = all_prediction.argmax(1)
        labels = F.one_hot(max_pos, num_classes=self.output_dim)
        
        # save the fractional population of different states
        population = torch.sum(labels,dim=0).float()/len(inputs)
        
        population_path = path + '_traj%d_state_population'%(traj_index) + str(index) + '.npy'
        os.makedirs(os.path.dirname(population_path), exist_ok=True)
        
        np.save(population_path, population.cpu().data.numpy())
        
        self.save_representative_parameters(path, index)

        # if the encoder is linear, output the parameters of the linear encoder
        if self.encoder_type == 'Linear': 
            z_mean_encoder_weight_path = path + '_z_mean_encoder_weight' + str(index) + '.npy'
            z_mean_encoder_bias_path = path + '_z_mean_encoder_bias' + str(index) + '.npy'
            os.makedirs(os.path.dirname(z_mean_encoder_weight_path), exist_ok=True)

            np.save(z_mean_encoder_weight_path, self.encoder_mean.weight.cpu().data.numpy())
            np.save(z_mean_encoder_bias_path, self.encoder_mean.bias.cpu().data.numpy())
            
        if SaveTrajResults:
        
            label_path = path + '_traj%d_labels'%(traj_index) + str(index) + '.npy'
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            np.save(label_path, labels.cpu().data.numpy())
            
            prediction_path = path + '_traj%d_data_prediction'%(traj_index) + str(index) + '.npy'
            representation_path = path + '_traj%d_representation'%(traj_index) + str(index) + '.npy'
            mean_representation_path = path + '_traj%d_mean_representation'%(traj_index) + str(index) + '.npy'
            
            os.makedirs(os.path.dirname(mean_representation_path), exist_ok=True)
            
            np.save(prediction_path, all_prediction.cpu().data.numpy())
            np.save(representation_path, all_z_sample.cpu().data.numpy())
            np.save(mean_representation_path, all_z_mean.cpu().data.numpy())
            
            

                
            
            
        
