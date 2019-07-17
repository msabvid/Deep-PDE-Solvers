import os
import numpy as np
import torch
import torch.nn as nn
import time
from numpy.linalg import norm
import copy
import math
from numpy.linalg import cholesky
import argparse


class Net(nn.Module):
    
    def __init__(self, dim, nOut, n_layers, vNetWidth, activation = "relu"):
        super(Net_timestep_big, self).__init__()
        self.dim = dim
        self.nOut = nOut
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation=="tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError("unknown activation function {}".format(activation))
        
        self.i_h = self.hiddenLayerT1(dim, vNetWidth)
        self.h_h = nn.ModuleList([self.hiddenLayerT1(vNetWidth, vNetWidth) for l in range(n_layers-1)])
        self.h_o = self.outputLayer(vNetWidth, nOut)
        
    def hiddenLayerT1(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn,nOut,bias=True),
                              self.activation)   
        return layer
    
    
    def outputLayer(self, nIn, nOut):
        layer = nn.Sequential(nn.Linear(nIn, nOut,bias=True))#,
        return layer
    
    def forward(self, S):
        h = self.i_h(S)
        for l in range(len(self.h_h)):
            h = self.h_h[l](h)
        output = self.h_o(h)
        return output


class BSDE_solver(nn.Module):
    
    def __init__(self, dim, r, sigma, covariance_mat, timegrid, n_layers, vNetWidth = 100, gradNetWidth=100):
        super(BSDE_solver, self).__init__()
        self.dim = dim
        self.timegrid = torch.Tensor(timegrid).to(device)
        self.r = r
        self.sigma = torch.Tensor(sigma).to(device) # this should be a vector of length dim
        self.covariance_mat = covariance_mat # covariance matrix
        self.C = cholesky(covariance_mat) # Cholesky decomposition of covariance matrix, with size (dim,dim)
        
        self.volatility_mat = torch.Tensor(self.C).to(device)  
        for i in range(self.dim):
            self.volatility_mat[i] = self.volatility_mat[i]*self.sigma[i]

        
        # Network for gradient
        self.net_timegrid_gradient = Net(dim=dim+1, nOut=dim, n_layers=n_layers, vNetWidth=vNetWidth)
        self.net_timegrid_value = Net(dim=dim+1, nOut=1, n_layers=n_layers, vNetWidth=vNetWidth)
    
    
    def control_variate(self, S0, imin=0, AD=False):
        S_old = S0
        timegrid_mat = self.timegrid.view(1,-1).repeat(S0.shape[0], 1)
        control_variate=0
        for i in range(imin+1, len(self.timegrid)):
            # Wiener process at time timegrid[i]
            h = self.timegrid[i]-self.timegrid[i-1]
            dW = math.sqrt(h)*torch.randn(S_old.data.size(), device=device)#.to(device)
            
            # volatility(t,S) * dW
            volatility_of_S_dW = S_old * torch.matmul(self.volatility_mat,dW.transpose(1,0)).transpose(1,0) # this is a matrix of size (batch_size x dim)
            
            # grad of value function and stochastic integral
            input_gradient = torch.cat([timegrid_mat[:,i-1].view(-1,1), S_old],1)
            Z = torch.exp(-self.r * self.timegrid[i-1])*self.net_timegrid_gradient(input_gradient)#self.runNetForGrad(i-1,S_old)
            stoch_int = torch.diag(torch.matmul(Z, volatility_of_S_dW.transpose(1,0))).view(-1,1)
            
            # control variate
            control_variate += stoch_int
            
            # we update the SDE path
            S_new = S_old  + self.r*S_old*h + volatility_of_S_dW 
            
            S_old = S_new
            
        return S_old, control_variate
            
    def forward(self, S0):        
        error = 0.0
        S_old = S0
        timegrid_mat = self.timegrid.view(1,-1).repeat(S0.shape[0],1)
        input_net = torch.cat([timegrid_mat[:,0].view(-1,1), S_old],1)
        vOld = self.net_timegrid_value(input_net)#self.runNetForV(0,S_old)
        for i in range(1,len(self.timegrid)):
            # Wiener process at time timegrid[i]
            h = self.timegrid[i]-self.timegrid[i-1]
            dW = math.sqrt(h)*torch.randn(S_old.data.size(), device=device)#.to(device)
                        
            # volatility(t,S) * dW
            volatility_of_S_dW = S_old * torch.matmul(self.volatility_mat,dW.transpose(1,0)).transpose(1,0) # this is a matrix of size (batch_size x dim)
            
            # grad of value function and stochastic integral
            input_gradient = torch.cat([timegrid_mat[:,i-1].view(-1,1), S_old],1)
            grad = self.net_timegrid_gradient(input_gradient)#self.runNetForGrad(i-1,S_old)
            stoch_int = torch.bmm(grad.unsqueeze(1), volatility_of_S_dW.unsqueeze(2)).squeeze(1)
            #stoch_int = torch.diag(torch.matmul(grad, volatility_of_S_dW.transpose(1,0))).view(-1,1)
            
            # we update the SDE path. Use one or the other. 
            S_new = S_old  + self.r*S_old*h + volatility_of_S_dW 
                
            # error from step i
            input_val = torch.cat([timegrid_mat[:,i].view(-1,1), S_new],1)
            vNew = self.net_timegrid_value(input_val)
            error_from_step_i = vNew.detach() - vOld*(1+self.r*h) - stoch_int
            error += (error_from_step_i**2)
            
            # we are done, prepare for next round
            S_old = S_new
            vOld = vNew
            
            
        return vOld, S_old, error

def train():
    n_iter = 20000   
    for it in range(n_iter):
        model.train()
        lr = base_lr * (0.1**(it//10000))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr']=lr
        optimizer.zero_grad()
        z = torch.randn([batch_size, dim]).to(device)
        input = torch.exp((mu-0.5*sigma**2)*tau + math.sqrt(tau)*z)
        init_time = time.time()
        v_of_S_T, S_T, error = model(input) 
        time_forward = time.time() - init_time
        
        target = g(S_T).view(-1,1)
        loss = (1.0/batch_size)*(torch.sum((target-v_of_S_T)**2)+torch.sum(error))
        init_time = time.time()
        loss.backward()
        time_backward = time.time() - init_time
        optimizer.step()
        

        with open(file_log_path, 'a') as f:
            f.write("Iteration=[{}/{}]\t loss={:.8f}\t time forward pass={:.3f}\t time backward pass={:.3f}\n".format(it,n_iter, loss.item(), time_forward, time_backward))

        if (it+1)%1000==0:
            state = {'epoch':it+1, 'state_dict':model.state_dict()}#, 'optimizer':optimizer.state_dict()}
            filename = 'model_2nets_onebig_'+str(n_layers)+'_'+str(vNetWidth)+'_'+str(timestep)+'_'+str(dim)+'_it'+str(it)+'.pth.tar'
            torch.save(state, filename)
        if (it+1)%100==0:
            var_MC_CV_estimator, var_MC_estimator, MC_CV_estimator, MC_estimator, corr_terminal_control_variate = get_prediction_CV(batch_size_MC=1000)
            with open(file_log_results, 'a') as f:
                    f.write('{},{},{},{},{}\n'.format(var_MC_CV_estimator, var_MC_estimator, MC_CV_estimator, MC_estimator, corr_terminal_control_variate))


####################
# Control variate ##
####################

def get_prediction_CV(batch_size_MC=1000):
    model.eval()
    #batch_size_MC=100000
    
    with torch.no_grad():
        if batch_size_MC > 1000:
            terminal_list = []
            control_variate_list = []
            for i in range(batch_size_MC//1000):
                print(i)
                input = torch.ones(1000, dim, device=device)
                with torch.nograd():
                    S_T, control_variate = model.control_variate(input)
                terminal = torch.exp(torch.tensor([-T*r], device=device))*g(S_T).view(-1,1)
                terminal_list.append(terminal)
                control_variate_list.append(control_variate)
            terminal = torch.cat(terminal_list, 0)
            control_variate = torch.cat(control_variate_list, 0)
        else:
            input = torch.ones(batch_size_MC, dim, device=device)
            with torch.no_grad():
                S_T, control_variate = model.control_variate(input)
            terminal = torch.exp(torch.tensor([-T*r], device=device))*g(S_T).view(-1,1)
    MC_estimator = torch.mean(terminal)
    var_terminal = torch.mean((terminal - torch.mean(terminal))**2)
    var_MC_estimator = 1/batch_size_MC*var_terminal
    
    cov_terminal_control_variate = torch.mean((terminal-torch.mean(terminal))*(control_variate-torch.mean(control_variate)))
    var_control_variate = torch.mean((control_variate-torch.mean(control_variate))**2)
    corr_terminal_control_variate = cov_terminal_control_variate/(torch.sqrt(var_control_variate)*torch.sqrt(var_terminal))
    
    # ratio of variance of the optimally controlled estimator to that of the uncontrolled estimator
    1-corr_terminal_control_variate.item()**2
    
    # Optimal coefficent b that minimises variance of optimally controlled estimator   
    b = cov_terminal_control_variate / var_control_variate
    
    
    # Monte Carlo controlled iterations
    MC_CV = terminal - b*(control_variate)
    
    # Monte Carlo controlled estimator
    MC_CV_estimator = torch.mean(MC_CV)
    
    # formula 4.2 of Monte Carlo Methods in Financial Engineering to get the variance of the Monte Carlo controlled estimator
    var_MC_CV_estimator = var_terminal - 2*b*torch.sqrt(var_control_variate)*torch.sqrt(var_terminal)*corr_terminal_control_variate + b**2*var_control_variate
    var_MC_CV_estimator = 1/batch_size_MC * var_MC_CV_estimator 
    
    return var_MC_CV_estimator.item(), var_MC_estimator.item(), MC_CV_estimator.item(), MC_estimator.item(), corr_terminal_control_variate.item()
   

# we initialise weights using Xavier initialisation
def weight_initialise(m):
    if isinstance(m, nn.Linear):
        gain=nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(m.weight, gain)
        #nn.init.xavier_uniform_(m.bias, gain)

if __name__ == '__main__':
    # we read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--vNetWidth', action="store", type=int, default=22, help="network width")
    parser.add_argument('--n-layers', action="store", type=int, default=2, help="number of layers")
    parser.add_argument('--timestep', action="store", type=float, default=0.01, help="timestep")
    parser.add_argument('--dim', action="store", type=int, default=2, help="dimension of the PDE")

    args = parser.parse_args()
    vNetWidth = args.vNetWidth
    n_layers = args.n_layers
    timestep = args.timestep
    dim = args.dim

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    PATH_RESULTS = os.getcwd()
    print(os.getcwd())
    log_results = 100
    log_evaluate = 10

    file_log_path = os.path.join(PATH_RESULTS, 'log_2nets_'+str(dim)+'_'+str(n_layers)+'_'+str(vNetWidth)+'.txt')
    file_log_results = os.path.join(PATH_RESULTS, 'results_2nets_'+str(dim)+'_'+str(n_layers)+'_'+str(vNetWidth)+'.txt')
    file_evaluation = os.path.join(PATH_RESULTS, 'evaluation_2nets_'+str(dim)+'_'+str(n_layers)+'_'+str(vNetWidth)+'.txt')
    with open(file_log_results, 'a') as f:
        f.write('var_MC_CV_estimator, var_MC_estimator, MC_CV_estimator, MC_estimator,corr_terminal_control_variate\n')


    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    
    ##################
    # Problem setup ##
    ##################
    init_t, T = 0,0.5
    #timestep = 0.01
    timegrid = np.arange(init_t, T+timestep/2, timestep)
    #dim = 2
    r = 0.05
    sigma = 0.3
    mu = 0.08
    tau = 0.1
    covariance_mat = np.identity(dim)    
    
    #########################
    # Network instantiation #
    #########################
    model = BSDE_solver(dim=dim, r=r, sigma=np.array([sigma]*dim), covariance_mat=covariance_mat, timegrid=timegrid, n_layers=n_layers, vNetWidth=vNetWidth, gradNetWidth=vNetWidth)      
    model.apply(weight_initialise)
    model.to(device)
    
    ##################################
    # Network weights initialisation #
    ##################################
    model.apply(weight_initialise)
    
    #######################
    # training parameters #
    #######################
    batch_size = 5000
    base_lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(),lr=base_lr, betas=(0.9, 0.999))
    n_iter = 20000
    train()
