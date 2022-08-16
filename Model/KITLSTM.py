#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:27:52 2021

@author: lucasliu
"""

import math
import torch
import torch.nn as nn

#only time gate    
class KG_TIMEAWARE_LSTM1_WithStaticF_HtConcat(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int): 
        super(KG_TIMEAWARE_LSTM1_WithStaticF_HtConcat, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t

        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t
        
        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1)) #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t
                
        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 

        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method
        
        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x,static_x, initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states

        #4. for each time step, compute each gate value for each sample
        h_t_list = []
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)  #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1) #[n_feature x batch_size]

            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (delta_t) + self.b_g) #[n_hidden x batch_size] 
            #discounted short-term memory, 
            cs_hat_t = cs_t * g_t        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            #adjusted previous memory
            c_star = cl + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi) #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf) #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho) #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)              #(n_hidden x batch_size)
            
            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
            
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y 

#Add second delta T
#only time gate    
class KG_TIMEAWARE_LSTM1_WithStaticF_HtConcat_2ndDelta_t(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int): 
        super(KG_TIMEAWARE_LSTM1_WithStaticF_HtConcat_2ndDelta_t, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t

        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t
        
        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1)) #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t
                
        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 

        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method
        
        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x,static_x, delta2_x, initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states

        #4. for each time step, compute each gate value for each sample
        h_t_list = []
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)  #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1) #[n_feature x batch_size]
            delta_t2 = torch.transpose(delta2_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t2 = torch.cat(self.hidden_size*[delta_t2],0)     #[n_hidden x batch_size] #repeat for each dim

            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (delta_t) + self.b_g) #[n_hidden x batch_size] 
            #discounted short-term memory, 
            cs_hat_t = cs_t * g_t        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            #discounted long-term memory
            cl_ds = cl * torch.sigmoid(1/delta_t2) # use sigmoid to avoid inf

            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi) #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf) #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho) #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)              #(n_hidden x batch_size)
            
            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
            
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y     

#Add third delta T and adjust different part of memory
#Add second delta T
#only time gate    
class KG_TIMEAWARE_LSTM1_WithStaticF_HtConcat_3rdDelta_t(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int): 
        super(KG_TIMEAWARE_LSTM1_WithStaticF_HtConcat_3rdDelta_t, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t

        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t
        
        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1)) #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t
                
        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 

        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method
        
        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x,static_x, delta3_x, initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states

        #4. for each time step, compute each gate value for each sample
        h_t_list = []
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)  #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1) #[n_feature x batch_size]
            delta_t3 = torch.transpose(delta3_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t3 = torch.cat(self.hidden_size*[delta_t3],0)     #[n_hidden x batch_size] #repeat for each dim

            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN            
            g_t =  torch.sigmoid(self.W_g @ (1/torch.sigmoid(delta_t)) + self.b_g) #[n_hidden x batch_size] 
            g_t2 =  torch.sigmoid(1/delta_t3) # use sigmoid to avoid inf #[n_hidden x batch_size] 
            
            #discounted short-term memory, 
            cs_hat_t = cs_t * g_t2        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            
            #discounted long-term memory
            cl_ds = cl * g_t               

            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi) #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf) #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho) #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)              #(n_hidden x batch_size)
            
            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
            
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y   
    
#This model contains the components: time gate, KG Binary feature, and with static feature , #No KG attention
#This model concat static feature to the concated hidden state ht

class KG_TIMEAWARE_LSTM2_WithStaticF_HtConcat(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int, ontology_size: int): 
        super(KG_TIMEAWARE_LSTM2_WithStaticF_HtConcat, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 

        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method
        
        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x, initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states
            
        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)      #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)         #[n_feature x batch_size]

            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (delta_t) + self.b_g) #[n_hidden x batch_size] 
            #discounted short-term memory
            cs_hat_t = cs_t * g_t        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            #adjusted previous memory
            c_star = cl + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t + KG_candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)
            
            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all  = torch.cat((h_t_all,static_x),0)
       
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
            
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y

#Add 2nd delta T
#This model contains the components: time gate, KG Binary feature, and with static feature , #No KG attention
#This model concat static feature to the concated hidden state ht
class KG_TIMEAWARE_LSTM2_WithStaticF_HtConcat_2ndDelta_t(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int, ontology_size: int): 
        super(KG_TIMEAWARE_LSTM2_WithStaticF_HtConcat_2ndDelta_t, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 

        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method
        
        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x,delta2_x, initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states
            
        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)      #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)         #[n_feature x batch_size]
            delta_t2 = torch.transpose(delta2_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t2 = torch.cat(self.hidden_size*[delta_t2],0)     #[n_hidden x batch_size] #repeat for each dim

            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (delta_t) + self.b_g) #[n_hidden x batch_size] 
            #discounted short-term memory
            cs_hat_t = cs_t * g_t        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]

            #discounted long-term memory
            cl_ds = cl * torch.sigmoid(1/delta_t2) # use sigmoid to avoid inf 
           
            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t + KG_candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)
            
            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all  = torch.cat((h_t_all,static_x),0)
       
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
            
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y

    
#This model concat static feature to the concated hidden state ht
class KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int,ontology_size: int, transE_D_size : int): 
        super(KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        
        #Attenton parameters
        self.W_a = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.W_b = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.W_r = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.softmax =  nn.Softmax(dim=0)
        self.W_k = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG Attention
        self.b_k = nn.Parameter(torch.Tensor(hidden_size, 1)) #KG Attention

        
        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method

        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x, ont_ent_emb,ont_target_emb,ont_rel_emb,initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states

        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)      #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)         #[n_ontology x batch_size] 
            
            #available entity embedding at time t
            aval_e_t = ont_ent_emb @ p_t                         #[n_transE_Embedding x batch_size]
            aval_eTarget_t = torch.cat(bs*[ont_target_emb],1)    #[n_transE_Embedding x batch_size]
            aval_r_t = ont_rel_emb                               #[n_transE_Embedding x n_ontology]
            
            #Ontology Attention 
            beta_t1 = torch.transpose(self.W_r @ aval_r_t,0,1)               #[1 x n_hidden_states]
            beta_t2 = torch.tanh(self.W_a @ aval_e_t + self.W_b @ aval_eTarget_t)
            beta_t = beta_t1 @ beta_t2
            aval_beta_t = beta_t * p_t
            a_t = self.softmax(aval_beta_t)
            
            #KG-Gate
            k_t =  torch.sigmoid(self.W_k @ (a_t) + self.b_k)         #[n_hidden x batch_size] 

            
            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (delta_t) + self.b_g) #[n_hidden x batch_size] 
            #discounted short-term memory
            cs_hat_t = cs_t * g_t        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            #adjusted previous memory
            c_star = cl + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t + k_t * KG_candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)

            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
        
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y 

#This model Add 2nd delta T and concat static feature to the concated hidden state ht
class KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_2ndDelta_t(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int,ontology_size: int, transE_D_size : int): 
        super(KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_2ndDelta_t, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        
        #Attenton parameters
        self.W_a = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.W_b = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.W_r = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.softmax =  nn.Softmax(dim=0)
        self.W_k = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG Attention
        self.b_k = nn.Parameter(torch.Tensor(hidden_size, 1)) #KG Attention

        
        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method

        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x, delta2_x,ont_ent_emb,ont_target_emb,ont_rel_emb,initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states

        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                     #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)           #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)              #[n_ontology x batch_size] 
            delta_t2 = torch.transpose(delta2_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t2 = torch.cat(self.hidden_size*[delta_t2],0)     #[n_hidden x batch_size] #repeat for each dim

            #available entity embedding at time t
            aval_e_t = ont_ent_emb @ p_t                         #[n_transE_Embedding x batch_size]
            aval_eTarget_t = torch.cat(bs*[ont_target_emb],1)    #[n_transE_Embedding x batch_size]
            aval_r_t = ont_rel_emb                               #[n_transE_Embedding x n_ontology]
            
            #Ontology Attention 
            beta_t1 = torch.transpose(self.W_r @ aval_r_t,0,1)               #[1 x n_hidden_states]
            beta_t2 = torch.tanh(self.W_a @ aval_e_t + self.W_b @ aval_eTarget_t)
            beta_t = beta_t1 @ beta_t2
            aval_beta_t = beta_t * p_t
            a_t = self.softmax(aval_beta_t)
            
            #KG-Gate
            k_t =  torch.sigmoid(self.W_k @ (a_t) + self.b_k)         #[n_hidden x batch_size] 

            
            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (delta_t) + self.b_g) #[n_hidden x batch_size] 
            #discounted short-term memory
            cs_hat_t = cs_t * g_t        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            #discounted long-term memory
            cl_ds = cl * torch.sigmoid(1/delta_t2) # use sigmoid to avoid inf

            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t + k_t * KG_candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)

            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
        
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y 
    
#Adjust differnet cell memory
#Add 3rd delta T (time since last visit)
#This model contains the components: time gate, KG Binary feature, and with static feature , #No KG attention
#This model concat static feature to the concated hidden state ht
class KG_TIMEAWARE_LSTM2_WithStaticF_HtConcat_3rdDelta_t(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int, ontology_size: int): 
        super(KG_TIMEAWARE_LSTM2_WithStaticF_HtConcat_3rdDelta_t, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 

        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method
        
        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x, delta3_x, initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states
            
        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)      #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)         #[n_feature x batch_size]
            delta_t3 = torch.transpose(delta3_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t3 = torch.cat(self.hidden_size*[delta_t3],0)     #[n_hidden x batch_size] #repeat for each dim

            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t  =  torch.sigmoid(self.W_g @ (1/torch.sigmoid(delta_t)) + self.b_g) #[n_hidden x batch_size] 
            g_t2 =  torch.sigmoid(1/delta_t3) # use sigmoid to avoid inf #[n_hidden x batch_size] 
            
            #discounted short-term memory
            cs_hat_t = cs_t * g_t2        #[n_hidden x batch_size]

            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]

            #discounted long-term memory
            cl_ds = cl * g_t # use delta t 1 to adjust long term monery
           
            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]

            #c_t = f_t * c_star + i_t * KG_candidate_t   #(n_hidden x batch_size)
            c_t = f_t * c_star + i_t * candidate_t + KG_candidate_t  #(n_hidden x batch_size)

            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)
            
            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all  = torch.cat((h_t_all,static_x),0)
       
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
            
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y

#Adjust cell memory LSTM3 and embedding dist and concept embedding
#Add 3rd delta T (time since last visit)
#This model concat static feature to the concated hidden state ht
class KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_3rdDelta_t(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int,ontology_size: int, transE_D_size : int): 
        super(KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_3rdDelta_t, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        
        #Ontology feature mapping 
        self.W_ont = nn.Parameter(torch.Tensor(1, transE_D_size))  
        self.b_ont = nn.Parameter(torch.Tensor(1, ontology_size))  
                
        #Attenton parameters
        self.W_a = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.W_b = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.W_r = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.softmax =  nn.Softmax(dim=0)
        self.W_k = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG Attention
        self.b_k = nn.Parameter(torch.Tensor(hidden_size, 1)) #KG Attention

        
        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method

        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x, delta3_x, concept_dist, ont_ent_emb,ont_target_emb,ont_rel_emb,initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states

        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                     #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)           #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)              #[n_ontology x batch_size] 
            delta_t3 = torch.transpose(delta3_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t3 = torch.cat(self.hidden_size*[delta_t3],0)     #[n_hidden x batch_size] #repeat for each dim
            
            # #Map rel embedding from n_transE_Embedding to dim = 1
            # ont_rel_emb_mapped = torch.sigmoid(self.W_ont @ ont_rel_emb  + self.b_ont) #[1 x n_ontology]
            # ont_rel_emb_mapped_all = torch.transpose(torch.cat(bs*[ont_rel_emb_mapped],0),0,1)               #[n_ontology x batch_size] #repeat for each pt

            #Mal ontology embedding from n_transE_Embedding to dim = 1
            ont_ent_emb_mapped = torch.sigmoid(self.W_ont @ ont_ent_emb  + self.b_ont)                #[1 x n_ontology]
            ont_ent_emb_mapped_all = torch.transpose(torch.cat(bs*[ont_ent_emb_mapped],0),0,1)        #[n_ontology x batch_size] #repeat for each pt
                          
            #Use concept distantance as relation
            ont_rel_emb_mapped_all = torch.transpose(torch.cat(bs*[concept_dist],0),0,1) #[n_ontology x batch_size] #repeat for each pt
            
            #Available relation at time t
            beta_t = ont_rel_emb_mapped_all * p_t  +   ont_ent_emb_mapped_all * p_t                #[n_ontology x batch_size]
            

            
            a_t = self.softmax(beta_t)                                    #[n_ontology x batch_size] 
        #     #available entity embedding at time t
        #     aval_eTarget_t = torch.cat(bs*[ont_target_emb],1)    #[n_transE_Embedding x batch_size]
        #     aval_r_t = ont_rel_emb                               #[n_transE_Embedding x n_ontology]
            
        #     #Ontology Attention 
        #     beta_t1 = torch.transpose(self.W_r @ aval_r_t,0,1)               #[1 x n_hidden_states]
        #     beta_t2 = torch.tanh(self.W_a @ aval_e_t + self.W_b @ aval_eTarget_t)
        #     beta_t = beta_t1 @ beta_t2
        #     aval_beta_t = beta_t * p_t
        #     a_t = self.softmax(aval_beta_t) #[n_ontology x batch_size] 
            
            #KG-Gate
            k_t =  torch.sigmoid(self.W_k @ (a_t) + self.b_k)         #[n_hidden x batch_size] 

            
            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (delta_t) + self.b_g) #[n_hidden x batch_size] 
            g_t2 =  torch.sigmoid(1/delta_t3) # use sigmoid to avoid inf #[n_hidden x batch_size] 

            #discounted short-term memory
            cs_hat_t = cs_t * g_t2        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            
            #discounted long-term memory
            cl_ds = cl * g_t # use delta t 1 to adjust long term monery

            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t + k_t * KG_candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)

            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
        
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y
    
    
#Adjust cell memory LSTM3 and
# only relation embedding  for attention
#OR use concept dist only  for attention 
#Or use concept embedding only for attention 
#Add 3rd delta T (time since last visit)
#This model concat static feature to the concated hidden state ht
class KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_3rdDelta_t_V2(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int,ontology_size: int, transE_D_size : int): 
        super(KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_3rdDelta_t_V2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        
        #Ontology feature mapping 
        self.W_ont = nn.Parameter(torch.Tensor(1, transE_D_size))  
        self.b_ont = nn.Parameter(torch.Tensor(1, ontology_size))  
                
        #Attenton parameters
        self.W_a = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.W_b = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.W_r = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.softmax =  nn.Softmax(dim=0)
        self.W_k = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG Attention
        self.b_k = nn.Parameter(torch.Tensor(hidden_size, 1)) #KG Attention

        
        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method

        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x, delta3_x, concept_dist, ont_ent_emb,ont_target_emb,ont_rel_emb,initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states

        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                     #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)           #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)              #[n_ontology x batch_size] 
            delta_t3 = torch.transpose(delta3_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t3 = torch.cat(self.hidden_size*[delta_t3],0)     #[n_hidden x batch_size] #repeat for each dim
            
            #Map rel embedding from n_transE_Embedding to dim = 1
            #ont_rel_emb_mapped = torch.sigmoid(self.W_ont @ ont_rel_emb  + self.b_ont) #[1 x n_ontology]
            #ont_rel_emb_mapped_all = torch.transpose(torch.cat(bs*[ont_rel_emb_mapped],0),0,1)               #[n_ontology x batch_size] #repeat for each pt

            #Mal ontology embedding from n_transE_Embedding to dim = 1
            ont_ent_emb_mapped = torch.sigmoid(self.W_ont @ ont_ent_emb  + self.b_ont)                #[1 x n_ontology]
            ont_ent_emb_mapped_all = torch.transpose(torch.cat(bs*[ont_ent_emb_mapped],0),0,1)        #[n_ontology x batch_size] #repeat for each pt
                          
            #Use concept distantance as relation
            ont_rel_emb_mapped_all = torch.transpose(torch.cat(bs*[concept_dist],0),0,1) #[n_ontology x batch_size] #repeat for each pt
            
            #Available relation at time t
            #beta_t = ont_rel_emb_mapped_all * p_t                 #[n_ontology x batch_size]
            beta_t = ont_rel_emb_mapped_all * p_t + ont_ent_emb_mapped_all * p_t              #[n_ontology x batch_size]


            
            a_t = self.softmax(beta_t)                                    #[n_ontology x batch_size] 
        #     #available entity embedding at time t
        #     aval_eTarget_t = torch.cat(bs*[ont_target_emb],1)    #[n_transE_Embedding x batch_size]
        #     aval_r_t = ont_rel_emb                               #[n_transE_Embedding x n_ontology]
            
        #     #Ontology Attention 
        #     beta_t1 = torch.transpose(self.W_r @ aval_r_t,0,1)               #[1 x n_hidden_states]
        #     beta_t2 = torch.tanh(self.W_a @ aval_e_t + self.W_b @ aval_eTarget_t)
        #     beta_t = beta_t1 @ beta_t2
        #     aval_beta_t = beta_t * p_t
        #     a_t = self.softmax(aval_beta_t) #[n_ontology x batch_size] 
            
            #KG-Gate
            k_t =  torch.sigmoid(self.W_k @ (a_t) + self.b_k)         #[n_hidden x batch_size] 

            
            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (1/torch.sigmoid(delta_t)) + self.b_g) #[n_hidden x batch_size] 
            g_t2 =  torch.sigmoid(1/delta_t3) # use sigmoid to avoid inf #[n_hidden x batch_size] 

            #discounted short-term memory
            cs_hat_t = cs_t * g_t2        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            
            #discounted long-term memory
            cl_ds = cl * g_t # use delta t 1 to adjust long term monery

            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]
            #KG_candidate_t = torch.tanh(self.W_pc @ p_t + self.b_pc + self.W_hc @ h_t + self.b_hc)    #[n_hidden x batch_size]

            c_t = f_t * c_star + i_t * candidate_t + k_t * KG_candidate_t  #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)

            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
        
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y

#Attention directly on physiological feature
#Adjust cell memory LSTM3 and
# only relation embedding  for attention
#OR use concept dist only  for attention 
#Or use concept embedding only for attention 
#Add 3rd delta T (time since last visit)
#This model concat static feature to the concated hidden state ht
class KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_3rdDelta_t_V3(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int,ontology_size: int, transE_D_size : int, dropout_v: float): 
        super(KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_3rdDelta_t_V3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        
        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 

        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        
        #Ontology feature mapping 
        #self.W_ont = nn.Parameter(torch.Tensor(1, transE_D_size))  
        #self.b_ont = nn.Parameter(torch.Tensor(1, ontology_size))  
                
        #Attenton parameters
        #self.W_a = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        #self.W_b = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        #self.W_r = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.softmax =  nn.Softmax(dim=0)
        self.W_k = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG Attention
        self.b_k = nn.Parameter(torch.Tensor(hidden_size, 1)) #KG Attention

        #Drop out layer
        self.dropout = nn.Dropout(dropout_v) #0.25
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        #Embedding layer for concept
        self.embedding = nn.Embedding(ontology_size + 1, 1) #n_num = n_ontoloty + n_target, output dim = 1
        
        #embedding layer for relation
        #self.embedding2 = nn.Embedding(ontology_size, 1) #n_ontoloty, output dim =1

        self.initial_weights() #Call initial weight method

        #softmax for dist norm
        self.norm = nn.Softmax(dim=0)
                
        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x, delta3_x, concept_dist, ont_ent_emb,ont_target_emb,ont_rel_emb,initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states
        
        #3.inital concept embeddings
        ont_concepts = torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) #input index of 20 ontology and 1 target         
        #concetpt embeddings
        ont_emb = self.embedding(ont_concepts)
        #Other concepts
        ont_ent_emb  = ont_emb[0:(len(ont_emb)-1)] #[n_ontology x 1] 

        #Target concept
        ont_target_emb = ont_emb[-1]              #[1 x 1] 

        #4.Relatino embeddings
        #rel_concepts = torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])    #input index of 20 ontology relations to the target
        #ont_rel_emb = self.embedding2(rel_concepts) #[n_ontology x 1] 

        #5.conpute distantce between other embs to target
        ont_dist = torch.sqrt((ont_ent_emb - ont_target_emb)**2)      #[n_ontology x 1] 
        ont_dist_normed = self.norm(ont_dist)
        
        #conpute distance between concept embed + relation embed and target embed
        #ont_dist = torch.sqrt((ont_target_emb - (ont_ent_emb + rel_emb))**2)
        #ont_dist_normed = self.norm(ont_dist)

        #repeat embeddings for every pts
        ont_ent_emb = torch.cat(bs*[ont_ent_emb],1)          #[n_ontology x batch_size] 
        #ont_rel_emb = torch.cat(bs*[ont_rel_emb],1)          #[n_ontology x batch_size] 

        


        h_t_list = []
        a_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                     #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)           #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)              #[n_ontology x batch_size] 
            delta_t3 = torch.transpose(delta3_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t3 = torch.cat(self.hidden_size*[delta_t3],0)     #[n_hidden x batch_size] #repeat for each dim
           
            #Use concept distantance as relation
            #ont_rel_emb = torch.transpose(torch.cat(bs*[concept_dist],0),0,1) #[n_ontology x batch_size] #repeat for each pt
            
            #Available relation at time t
            #beta_t = ont_rel_emb * p_t                                 #[n_ontology x batch_size]
            #beta_t = ont_rel_emb * p_t + ont_ent_emb * p_t              #[n_ontology x batch_size]
            beta_t = ont_ent_emb * p_t

            
            a_t = self.softmax(beta_t)                                    #[n_ontology x batch_size] 
            
            #KG-Gate
            k_t =  torch.sigmoid(self.W_k @ (a_t) + self.b_k)         #[n_hidden x batch_size] 

            
            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (1/torch.sigmoid(delta_t)) + self.b_g) #[n_hidden x batch_size] 
            g_t2 =  torch.sigmoid(1/delta_t3) # use sigmoid to avoid inf #[n_hidden x batch_size] 

            #discounted short-term memory
            cs_hat_t = cs_t * g_t2        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            
            #discounted long-term memory
            cl_ds = cl * g_t # use delta t 1 to adjust long term monery

            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]
            #KG_candidate_t = torch.tanh(self.W_pc @ p_t + self.b_pc + self.W_hc @ h_t + self.b_hc)    #[n_hidden x batch_size]

            #c_t = f_t * c_star + k_t * i_t * candidate_t   #(n_hidden x batch_size)
            c_t = f_t * c_star + i_t * candidate_t +  k_t * KG_candidate_t   #(n_hidden x batch_size)
            #c_t = f_t * c_star + i_t * candidate_t + i_t * k_t    #current best #(n_hidden x batch_size) 

            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)

            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
            a_t_list.append(a_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        a_t_all = torch.cat(a_t_list, 1)  #concatenate all h_t (n_ontology x n_time x batch_size)

        #directly sum over time steps
        h_t_all = torch.sum(h_t_all, dim=1) #Sum over all time steps for each hidden (n_hidden x batch_size)
        #h_t_all = torch.max(h_t_all, dim=1)[0] #MAx over all time steps for each hidden (n_hidden x batch_size)

        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
        
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y, ont_dist_normed,a_t_all
    
#Add time attention
class KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_3rdDelta_t_V3_ADDTimeATTENTION(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int,ontology_size: int, transE_D_size : int, dropout_v: float): 
        super(KG_TIMEAWARE_LSTM3_WithStaticF_HtConcat_3rdDelta_t_V3_ADDTimeATTENTION, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        
        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        #Weights for time-aware gates
        self.W_d = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  #y: [n_hidden x n_hidden] 
        self.b_d = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 
        
        self.W_g = nn.Parameter(torch.Tensor(hidden_size, input_size))  #y: [n_hidden x input_size] 
        self.b_g = nn.Parameter(torch.Tensor(hidden_size, 1))            #y: [n_hidden x 1] 

        #weights for KG candidate memory
        self.W_ic = nn.Parameter(torch.Tensor(hidden_size, input_size)) #KG candidate memory 
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG candidate memory 

        self.b_ic = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        
        #Ontology feature mapping 
        #self.W_ont = nn.Parameter(torch.Tensor(1, transE_D_size))  
        #self.b_ont = nn.Parameter(torch.Tensor(1, ontology_size))  
                
        #Attenton parameters
        #self.W_a = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        #self.W_b = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        #self.W_r = nn.Parameter(torch.Tensor(hidden_size, transE_D_size)) #KG Attention
        self.softmax =  nn.Softmax(dim=0)
        self.W_k = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #KG Attention
        self.b_k = nn.Parameter(torch.Tensor(hidden_size, 1)) #KG Attention

        '''Added 011722 Time Attention paramteres
        '''
        self.Linear_a = nn.Linear(in_features = hidden_size, out_features= 1)
        self.softmax_a = nn.Softmax(dim = 1) #in time dimenstion

        #Drop out layer
        self.dropout = nn.Dropout(dropout_v) #0.25
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        #Embedding layer for concept
        self.embedding = nn.Embedding(ontology_size + 1, 1) #n_num = n_ontoloty + n_target, output dim = 1
        
        #embedding layer for relation
        #self.embedding2 = nn.Embedding(ontology_size, 1) #n_ontoloty, output dim =1

        self.initial_weights() #Call initial weight method

        #softmax for dist norm
        self.norm = nn.Softmax(dim=0)
                
        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, delta_x, binary_x, static_x, delta3_x, concept_dist, ont_ent_emb,ont_target_emb,ont_rel_emb,initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states
        
        #3.inital concept embeddings
        ont_concepts = torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]) #input index of 20 ontology and 1 target         
        #concetpt embeddings
        ont_emb = self.embedding(ont_concepts)
        #Other concepts
        ont_ent_emb  = ont_emb[0:(len(ont_emb)-1)] #[n_ontology x 1] 

        #Target concept
        ont_target_emb = ont_emb[-1]              #[1 x 1] 

        #4.Relatino embeddings
        #rel_concepts = torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])    #input index of 20 ontology relations to the target
        #ont_rel_emb = self.embedding2(rel_concepts) #[n_ontology x 1] 

        #5.conpute distantce between other embs to target
        ont_dist = torch.sqrt((ont_ent_emb - ont_target_emb)**2)      #[n_ontology x 1] 
        ont_dist_normed = self.norm(ont_dist)
        
        #conpute distance between concept embed + relation embed and target embed
        #ont_dist = torch.sqrt((ont_target_emb - (ont_ent_emb + rel_emb))**2)
        #ont_dist_normed = self.norm(ont_dist)

        #repeat embeddings for every pts
        ont_ent_emb = torch.cat(bs*[ont_ent_emb],1)          #[n_ontology x batch_size] 
        #ont_rel_emb = torch.cat(bs*[ont_rel_emb],1)          #[n_ontology x batch_size] 

        


        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                     #[n_feature x batch_size]
            delta_t = torch.transpose(delta_x[:,t,:],0,1)           #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)              #[n_ontology x batch_size] 
            delta_t3 = torch.transpose(delta3_x[:,t,:],0,1)         #[1 x batch_size] 
            delta_t3 = torch.cat(self.hidden_size*[delta_t3],0)     #[n_hidden x batch_size] #repeat for each dim
           
            #Use concept distantance as relation
            #ont_rel_emb = torch.transpose(torch.cat(bs*[concept_dist],0),0,1) #[n_ontology x batch_size] #repeat for each pt
            
            #Available relation at time t
            #beta_t = ont_rel_emb * p_t                                 #[n_ontology x batch_size]
            #beta_t = ont_rel_emb * p_t + ont_ent_emb * p_t              #[n_ontology x batch_size]
            beta_t = ont_ent_emb * p_t

            
            a_t = self.softmax(beta_t)                                    #[n_ontology x batch_size] 
            
            #KG-Gate
            k_t =  torch.sigmoid(self.W_k @ (a_t) + self.b_k)         #[n_hidden x batch_size] 

            
            #short-term memory
            cs_t = torch.tanh(self.W_d @ c_t + self.b_d) #[n_hidden x batch_size] 
            #Time aware gate
            #NOTE: if use 1/delta_t, the model predicts NAN
            g_t =  torch.sigmoid(self.W_g @ (1/torch.sigmoid(delta_t)) + self.b_g) #[n_hidden x batch_size] 
            g_t2 =  torch.sigmoid(1/delta_t3) # use sigmoid to avoid inf #[n_hidden x batch_size] 

            #discounted short-term memory
            cs_hat_t = cs_t * g_t2        #[n_hidden x batch_size]
            #long-term memory
            cl = c_t - cs_t                #[n_hidden x batch_size]
            
            #discounted long-term memory
            cl_ds = cl * g_t # use delta t 1 to adjust long term monery

            #adjusted previous memory
            c_star = cl_ds + cs_hat_t        #[n_hidden x batch_size]
            

            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_ic @ x_t + self.b_ic + self.W_hc @ h_t + self.b_hc + self.W_pc @ p_t + self.b_pc)    #[n_hidden x batch_size]
            #KG_candidate_t = torch.tanh(self.W_pc @ p_t + self.b_pc + self.W_hc @ h_t + self.b_hc)    #[n_hidden x batch_size]

            #c_t = f_t * c_star + k_t * i_t * candidate_t   #(n_hidden x batch_size)
            c_t = f_t * c_star + i_t * candidate_t +  k_t * KG_candidate_t   #(n_hidden x batch_size)
            #c_t = f_t * c_star + i_t * candidate_t + i_t * k_t    #current best #(n_hidden x batch_size) 

            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)

            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        
        '''Added 011722 Time Attention 
        '''
        h_t_all_trans = h_t_all.transpose(0, 2).contiguous()  #(batch_size x n_time x n_hidden)
        e = self.Linear_a(h_t_all_trans)                      #(batch_size x n_time x 1)
        a = self.softmax_a(e)                                 #(batch_size x n_time x 1)
        a = a.transpose(0, 2).contiguous()                    #(1 x n_time x batch_size)
        h_t_all_weighted =  a*h_t_all                         #(n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all_weighted, dim=1) #Sum over all time steps for each hidden (n_hidden x batch_size)
         
        #Previous version directly sum over time steps
        #h_t_all = torch.sum(h_t_all, dim=1) #Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
        
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y, ont_dist_normed
    

#Only binary feature as part of candidate memory Or add KG candidate memory, no time gate for ajusting memory, no attention
#This model concat static feature to the concated hidden state ht
class KG_TIMEAWARE_LSTM4_WithStaticF_HtConcat(nn.Module):          
    def __init__(self, input_size: int, hidden_size: int, static_size: int, ontology_size: int): 
        super(KG_TIMEAWARE_LSTM4_WithStaticF_HtConcat, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #In pytroch nn.LSTM,
        #the order of initial weight matrix is W_ii, W_if, W_ig,W_hg (Sothe random generated value will be in this order)
        #Then W_hi,W_hf
        #W_ih
        self.W_ii = nn.Parameter(torch.Tensor(hidden_size, input_size)) #i_t: [n_hidden x n_feature]
        self.W_if = nn.Parameter(torch.Tensor(hidden_size, input_size)) #f_t
        self.W_ig = nn.Parameter(torch.Tensor(hidden_size, input_size)) #g_t
        self.W_io = nn.Parameter(torch.Tensor(hidden_size, input_size)) #o_t


        #W_hh 
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #i_t: [n_hidden x n_hidden]
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #f_t
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #g_t
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #o_t
        

        #b_ih 
        self.b_ii = nn.Parameter(torch.Tensor(hidden_size , 1))  #i_t: [n_hidden x 1] 
        self.b_if  = nn.Parameter(torch.Tensor(hidden_size, 1)) #f_t
        self.b_ig = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_io = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #b_hh
        self.b_hi = nn.Parameter(torch.Tensor(hidden_size, 1))  #i_t: [n_hidden x 1] 
        self.b_hf  = nn.Parameter(torch.Tensor(hidden_size,1))  #f_t
        self.b_hg = nn.Parameter(torch.Tensor(hidden_size, 1))  #g_t
        self.b_ho = nn.Parameter(torch.Tensor(hidden_size, 1))  #o_t

        #Weights for prediction layer
        self.W_y = nn.Parameter(torch.Tensor(1, hidden_size + static_size))  #y: [1 x n_hidden + static_size] 
        self.b_y = nn.Parameter(torch.Tensor(1, 1))            #y: [1 x 1] 
        
        
        #weights for KG candidate memory
        self.W_hc = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) #KG candidate memory 
        self.W_pc = nn.Parameter(torch.Tensor(hidden_size, ontology_size)) #binary feature
        
        self.b_hc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        self.b_pc = nn.Parameter(torch.Tensor(hidden_size, 1))  #KG candidate memory 
        
        #Drop out layer
        self.dropout = nn.Dropout(0.25)
        
        #Batch normalization
        self.bn1 = nn.BatchNorm1d(num_features=1)
        
        self.initial_weights() #Call initial weight method

        #add a bn layer
        #self.BN = nn.BatchNorm1d(hidden_size)
    def initial_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, binary_x, static_x ,initial_states = None, drop_out_flag = False):
        #1. Get batch_size, squence_length/time steps (Assume x in shape (batch_size, sequence length/ time_steps, n_features))
        bs, ts, _ = x.size() #batch size, time steps, _
        
        #2.Initial states
        if initial_states == None:
            h_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size], for each sample, the dimension is [n_hidden_states x 1]
            c_t = torch.ones(self.hidden_size, bs).to(x.device) #[n_hidden_states x batch_size]
        else :
            h_t, c_t = initial_states

        h_t_list = []
        #4. for each time step, compute each gate value for each sample
        for t in range(ts):
            x_t = torch.transpose(x[:,t,:],0,1)                     #[n_feature x batch_size]
            p_t = torch.transpose(binary_x[:,t,:],0,1)              #[n_ontology x batch_size] 
            
            
            #[n_hidden  x batch_size] +  [n_hidden x 1]  (b_ii is added for each sample)
            i_t = torch.sigmoid(self.W_ii @ x_t + self.b_ii + self.W_hi @ h_t + self.b_hi)         #[n_hidden x batch_size]
            f_t = torch.sigmoid(self.W_if @ x_t + self.b_if + self.W_hf @ h_t + self.b_hf)         #[n_hidden x batch_size]
            o_t = torch.sigmoid(self.W_io @ x_t + self.b_io + self.W_ho @ h_t + self.b_ho)         #[n_hidden x batch_size]
            candidate_t = torch.tanh(self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            KG_candidate_t = torch.tanh(self.W_pc @ p_t + self.b_pc + self.W_hc @ h_t + self.b_hc)    #[n_hidden x batch_size]
            
            #OR add pt as part of the candidate memory
            #candidate_t = torch.tanh(self.W_pc @ p_t + self.b_pc  + self.W_ig @ x_t + self.b_ig + self.W_hg @ h_t + self.b_hg)    #[n_hidden x batch_size]
            #c_t = f_t * c_t + i_t * candidate_t                     #(n_hidden x batch_size)

            c_t = f_t * c_t + i_t * candidate_t + KG_candidate_t     #(n_hidden x batch_size)
            h_t = o_t * torch.tanh(c_t)                              #(n_hidden x batch_size)

            h_t_list.append(h_t.unsqueeze(1))  #add a dim for timestep 
        
        h_t_all = torch.cat(h_t_list, 1)  #concatenate all h_t (n_hidden x n_time x batch_size)
        h_t_all = torch.sum(h_t_all, dim=1) #Pooling Sum over all time steps for each hidden (n_hidden x batch_size)
        
        #Concatenate static feature to the last h_t
        static_x     = static_x.transpose(0, 1).contiguous() #reshape to [N_STATIC_FEATURE, BATCH_SIZE]
        h_t_all      = torch.cat((h_t_all,static_x),0)
        
        out = self.W_y @ h_t_all + self.b_y
        out = out.transpose(0,1).contiguous() #reshape so that internal batch normalzation layer could work
        out = self.bn1(out)  #batch norm
        out = out.transpose(0,1).contiguous() #reshape back
        
        #Drop out 
        if drop_out_flag :
            out = self.dropout(out)
        
        #Prediction using the last hidden state h_t
        #(1 x n_hidden)      x (n_hidden x batch_size) + (1 x 1)
        y = torch.sigmoid(out)
         
        #reshape y
        y = y.transpose(0, 1).contiguous() 
        
        return y