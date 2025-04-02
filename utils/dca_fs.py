# Netork modules
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
from sklearn.svm import l1_min_c  # for L1 regluarization path
import dca_fs_tools as dcat
import copy



#==================#
#    Load Data     #
#==================#


class SynthDataset(Dataset):


    def __init__(self, df, dependent, independent_in):
        
        if independent_in == None:
            independent = list(df.drop([dependent], axis=1).columns)
        else:
            independent = independent_in


        # The target is binary
        self.target = [dependent]   #["flow", "handling", "respect"]
        self.features = independent
        self.data_frame = df.loc[:, self.target+self.features]
        self.n_features = len(self.features)

        # Save target and predictors
        self.x = self.data_frame[self.features]
        self.y = self.data_frame.loc[:,"y"]


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return  [self.x.iloc[idx].values.astype(float),  self.y.iloc[idx].astype(float)] #,  self.handling.iloc[idx].astype(float),  self.respect.iloc[idx].astype(float)]



# Create model
# f = wx + b, sigmoid at the end

class TorchLogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(TorchLogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        #y_predicted = torch.sigmoid(self.linear(x))
        y_predicted = self.linear(x)
        return y_predicted
    



class CrossEntropyWithLogit(nn.Module):
    def __init__(self):
        super(mnbLoss, self).__init__()
        #self.weight = weight

    def forward(self, input, target):
        # Compute the loss
        
        # Binary cross entropy with logits
        # this replicates BCEWithLogitsLoss
        #-----------------------------------
        A = torch.log(1.0 + torch.exp(-input))
        loss = target*A+(1.-target)*(input+A)
        loss = torch.mean(loss)



        return loss
    

class mnbLoss(nn.Module):
    def __init__(self):
        super(mnbLoss, self).__init__()
        #self.weight = weight

    def forward(self, input, target):
        # Compute the loss
        
        # Binary cross entropy with logits
        # this replicates BCEWithLogitsLoss
        #-----------------------------------
        A = torch.log(1.0 + torch.exp(-input))
        loss = (1.-target)*(input+A)-torch.sigmoid(input)
        loss = torch.mean(loss)

        return loss
    


#=======================#
#     LR Training       #
#=======================#


def lr_train(train_in, 
             n_epochs=100, 
             learn_rate = 0.01, 
             loss_fun = 'log',
             dependent = "y",
             independent = None):

    # Load dataset
    train_dataset_orig = SynthDataset(train_in, dependent, independent)

    train_x = torch.from_numpy(train_dataset_orig.x.to_numpy().astype(np.float32) )
    train_y = torch.from_numpy(train_dataset_orig.y.to_numpy().astype(np.float32))

    train_y = train_y.view(train_y.shape[0], 1)

    # Define the model
    net=TorchLogisticRegression(train_dataset_orig.n_features)
    
    # Loss function

    if loss_fun == 'log':
        #criterion = nn.BCELoss()
        criterion = nn.BCEWithLogitsLoss()
        #criterion = mnbLoss()
    elif loss_fun == 'mnb':
        criterion = mnbLoss()

    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)
    #optimizer = optim.Adam(net.parameters(),  lr = 0.001, weight_decay = 0.0001 )

    loss_per_epoch = []

    for epoch in range(n_epochs):

        y_pred = net(train_x)
        loss=criterion(y_pred, train_y)
        loss.backward()
        optimizer.step()

        # Zero the parameter gradients
        optimizer.zero_grad()
         
        loss_per_epoch.append(loss.item())

    #return net
    return {"net": net, "loss": loss_per_epoch}


#===================================#
#     Regularized LR Training       #
#===================================#

#tps://www.geeksforgeeks.org/l1l2-regularization-in-pytorch/

def reg_lr_train_legacy(train_in, 
                        net = 'none',
                        n_epochs=100, 
                        learn_rate = 0.01, 
                        loss_fun = 'log', 
                        regularization_type = 'L1', 
                        weights = [], 
                        lambda_reg = 0.01, 
                        dependent = "y",
                        independent = None):

    # Load dataset
    train_dataset_orig = SynthDataset(train_in, dependent, independent)

    train_x = torch.from_numpy(train_dataset_orig.x.to_numpy().astype(np.float32))
    train_y = torch.from_numpy(train_dataset_orig.y.to_numpy().astype(np.float32))
    train_y = train_y.view(train_y.shape[0], 1)

    # Define the model
    if net == 'none':
        net=TorchLogisticRegression(train_dataset_orig.n_features)
    
    # Loss function

    if loss_fun == 'log':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_fun == 'mnb':
        criterion = mnbLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(),  lr = 1.0)
 
    # A tensor of weights for the weighted L1
    if regularization_type == 'weighted_L1':
        weights = [
            torch.from_numpy(np.array([weights])),
            torch.from_numpy(np.array([1.0])),
        ]

    
    loss_per_epoch = []

    for epoch in range(n_epochs):

        y_pred = net(train_x )

        loss=criterion(y_pred, train_y )

        # add regularization term #
        #-------------------------#

        # Apply L1 regularization
        if regularization_type == 'L1':
            #l1_norm = sum(p.abs().sum() for p in net.parameters() )
            l1_norm = [p.abs().sum() for p in net.parameters()][0] 
            #l1_norm = [p.abs().sum() for p in net.parameters()][0] 
            #l1_norm = torch.mean(p.abs().sum() for p in net.parameters() )

            loss += lambda_reg * l1_norm
            
        # Apply weighted L1 regularization
        elif regularization_type == 'weighted_L1':
            #wl1_norm = sum(torch.mul(weights[p[0]], p[1]).abs().sum() for p in enumerate(net.parameters()))
            wl1_norm = [torch.mul(weights[p[0]], p[1]).abs().sum() for p in enumerate(net.parameters())][0]
            loss += lambda_reg * wl1_norm



        loss.backward()
        
        # Gradient cliping
        clipping_value = 1 # arbitrary value of your choosing
        torch.nn.utils.clip_grad_norm(net.parameters(), clipping_value)

        optimizer.step()
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        loss_per_epoch.append(loss.item())

    #return net
    return {"net": net, "loss": loss_per_epoch}




#===================================#
#     Regularized LR Training       #
#===================================#

#tps://www.geeksforgeeks.org/l1l2-regularization-in-pytorch/

def reg_lr_train(train_in, net = 'none',  
                 n_epochs=100, 
                 learn_rate = 1.0, 

                 history_size = 100, 
                 max_iter = 20, 
                 tolerance_grad=1e-07, 
                 tolerance_change=1e-09,


                 loss_fun = 'log', 
                 regularization_type = 'L1',    #'weighted_L1'
                 weights = [], 
                 lambda_reg = 0.01,
                 dependent = "y",
                 independent = None,
                 
                 grad_max_norm = 1.):

    # Load dataset
    train_dataset_orig = SynthDataset(train_in, dependent, independent)

    train_x = torch.from_numpy(train_dataset_orig.x.to_numpy().astype(np.float32))
    train_y = torch.from_numpy(train_dataset_orig.y.to_numpy().astype(np.float32))
    train_y = train_y.view(train_y.shape[0], 1)

    # Define the model

    if net == 'none':
        net=TorchLogisticRegression(train_dataset_orig.n_features)
    
    # Loss function
    if loss_fun == 'log':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_fun == 'mnb':
        criterion = mnbLoss()

    # Optimizer
    def closure():
        
        lbfgs.zero_grad()
        y_pred = net(train_x)
        loss=criterion(y_pred, train_y)

        # add regularization term #
        #-------------------------#

        # Apply L1 regularization
        if regularization_type == 'L1':
            #l1_norm = sum(p.abs().sum() for p in net.parameters() )
            l1_norm = [p.abs().sum() for p in net.parameters()][0] 
            #l1_norm = [p.abs().sum() for p in net.parameters()][0] 
            #l1_norm = torch.mean(p.abs().sum() for p in net.parameters() )

            loss += lambda_reg * l1_norm
            
        # Apply weighted L1 regularization
        elif regularization_type == 'weighted_L1':
            #wl1_norm = sum(torch.mul(weights[p[0]], p[1]).abs().sum() for p in enumerate(net.parameters()))
            wl1_norm = [torch.mul(weights[p[0]], p[1]).abs().sum() for p in enumerate(net.parameters())][0]
            loss += lambda_reg * wl1_norm

        loss.backward()

        # Gradient clipping:
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_max_norm )
        
        loss_per_epoch.append(loss.item())
        return loss
    
        #optimizer.step()
        
        # Zero the parameter gradients
        #optimizer.zero_grad()
    

    #tol=1e-6,
    #max_iter=int(1e6),
    #warm_start=True,
    #intercept_scaling=10000.0,


    lbfgs = optim.LBFGS(net.parameters(), 
                        lr = learn_rate, 
                        history_size=history_size, 
                        max_iter = max_iter, 
                        tolerance_grad = tolerance_grad,  
                        tolerance_change = tolerance_change, 
                        line_search_fn="strong_wolfe")

    # A tensor of weights for the weighted L1
    if regularization_type == 'weighted_L1':
        weights = [
            torch.from_numpy(np.array([weights])),
            torch.from_numpy(np.array([1.0])),
        ]

    loss_per_epoch = []

    for epoch in range(n_epochs):

        lbfgs.step(closure)

    #print("loss : ")     
    #print(loss)
    #return net
    return {"net": net, "loss": loss_per_epoch}




def validate(loader, model, criterion, device):                       
    #correct = 0                                               
    #total = 0                                                 
    running_loss = 0.0                                        
    model.eval()
      
    with torch.no_grad():                                     
        for i, data in enumerate(loader):                     
            inputs, labels = data                             
            inputs = inputs.to(device)                        
            labels = labels.to(device)                        
            
            outputs = model(inputs.float()).squeeze()    

            labels_hat=outputs.float()

            #loss = criterion(labels_hat, labels1.squeeze().type(torch.LongTensor)-1)
            loss = criterion(labels_hat.double(), labels.double())

            running_loss = running_loss + loss.item()
         
    return running_loss/(i+1)



def lr_train_minibatch(train_in, 
                        n_epochs=100, 
                        batch_size = 30, 
                        learn_rate = 0.01, 
                        loss_fun = 'log',
                        dependent = "y",
                        independent = None):

    # Load dataset
    train_dataset_orig = SynthDataset(train_in, dependent, independent)

    train_size = int(0.8 * len(train_dataset_orig))
    valid_size = len(train_dataset_orig) - train_size
    train_dataset, valid_dataset = random_split(train_dataset_orig, [train_size, valid_size])

    # Dataloaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    net=TorchLogisticRegression(train_dataset_orig.n_features)
    
    # Loss function

    if loss_fun == 'log':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_fun == 'mnb':
        criterion = mnbLoss()

    # Optimizer
  
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)

    # Train the net
    loss_per_iter = []
    loss_per_batch = []

    #valid_loss_per_iter = []
    valid_loss_per_batch = []

    best_loss = np.inf

    for epoch in range(n_epochs):

        running_loss = 0.0
        for i, (inputs, label) in enumerate(trainloader):
            inputs = inputs.to(device)
            label = label.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs.float() ).squeeze()

            label_hat=outputs.float()

            #print("main: inputs = "+str(inputs.shape) + " outputs = "+ str(outputs.shape))
            #print("main: label_hat = "+str(label_hat.shape) + " label = "+ str(label.shape))

            # calculate loss
            #loss=criterion(label_hat, label.squeeze().type(torch.LongTensor)-1)
            loss=criterion(label_hat.double(), label.double() )

            

            #loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        loss_per_batch.append(running_loss / (i + 1))
        running_loss = 0.0

        # Evaluate validation loss
        vloss = validate(validloader, net, criterion, device)
        valid_loss_per_batch.append(vloss)

        # If validation loss is lower than lowest then save the model
        #if vloss < best_loss:
        #    save_network(net, "best_model"+save_name_mod)
        #    best_loss = vloss

  
    #print("Root mean squared error")
    #print("Training:", np.sqrt(loss_per_batch[-1]))

    return{"model":net,
           "loss_per_batch": loss_per_batch,
           "valid_loss_per_batch": valid_loss_per_batch}



def reg_validate(loader, model, criterion, device, regularization_type = 'L1', weights = [], lambda_reg = 0.01):                       
    #correct = 0                                               
    #total = 0                                                 
    running_loss = 0.0                                        
    model.eval()
      
    with torch.no_grad():                                     
        for i, data in enumerate(loader):                     
            inputs, labels = data                             
            inputs = inputs.to(device)                        
            labels = labels.to(device)                        
            
            outputs = model(inputs.float()).squeeze()    

            labels_hat=outputs.float()

            #loss = criterion(labels_hat, labels1.squeeze().type(torch.LongTensor)-1)
            loss = criterion(labels_hat.double(), labels.double())


            # add regularization term #
            #-------------------------#

            # Apply L1 regularization
            if regularization_type == 'L1':
                #l1_norm = sum(p.abs().sum() for p in net.parameters() )
                #l1_norm = [p.abs().sum() for p in net.parameters()][0] 
                l1_norm = [p.abs().sum() for p in model.parameters()][0] 
                #l1_norm = torch.mean(p.abs().sum() for p in net.parameters() )

                loss += lambda_reg * l1_norm
            
            #  Apply weighted L1 regularization
            elif regularization_type == 'weighted_L1':
                #wl1_norm = sum(torch.mul(weights[p[0]], p[1]).abs().sum() for p in enumerate(model.parameters()))
                wl1_norm = [torch.mul(weights[p[0]], p[1]).abs().sum() for p in enumerate(model.parameters())][0]
                loss += lambda_reg * wl1_norm
                

            running_loss = running_loss + loss.item()
         
    return running_loss/(i+1)



def reg_lr_train_minibatch(train_in, 
                            n_epochs=100, 
                            batch_size = 30, 
                            learn_rate = 0.01, 
                            loss_fun = 'log', 
                            regularization_type = 'L1', 
                            weights = [], 
                            lambda_reg = 0.01,
                            dependent = "y",
                            independent = None):

    # Load dataset
    train_dataset_orig = SynthDataset(train_in, dependent, independent)

    train_size = int(0.8 * len(train_dataset_orig))
    valid_size = len(train_dataset_orig) - train_size
    train_dataset, valid_dataset = random_split(train_dataset_orig, [train_size, valid_size])

    # Dataloaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    net=TorchLogisticRegression(train_dataset_orig.n_features)
    
    # Loss function

    if loss_fun == 'log':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_fun == 'mnb':
        criterion = mnbLoss()

    # Optimizer
  
    optimizer = torch.optim.SGD(net.parameters(), lr=learn_rate)

    # Train the net
    loss_per_iter = []
    loss_per_batch = []

    #valid_loss_per_iter = []
    valid_loss_per_batch = []

    best_loss = np.inf

    for epoch in range(n_epochs):

        running_loss = 0.0
        for i, (inputs, label) in enumerate(trainloader):
            inputs = inputs.to(device)
            label = label.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs.float() ).squeeze()

            label_hat=outputs.float()

            #print("main: inputs = "+str(inputs.shape) + " outputs = "+ str(outputs.shape))
            #print("main: label_hat = "+str(label_hat.shape) + " label = "+ str(label.shape))

            # calculate loss
            #loss=criterion(label_hat, label.squeeze().type(torch.LongTensor)-1)
            loss=criterion(label_hat.double(), label.double() )

            # add regularization term #
            #-------------------------#

            # Apply L1 regularization
            if regularization_type == 'L1':
                #l1_norm = sum(p.abs().sum() for p in net.parameters() )
                #l1_norm = [p.abs().sum() for p in net.parameters()][0] 
                l1_norm = [p.abs().sum() for p in net.parameters()][0] 
                #l1_norm = torch.mean(p.abs().sum() for p in net.parameters() )

                loss += lambda_reg * l1_norm
            
            #  Apply weighted L1 regularization
            elif regularization_type == 'weighted_L1':
                #wl1_norm = sum(torch.mul(weights[p[0]], p[1]).abs().sum() for p in enumerate(net.parameters()))
                wl1_norm = [torch.mul(weights[p[0]], p[1]).abs().sum() for p in enumerate(net.parameters())][0]
                loss += lambda_reg * wl1_norm

            #loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Save loss to plot
            running_loss += loss.item()
            loss_per_iter.append(loss.item())

        loss_per_batch.append(running_loss / (i + 1))
        running_loss = 0.0

        # Evaluate validation loss
        vloss = reg_validate(validloader, net, criterion, device, regularization_type = regularization_type, weights = weights, lambda_reg = lambda_reg)
        valid_loss_per_batch.append(vloss)

        # If validation loss is lower than lowest then save the model
        #if vloss < best_loss:
        #    save_network(net, "best_model"+save_name_mod)
        #    best_loss = vloss

  
    #print("Root mean squared error")
    #print("Training:", np.sqrt(loss_per_batch[-1]))

    return{"model":net,
           "loss_per_batch": loss_per_batch,
           "valid_loss_per_batch": valid_loss_per_batch}





def mydot(K, L):
   if len(K) != len(L):
      return 0

   return sum(i[0] * i[1] for i in zip(K, L))







def reg_path(
        df_train,
        df_test,
        dependent = "y",
        independent_in = None,
        test_harms = None,

        n_epochs = 10000,
        learn_rate = 1.0,
        loss_fun = "log",

        regularization_type = 'L1',

        log_space_min = 0,
        log_space_max = 5,
        log_space_steps = 16,

        label = "reg_path",
        
        history_size = 100, 
        max_iter = 20, 
        tolerance_grad=1e-07, 
        tolerance_change=1e-09,

        grad_max_norm = 1.
        ):


        if independent_in == None:
            independent = list(df_train.drop([dependent], axis=1).columns)
        else:
            independent = independent_in
            
        cs = l1_min_c(df_train[independent], df_train[dependent], loss="log") * np.logspace(log_space_max, log_space_min, log_space_steps)

        if test_harms == None:
             test_harms = [0. for i in independent]
        

        n_sample = df_train.shape[0]

        test_reg_path = {}
        path_df = []

        start_model = lr_train(df_train, n_epochs=n_epochs, learn_rate=learn_rate, loss_fun = loss_fun)['net']

        test_train_dataset_orig = SynthDataset(df_test, dependent, independent)
        test_x = torch.from_numpy(test_train_dataset_orig.x.to_numpy().astype(np.float32) )

        total_harm = sum(test_harms)
        if total_harm > 0:
            tot = sum(test_harms)
            l1_weights = [i/tot for i in test_harms]
        else:
            l1_weights = [1. for i in test_harms]
             

        for i, inv_lambda_reg in enumerate(cs):
            print(i)
            print("lambda")
            print(1./inv_lambda_reg)
            lda = 1./(inv_lambda_reg)
            lda = lda/n_sample
             
            new_model = \
                        reg_lr_train(df_train, 
                                   net = start_model, 
                                   n_epochs=n_epochs, 
                                   learn_rate = learn_rate, 
                                   loss_fun = loss_fun, 
                                   regularization_type = regularization_type, 
                                   weights = l1_weights, 
                                   lambda_reg=lda,
                                   
                                    history_size = history_size, 
                                    max_iter = max_iter, 
                                    tolerance_grad=tolerance_grad, 
                                    tolerance_change=tolerance_change,

                                    grad_max_norm = grad_max_norm
                                   )
            # Calculate the net benefit accounting for test harm
            pred = torch.sigmoid(new_model["net"](test_x )).detach().numpy()

            # mean net benefit with no test harm
            mnb0 = dcat.mean_net_benefit(df_test['y'], pred, n_thresh=100)['mnb'] 

            # Calculate the test harm
            coefs = new_model["net"].linear.weight.detach().numpy()
            coefs_used = [np.heaviside(i, 0.) for i in abs(coefs[0])]

            n_coefs = len([i for i in coefs_used if i>0])

            harm = mydot(coefs_used, test_harms)

                
            

            coefs_df = pd.DataFrame(new_model["net"].linear.weight.detach().numpy(), columns=independent)
            coefs_used_df = pd.DataFrame([coefs_used], columns = [i+"_used" for i in independent])

            out = pd.DataFrame({
                    "c": [inv_lambda_reg],
                    "lambda": [lda],
                    "label": [label],
                    "mnb0" : [mnb0],
                    "harm" : [harm],
                    "mnb": [mnb0-harm],
                    "n_coefs": [n_coefs]
                })
                
            out = pd.concat([out, coefs_df, coefs_used_df], axis=1)
            path_df.append(out)

            #-----------------#
            test_reg_path[i] = copy.deepcopy(new_model['net'])
            start_model = copy.deepcopy(new_model['net'])
        
        path_df = pd.concat(path_df, axis=0).reset_index().drop("index", axis=1)

        return {"models": test_reg_path, "path": path_df}
