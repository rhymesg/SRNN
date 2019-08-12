# -*- coding: utf-8 -*-

# author: Youngjoo Kim (16 Jan 2019)
# Youngjoo Kim, Peng Wang, Lyudmila Mihaylova, "Scalable Learning with a 
# Structural Recurrent Neural Network for Short-Term Traffic Prediction", \
# Submitted to IEEE Sensors Journal, Jan 2019

# main script

import argparse
import os
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np

from dataLoader import DataLoader
from st_graph import ST_GRAPH
from model import SRNN

data_dir = 'dataset/Santander/'
save_dir = 'save/'
log_dir = 'log/'

def main():
    parser = argparse.ArgumentParser()

    # Data size
    parser.add_argument('--numNodes_set', type=int, default=-1,
                        help='Number of nodes to be used')
    parser.add_argument('--numData_set', type=int, default=-1,
                        help='Number of time steps to be used')
    parser.add_argument('--numData_train_set', type=int, default=-1,
                        help='Number of train data')

    # RNN size
    parser.add_argument('--node_rnn_size', type=int, default=64,
                        help='Size of Node RNN hidden state')
    parser.add_argument('--edge_rnn_size', type=int, default=64,
                        help='Size of Edge RNN hidden state')

    # Embedding size
    parser.add_argument('--node_embedding_size', type=int, default=32,
                        help='Embedding size of node features')
    parser.add_argument('--edge_embedding_size', type=int, default=32,
                        help='Embedding size of edge features')
    
    # Multi-layer RNN layer size
    parser.add_argument('--num_layer', type=int, default=3,
                        help='Number of layers of RNN')

    # Sequence length
    parser.add_argument('--seq_length', type=int, default=10,
                        help='Sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of epochs')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=1.,
                        help='clip gradients at this value')
    
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00001,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for the optimizer')

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')
    
    # Print every x batch
    parser.add_argument('--printEvery', type=int, default=1,
                        help='Train/Eval result print period')

    # Input and output size
    parser.add_argument('--node_input_size', type=int, default=1,
                        help='Dimension of the node features')
    parser.add_argument('--edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--node_output_size', type=int, default=1,
                        help='Dimension of the node output')

    args = parser.parse_args()

    Run_SRNN_NormalCase(args, no_dataset = 1)
    
    #Run_SRNN_Scalability(args)
    
    #Run_SRNN_Different_Dataset(args,2,1)
    
    #Run_SRNN_test_parameters(args)
    
    
# run with various combinations of hyperparameters to find the best one
def Run_SRNN_test_parameters(args):
    
    args.num_epochs = 1
    args.numData_set = 2000
    args.printEvery = 50
    
    grad_clip_list = [1.0, 5.0]
    learning_rate_list = [0.0001, 0.0005]
    lambda_list = [0.00001, 0.00005]
    node_rnn_size_list = [64, 128]
    edge_rnn_size_list = [64, 128]
    
    log_dir_test = log_dir+'parameter_test/'
    if not os.path.exists(log_dir_test):
        os.makedirs(log_dir_test)
    f1 = open(log_dir_test+"parameter_test3_4.txt", "w")
    f2 = open(log_dir_test+"parameter_test5_5.txt", "w")
    
    out_str_merge1 = ''
    out_str_merge2 = ''
    
    for lr in learning_rate_list:
        args.learning_rate = lr
        for ll in lambda_list:
            args.lambda_param = ll
            for nr in node_rnn_size_list:
                args.node_rnn_size = nr
                for er in edge_rnn_size_list:
                    args.edge_rnn_size = er
                    for gc in grad_clip_list:
                        args.grad_clip = gc
                        
                        out_str = '== learning rate: {}, lambda: {}, node rnn size: {}, edge rnn size: {}, grad_clip: {}:\n'.format(lr, 
                              ll, nr, er, gc)
                        out_str += str(Run_SRNN_Different_Dataset(args,3,4)) + '\n'
                        print(out_str)
                        out_str_merge1 += out_str
                        
                        out_str = '== learning rate: {}, lambda: {}, node rnn size: {}, edge rnn size: {}, grad_clip: {}:\n'.format(lr, 
                              ll, nr, er, gc)
                        out_str += str(Run_SRNN_Different_Dataset(args,5,5)) + '\n'
                        print(out_str)
                        out_str_merge2 += out_str

    print(out_str_merge1)
    print('')
    print(out_str_merge2)

    f1.write(out_str_merge1)
    f2.write(out_str_merge2)
    f1.close()
    f2.close()
    

# run for testing the sacalability
def Run_SRNN_Scalability(args):
    
    no_dataset_list = [1, 2, 3, 4]
    
    args.num_epochs = 10
    args.numData_set = -1
    args.printEvery = 50
    
    for no_dataset_train in no_dataset_list:
        for no_dataset_eval in no_dataset_list:
            Run_SRNN_Different_Dataset(args, no_dataset_train, no_dataset_eval)

# train with no_dataset_train and evalaute with no_dataset_eval
def Run_SRNN_Different_Dataset(args, no_dataset_train, no_dataset_eval):
    
    print('')
    print('')
    print('[[ Train on Dataset {} and Evaluation on Dataset {} ]]'.format(no_dataset_train, no_dataset_eval))
    
    # Initialize net
    net = SRNN(args)
    # Construct the DataLoader object that loads data
    dataloader = DataLoader(args)
    # Construct the ST-graph object that reads graph
    stgraph = ST_GRAPH(args)

    optimizer = torch.optim.Adagrad(net.parameters())

    print('- Number of trainable parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))

    best_eval_loss = 10000
    best_epoch = 0
    
    eval_loss_res = np.zeros((args.num_epochs+1, 2))
    for e in range(args.num_epochs):
        epoch = e + 1
        start_train = time.time()

        ####  Training #### 
        print('')
        print('-- Training, epoch {}/{}, Dataset {} on {}'.format(epoch, args.num_epochs, no_dataset_train, no_dataset_eval))
        loss_epoch = 0

        if (epoch > 1):
            net.initialize()

        data_path, graph_path = Data_path(no_dataset_train)
        
        dataloader.load_data(data_path)

        stgraph.readGraph(dataloader.num_sensor, graph_path)

        net.setStgraph(stgraph)

        # For each batch
        for b in range(dataloader.num_batches_train):
            batch = b + 1;
            start = time.time()

            # Get batch data
            x = dataloader.next_batch_train()

            # Loss for this batch
            loss_batch = 0

            # For each sequence in the batch
            for sequence in range(dataloader.batch_size):
                
                # put node and edge features
                stgraph.putSequenceData(x[sequence]) 
                
                # get data to feed
                data_nodes, data_temporalEdges, data_spatialEdges = stgraph.getSequenceData() 

                # put a sequence to net
                loss_output, data_nodes, outputs = forward(net, optimizer, args, stgraph, 
                                                           data_nodes, data_temporalEdges, data_spatialEdges)
                loss_output.backward()
                loss_batch += loss_RMSE(data_nodes[-1], outputs[-1], dataloader.scaler)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch
            if ((e * dataloader.num_batches_train + batch) % args.printEvery == 1):
                print(
                'Train: {}/{}, train_loss = {:.3f}, time/batch = {:.3f}'.format(e * dataloader.num_batches_train + batch,
                                                                                    args.num_epochs * dataloader.num_batches_train,
                                                                                    loss_batch,
                                                                                    end - start))
        end_train = time.time()
        
        # Compute loss for the entire epoch
        loss_epoch /= dataloader.num_batches_train 
        print('(epoch {}), train_loss = {:.3f}, time/train = {:.3f}'.format(epoch, loss_epoch,
                                                                      end_train - start_train))
        
        # Save the model after each epoch
        save_path = Save_path(no_dataset_train, epoch)
        print('Saving model to '+save_path)
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)
        print('')
        
        
        #### Evaluation #### 
        print('-- Evaluation, epoch {}/{}, Dataset {} on {}'.format(epoch, args.num_epochs, no_dataset_train, no_dataset_eval))
        
        data_path, graph_path = Data_path(no_dataset_eval)
        log_path = Log_path(no_dataset_train, no_dataset_eval, 'SRNN')
    
        dataloader.load_data(data_path)

        stgraph.readGraph(dataloader.num_sensor, graph_path)
        
        net.setStgraph(stgraph)
        
        loss_epoch = 0
        for b in range(dataloader.num_batches_eval):
            batch = b + 1;
            start = time.time()

            # Get batch data
            x = dataloader.next_batch_eval()

            # Loss for this batch
            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                
                # put node and edge features
                stgraph.putSequenceData(x[sequence]) 
                
                # get data to feed
                data_nodes, data_temporalEdges, data_spatialEdges = stgraph.getSequenceData()

                # put a sequence to net
                _, data_nodes, outputs = forward(net, optimizer, args, stgraph, 
                                                           data_nodes, data_temporalEdges, data_spatialEdges)
                
                loss_batch += loss_RMSE(data_nodes[-1], outputs[-1], dataloader.scaler)

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch
            if ((e * dataloader.num_batches_eval + batch) % args.printEvery == 1):
                print(
                'Eval: {}/{}, eval_loss = {:.3f}, time/batch = {:.3f}'.format(e * dataloader.num_batches_eval + batch,
                                                                                    args.num_epochs * dataloader.num_batches_eval,
                                                                                    loss_batch, 
                                                                                    end - start))
        loss_epoch /= dataloader.num_batches_eval
        eval_loss_res[e] = (epoch, loss_epoch)

        # Update best validation loss until now
        if loss_epoch < best_eval_loss:
            best_eval_loss = loss_epoch
            best_epoch = epoch

        print('(epoch {}), eval_loss = {:.3f}'.format(epoch, loss_epoch))
        print('--> Best epoch: {}, Best evaluation loss {:.3f}'.format(best_epoch, best_eval_loss))
        
    # Record the best epoch and best validation loss overall
    eval_loss_res[-1] = (best_epoch, best_eval_loss)
    np.savetxt(log_path, eval_loss_res, fmt='%d, %.3f')
    print ('- Eval result has been saved in ', log_path)
    
    return eval_loss_res[-1,1]
    
# train with no_dataset and evaluate with the same dataset
def Run_SRNN_NormalCase(args, no_dataset):

    data_path, graph_path = Data_path(no_dataset)
    log_path = Log_path(no_dataset)

    # Construct the DataLoader object that loads data
    dataloader = DataLoader(args)
    dataloader.load_data(data_path)
    
    # Construct the ST-graph object that reads graph
    stgraph = ST_GRAPH(args)
    stgraph.readGraph(dataloader.num_sensor, graph_path)
    
    # Initialize net
    net = SRNN(args)
    net.setStgraph(stgraph)
        
    print('- Number of trainable parameters:', sum(p.numel() for p in net.parameters() if p.requires_grad))

    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)
    optimizer = torch.optim.Adagrad(net.parameters())

    best_eval_loss = 10000
    best_epoch = 0
    
    print('')
    print('---- Train and Evaluation ----')
    
    eval_loss_res = np.zeros((args.num_epochs+1, 2))
    for e in range(args.num_epochs):
        epoch = e + 1
        
        ####  Training #### 
        print('-- Training, epoch {}/{}'.format(epoch, args.num_epochs))
        loss_epoch = 0

        # For each batch
        for b in range(dataloader.num_batches_train):
            batch = b + 1;
            start = time.time()

            # Get batch data
            x = dataloader.next_batch_train()

            # Loss for this batch
            loss_batch = 0

            # For each sequence in the batch
            for sequence in range(dataloader.batch_size):
                
                # put node and edge features
                stgraph.putSequenceData(x[sequence]) 
                
                # get data to feed
                data_nodes, data_temporalEdges, data_spatialEdges = stgraph.getSequenceData() 

                # put a sequence to net
                loss_output, data_nodes, outputs = forward(net, optimizer, args, stgraph, 
                                                           data_nodes, data_temporalEdges, data_spatialEdges)
                loss_output.backward()
                loss_batch += loss_RMSE(data_nodes[-1], outputs[-1], dataloader.scaler)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            print(
                'Train: {}/{}, train_loss = {:.3f}, time/batch = {:.3f}'.format(e * dataloader.num_batches_train + batch,
                                                                                    args.num_epochs * dataloader.num_batches_train,
                                                                                    loss_batch,
                                                                                    end - start))
        # Compute loss for the entire epoch
        loss_epoch /= dataloader.num_batches_train 
        print('(epoch {}), train_loss = {:.3f}'.format(epoch, loss_epoch))
        
        # Save the model after each epoch
        save_path = Save_path(no_dataset, epoch)
        print('Saving model to '+save_path)
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, save_path)           
    
        #### Evaluation #### 
        print('-- Evaluation, epoch {}/{}'.format(epoch, args.num_epochs))
        loss_epoch = 0
        for b in range(dataloader.num_batches_eval):
            batch = b + 1;
            start = time.time()

            # Get batch data
            x = dataloader.next_batch_eval()

            # Loss for this batch
            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                
                # put node and edge features
                stgraph.putSequenceData(x[sequence]) 
                
                # get data to feed
                data_nodes, data_temporalEdges, data_spatialEdges = stgraph.getSequenceData()

                # put a sequence to net
                _, data_nodes, outputs = forward(net, optimizer, args, stgraph, 
                                                           data_nodes, data_temporalEdges, data_spatialEdges)
                
                loss_batch += loss_RMSE(data_nodes[-1], outputs[-1], dataloader.scaler)

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch
            
            print(
                'Eval: {}/{}, eval_loss = {:.3f}, time/batch = {:.3f}'.format(e * dataloader.num_batches_eval + batch,
                                                                                    args.num_epochs * dataloader.num_batches_eval,
                                                                                    loss_batch, 
                                                                                    end - start))
        loss_epoch /= dataloader.num_batches_eval
        eval_loss_res[e] = (epoch, loss_epoch)

        # Update best validation loss until now
        if loss_epoch < best_eval_loss:
            best_eval_loss = loss_epoch
            best_epoch = epoch

        print('(epoch {}), eval_loss = {:.3f}'.format(epoch, loss_epoch))

    # Record the best epoch and best validation loss overall
    print('Best epoch: {}, Best evaluation loss {:.3f}'.format(best_epoch, best_eval_loss))
    eval_loss_res[-1] = (best_epoch, best_eval_loss)
    np.savetxt(log_path, eval_loss_res, fmt='%d, %.3f')
    print ('- Eval result has been saved in ', log_path)
    print('')
    
def forward(net, optimizer, args, stgraph, data_nodes, data_temporalEdges, data_spatialEdges):
    
    # Convert to Torch variables
    data_nodes = Variable(torch.from_numpy(data_nodes).float())
    data_temporalEdges = Variable(torch.from_numpy(data_temporalEdges).float())
    data_spatialEdges = Variable(torch.from_numpy(data_spatialEdges).float())
        
    # RNN hidden states and cell states
    h_nodeRNN = Variable(torch.zeros(stgraph.numNodes, args.node_rnn_size))
    c_nodeRNN = Variable(torch.zeros(stgraph.numNodes, args.node_rnn_size))
    
    h_temporalEdgeRNN = Variable(torch.zeros(stgraph.numTemporalEdges, args.edge_rnn_size))
    c_temporalEdgeRNN = Variable(torch.zeros(stgraph.numTemporalEdges, args.edge_rnn_size))
 
    h_spatialEdgeRNN = Variable(torch.zeros(stgraph.numSpatialEdges, args.edge_rnn_size))
    c_spatialEdgeRNN = Variable(torch.zeros(stgraph.numSpatialEdges, args.edge_rnn_size))
          
    # Zero out the gradients
    net.zero_grad()
    optimizer.zero_grad()

    # Forward propagation
    outputs = net.forward(data_nodes[:args.seq_length], data_temporalEdges[:args.seq_length], data_spatialEdges[:args.seq_length],
                          h_nodeRNN, c_nodeRNN,
                          h_temporalEdgeRNN, c_temporalEdgeRNN,
                          h_spatialEdgeRNN, c_spatialEdgeRNN)

    loss = nn.MSELoss()
    loss_output = loss(outputs[-1], data_nodes[-1])
    
    return loss_output, data_nodes, outputs


def loss_RMSE(x, y, scaler):
    
    x = np.array(x.detach())
    y = np.array(y.detach())
    
    x = scaler.scale_inverse(x)
    y = scaler.scale_inverse(y)

    return np.sqrt(np.mean((x-y)*(x-y)))

def Data_path(no_dataset):

    data_path = data_dir + 'Data_'+str(no_dataset)+'.csv'
    graph_path = data_dir + 'Adjacency_'+str(no_dataset)+'.csv'

    return data_path, graph_path

def Log_path(no_dataset_train, no_dataset_eval = None, prefix=''):
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  
        
    if no_dataset_eval == None:
        log_path = log_dir+prefix+'_loss_eval_dataset_'+str(no_dataset_train)+'.csv'
    else:
        log_path = log_dir+prefix+'_loss_eval_dataset_'+str(no_dataset_train)+'_on_'+str(no_dataset_eval)+'.csv'
    
    return log_path
    
# Path to store the checkpoint file
def Save_path(no_dataset, no_epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir_dataset = save_dir+'dataset_'+str(no_dataset)
    if not os.path.exists(save_dir_dataset):
        os.makedirs(save_dir_dataset)
    return save_dir_dataset+'/srnn_model_epoch'+str(no_epoch)+'.tar'

if __name__ == '__main__':
    main()
