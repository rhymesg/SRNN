
# author: Youngjoo Kim

import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np

class NodeRNN(nn.Module):
    '''
    Class representing the nodeRNN in the st graph
    '''
    def __init__(self, args):
        '''
        Initializer function
        [params]
        args : Training arguments
        '''
        super(NodeRNN, self).__init__()

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(args.node_input_size, args.node_embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # Linear layer to embed edgeRNN hidden states
        self.edge_embed = nn.Linear(args.edge_rnn_size*2, args.node_embedding_size)

        # The LSTM cell
        self.cell = nn.LSTMCell(args.node_embedding_size*2, args.node_rnn_size)

        # Output linear layer
        self.output_linear = nn.Linear(args.node_rnn_size, args.node_output_size)

    def forward(self, inp, h_edgeRNN, h, c):
        '''
        Forward pass for the model
        [params]
        h_edgeRNN : concatenated hidden state from edgeRNNs
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        
        [returns]
        out : predicted node features
        '''
        
        # Encode the input
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # embeddings
        h_edgeRNN_embedded = self.edge_embed(h_edgeRNN)
        h_edgeRNN_embedded = self.relu(h_edgeRNN_embedded)
        h_edgeRNN_embedded = self.dropout(h_edgeRNN_embedded)

        concat_encoded = torch.cat((encoded_input, h_edgeRNN_embedded), 1)

        # One-step of LSTM
        h_new, c_new = self.cell(concat_encoded, (h, c))

        # Get output
        out = self.output_linear(h_new)

        return out, h_new, c_new

class EdgeRNN(nn.Module):
    '''
    Class representing the edgeRNN in the st graph
    '''
    def __init__(self, args):
        '''
        Initializer function
        [params]
        args : Training arguments
        '''
        super(EdgeRNN, self).__init__()

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(args.edge_input_size, args.edge_embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.cell = nn.LSTMCell(args.edge_embedding_size, args.edge_rnn_size)

    def forward(self, inp, h, c):
        '''
        Forward pass for the model
        [params]
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        '''
        
        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new

class SRNN(nn.Module):
    '''
    Class representing the SRNN model
    '''
    def __init__(self, args):
        '''
        Initializer function
        [params]
        args : Training arguments
        '''
        super(SRNN, self).__init__()

        self.seq_length = args.seq_length
        self.node_output_size = args.node_output_size

        # Initialize the Node and Edge RNNs
        self.NodeRNN = NodeRNN(args)
        self.EdgeRNN_spatial = EdgeRNN(args)
        self.EdgeRNN_temporal = EdgeRNN(args)
        
        self.stgraph = None
        
        self.initialize()
        
    def initialize(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                
        print('- SRNN has been initialized.')

    def setStgraph(self, stgraph):
        self.stgraph = stgraph

    def forward(self, data_nodes, data_temporalEdges, data_spatialEdges,
                h_nodeRNN, c_nodeRNN, h_temporalEdgeRNN, c_temporalEdgeRNN,
                h_spatialEdgeRNN, c_spatialEdgeRNN):
        '''
        Forward pass for the SRNN
        [params]
        data_nodes : input node features (seq_length_p1 x numNodes x 1)
        data_temporalEdges : input temporal edge features (seq_length_p1 x numTemporalEdges x 2)
        data_spatialEdges : input temporal edge features (seq_length_p1 x numSpatialEdges x 2)
        h's and c's : hidden states and cell states for corresponding RNNs

        [returns]
        outputs : predicted node features for all the time step in the sequence
                  (seq_length x numNodes x 1)
        '''
        
        assert self.stgraph != None # run setStgraph after reading the graph
        
        outputs = Variable(torch.zeros(self.seq_length, self.stgraph.numNodes, self.node_output_size))
        
        # For each time step
        for dataIdx in range(self.seq_length):
        
            # Temporal Edge RNNs
            h_temporalEdgeRNN, c_temporalEdgeRNN = self.EdgeRNN_temporal(data_temporalEdges[dataIdx],
                                                                                               h_temporalEdgeRNN,
                                                                                               c_temporalEdgeRNN)
            # Spatial Edge RNNs
            h_spatialEdgeRNN, c_spatialEdgeRNN = self.EdgeRNN_spatial(data_spatialEdges[dataIdx],
                                                                                               h_spatialEdgeRNN,
                                                                                               c_spatialEdgeRNN)
            
            # Node RNNs
            isFirstNode = True
            for (n, nodeId) in enumerate(self.stgraph.nodeList):
                
                if (len(self.stgraph.edgesConnected[nodeId])) > 0:
                    indexList_edgesConnected = np.array([self.stgraph.spatialEdgeList.index(edgeId) for edgeId in self.stgraph.edgesConnected[nodeId]])
                    indexList_edgesConnected = Variable(torch.LongTensor(indexList_edgesConnected))
    
                    h_spatialEdgeRNN_connected = torch.index_select(h_spatialEdgeRNN, 0, indexList_edgesConnected)
                              
                    # sum hidden states of spatial edgeRNNs of connected to the node
                    h_spatialEdgeRNN_sum = torch.sum(h_spatialEdgeRNN_connected, dim=0)
                    
                    # concatenate hidden states of temporal edgeRNN and summed spatial edgeRNNs states for node n
                    h_edgeRNN_eachNode = torch.cat((h_temporalEdgeRNN[n], h_spatialEdgeRNN_sum), dim=0).unsqueeze(0)
                else:
                    h_edgeRNN_eachNode = h_temporalEdgeRNN[n]
                
                if (isFirstNode):
                    h_edgeRNN = h_edgeRNN_eachNode
                    isFirstNode = False
                else:
                    h_edgeRNN = torch.cat((h_edgeRNN, h_edgeRNN_eachNode), dim=0)
                        
            outputs[dataIdx], h_nodeRNN, c_nodeRNN = self.NodeRNN(data_nodes[dataIdx], 
                                                                   h_edgeRNN,
                                                                   h_nodeRNN, 
                                                                   c_nodeRNN)
        return outputs