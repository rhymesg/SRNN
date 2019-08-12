# -*- coding: utf-8 -*-

# author: Youngjoo Kim

import numpy as np
import pandas as pd

class ST_GRAPH():

    def __init__(self, args):

        self.batch_size = args.batch_size
        self.seq_length_p1 = args.seq_length + 1 # Sequence length (input length) plus 1

        self.nodes = {}
        self.edges = {}


    def readGraph(self, numNodes, graph_path):
        
        self.nodes = {}
        self.edges = {}
        
        ### load adjacency matrix 
        df_graph_read = pd.read_csv(graph_path, header=None)
        graph_read = np.array(df_graph_read.values, dtype=np.float32)

        numNodes_row, numNodes_col = graph_read.shape
        assert numNodes_row == numNodes_col
        assert numNodes_row == numNodes
        
        # nodes
        for n in range(numNodes):
            
            node_type = 'O'
            node_id = n
            self.nodes[node_id] = ST_NODE(node_type, node_id)
        
        nodeList = list(self.nodes.keys())
        
        # edges
        edgesConnected = {}
        for node in nodeList:
            edgesConnected[node] = []
        for (n1, node1) in enumerate(nodeList):
            for (n2, node2) in enumerate(nodeList):
                
                # temporal edges
                if (n1 == n2):
                    edge_type = 'T'                    
                    edge_id = (node1, node1)
                    self.edges[edge_id] = ST_EDGE(edge_type, edge_id)
                    
                # spatial edges
                if (graph_read[n1][n2] == 1):
                    edge_type = 'S'
                        
                    edge_id = (node1, node2)
                    self.edges[edge_id] = ST_EDGE(edge_type, edge_id)
                    edgesConnected[node1].append(edge_id)
                    edgesConnected[node2].append(edge_id)
                
        edgeList = list(self.edges.keys())
        numNodes = len(nodeList)
        assert numNodes_row == numNodes
        numEdges = len(edgeList)
        temporalEdgeList = [e for e in edgeList if e[0] == e[1]]
        spatialEdgeList = [e for e in edgeList if e[0] != e[1]]
        numTemporalEdges = len(temporalEdgeList)
        numSpatialEdges = len(spatialEdgeList)
        assert numTemporalEdges + numSpatialEdges == numEdges
        
        self.numNodes = numNodes
        self.numEdges = numEdges
        self.nodeList = nodeList
        self.edgeList = edgeList
        self.temporalEdgeList = temporalEdgeList
        self.spatialEdgeList = spatialEdgeList
        self.numTemporalEdges = numTemporalEdges
        self.numSpatialEdges = numSpatialEdges
        self.edgesConnected = edgesConnected
        
        print ('- Graph from', graph_path, 'has been loaded (numNodes: {}).'.format(self.numNodes))
        
#        self.printGraph()
        print('- numNodes: {}, numTemporalEdges: {}, numSpatialEdges: {}'.format(numNodes, numTemporalEdges, numSpatialEdges))


    def putSequenceData(self, X):
        
        for t in range(self.seq_length_p1):
            
            for n in self.nodes.keys():
                feature = X[t][n]
                self.nodes[n].node_data_list[t] = feature
                
            for e in self.edges.keys():
                # temporal edges
                if self.edges[e].edge_type == 'T':
                    n1 = self.edges[e].edge_id[0]
                    n2 = self.edges[e].edge_id[1]
                    assert n1 == n2
                    if t == 0: # first data
                        feature = (X[0][n1], X[0][n1])
                    else:
                        feature = (X[t-1][n1], X[t][n1])
                    
                    self.edges[e].edge_data_list[t] = feature
                
                # spatial edges
                if self.edges[e].edge_type == 'S': 
                    n1 = self.edges[e].edge_id[0]
                    n2 = self.edges[e].edge_id[1]
                    assert n1 != n2
                    
                    feature = (X[t][n1], X[t][n2])
                    self.edges[e].edge_data_list[t] = feature


    def getSequenceData(self):
        
        data_nodes = np.zeros((self.seq_length_p1,self.numNodes,1))
        data_temporalEdges = np.zeros((self.seq_length_p1,self.numTemporalEdges,2))
        data_spatialEdges = np.zeros((self.seq_length_p1,self.numSpatialEdges,2))
    
        for dataIdx in range(self.seq_length_p1):
            for (n, nodeId) in enumerate(self.nodeList):
                data_nodes[dataIdx, n, :] = self.nodes[nodeId].node_data_list[dataIdx]
  
            for (e, edgeId) in enumerate(self.temporalEdgeList):
                data_temporalEdges[dataIdx, e, :] = self.edges[edgeId].edge_data_list[dataIdx]
        
            for (e, edgeId) in enumerate(self.spatialEdgeList):
                data_spatialEdges[dataIdx, e, :] = self.edges[edgeId].edge_data_list[dataIdx]
                
        return data_nodes, data_temporalEdges, data_spatialEdges


    def printGraph(self):
        '''
        Print function for the graph
        For debugging purposes
        '''
        print('')
        print('---- Print Graph ----')
        print('- numNodes: {}, numTemporalEdges: {}, numSpatialEdges: {}'.format(self.numNodes, 
                                                      self.numTemporalEdges, self.numSpatialEdges))
        print('Nodes:')
        print(self.nodeList)
        print('Spatial Edges:')
        print(self.spatialEdgeList)
        
        print('Connections:')
        for node in self.nodeList:
            print('- Node {} is connected with: {}'.format(node, self.edgesConnected[node]))
        print('--------')
            
class ST_NODE():

    def __init__(self, node_type, node_id):
   
        self.node_type = node_type
        self.node_id = node_id
        self.node_data_list = {}

class ST_EDGE():

    def __init__(self, edge_type, edge_id):
     
        self.edge_type = edge_type
        self.edge_id = edge_id
        self.edge_data_list = {}

