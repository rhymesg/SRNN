# -*- coding: utf-8 -*-

# author: Youngjoo Kim

import numpy as np
import pandas as pd
from MinMaxScaler import MinMaxScaler

class DataLoader():

    def __init__(self, args):

        # Store the arguments
        self.batch_size = args.batch_size
        self.seq_length_p1 = args.seq_length + 1 # Sequence length (input length) plus 1
        self.numNodes_set = args.numNodes_set
        self.numData_set = args.numData_set
        self.numData_train_set = args.numData_train_set
        self.num_data_eval = 0

        self.scaler = MinMaxScaler() # scale data to (0, 1) before feeding into SRNN

    def reset_pointers(self):
        # Reset data pointers of the dataloader object
        self.data_pointer_train = 0
        self.data_pointer_eval = 0

    def load_data(self, data_path):

        # read traffic data file
        df_data_read = pd.read_csv(data_path, header=None)
        allData = np.array(df_data_read.values, dtype=np.float32)
        
        self.scaler.fit(allData, feature_range=(0,1), Min=0, Max=150) # scale data to (0, 1) before feeding into SRNN
        allData_scaled = self.scaler.scale(allData)
        
        print ('- Data from', data_path, 'has been loaded (size: {0}x{1}).'.format(allData.shape[0], allData.shape[1]))
        
        self.num_sensor = self.numNodes_set
        self.num_data = self.numData_set
        self.num_data_train = self.numData_train_set
        num_data_read, num_sensor_read = allData.shape
        if (self.numNodes_set == -1 or self.num_sensor > num_sensor_read):
            self.num_sensor = allData.shape[1]
        if (self.numData_set == -1 or self.num_data > num_data_read):
            self.num_data = allData.shape[0]
        if (self.numData_train_set == -1 or self.numData_train_set > self.num_data):
            self.num_data_train = int(self.num_data*3/4)
        
        self.num_data_eval = self.num_data - self.num_data_train
                
        scaled_train = allData_scaled[:self.num_data_train,:self.num_sensor]
        scaled_eval = allData_scaled[self.num_data_train:self.num_data,:self.num_sensor]

        self.dataset_train = []
        self.dataset_eval = []
        
        for i in range(self.num_data_train):
            self.dataset_train.append(scaled_train[i,:])
            
        for i in range(self.num_data_eval):
            self.dataset_eval.append(scaled_eval[i,:])
      
        num_seq_train = int((self.num_data_train) / (self.seq_length_p1))
        num_seq_eval = int((self.num_data_eval) / (self.seq_length_p1))
        
        self.num_batches_train = int(num_seq_train / self.batch_size)
        self.num_batches_eval = int(num_seq_eval / self.batch_size)

        self.reset_pointers()

    def print_data_info(self):
        print ("- num_sensor: {}, num_data: {}, num_data_train: {}, num_data_eval: {}".format(self.num_sensor, 
               self.num_data, self.num_data_train, self.num_data_eval))
        print ('- trainData: {0}x{1}, evalData: {2}x{3}'.format(
                len(self.dataset_train), len(self.dataset_train[0]), len(self.dataset_eval), len(self.dataset_eval[0])))
        print ('- Sequence length: {} + 1, batch size: {}'.format(self.seq_length_p1 - 1, self.batch_size))
        print ('- Number of training batches: {}'.format(self.num_batches_train))
        print ('- Number of evaluation batches: {}'.format(self.num_batches_eval))

    def next_batch_train(self):
        '''
        Function to get the next batch of points
        '''
        x_batch = []
       
        for i in range(self.batch_size):
            
            idx = self.data_pointer_train
            
            assert idx + self.seq_length_p1 + 1 < self.num_data_train
            
            seq_x = self.dataset_train[idx:idx+self.seq_length_p1]
            
            x_batch.append(seq_x)
                
            self.data_pointer_train += self.seq_length_p1

        return x_batch
    
    def next_batch_eval(self):
        '''
        Function to get the next batch of points
        '''
        x_batch = []

        for i in range(self.batch_size):
            
            idx = self.data_pointer_eval
            
            assert idx + self.seq_length_p1 + 1 < self.num_data_eval
            
            seq_x = self.dataset_eval[idx:idx+self.seq_length_p1]
            
            x_batch.append(seq_x)
            
            self.data_pointer_eval += self.seq_length_p1

        return x_batch