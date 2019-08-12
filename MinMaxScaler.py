# -*- coding: utf-8 -*-

# author: Youngjoo Kim

import numpy as np

class MinMaxScaler(object):
    def __init__(self):
        
        self.isFit = False        
         
    def fit(self,data,feature_range=(-1,1), Max = None, Min = None):
        
        self.Max = Max
        self.Min = Min
        
        if Max == None:
            self.Max = np.max(data)
        if Min == None:
            self.Min = np.min(data)
        self.feature_range = feature_range
        self.isFit = True
    
    def scale(self, data):
    
        assert self.Max is not None
        assert self.isFit is True
        scaled = data
        scaled = (data - self.Min) * (self.feature_range[1] - self.feature_range[0]) / (self.Max - self.Min) + self.feature_range[0]
    
        return scaled
    
    def scale_inverse(self, scaled):

        assert self.Max is not None
        assert self.isFit is True
        data = scaled
        data = (scaled - self.feature_range[0]) * (self.Max - self.Min) / (self.feature_range[1] - self.feature_range[0]) + self.Min

        return data