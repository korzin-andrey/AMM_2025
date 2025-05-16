import numpy as np
import scipy
from .model_output import SEIRModelOutput


class NaiveErrorModel():
    def __init__(self, time_series: np.array,
                 mean_delay: float = 1/20,
                 mean_underreporting: float = 1/10):
        '''
        Takes time series array as input and add noise to data: delay and underreporting.
        The resulting
        '''
        self.tmax = len(time_series)
        self.timespace = np.arange(self.tmax)
        self.incidence_arr = time_series
        self.delay_arr = self.generate_delay(mean_delay,
                                             mode='geom')
        self.underreporting_arr = self.generate_underreporting(mean_underreporting,
                                                               mode='uniform')

    def generate_delay(self, mean_delay, mode='geom'):
        if mode == 'geom':
            return scipy.stats.geom.rvs(mean_delay, size=self.tmax)
        else:
            raise Exception('Not implemented!')

    def generate_underreporting(self, mean_underreporting, mode='uniform'):
        if mode == 'uniform':
            return scipy.stats.uniform.rvs(loc=mean_underreporting, scale=0.02, size=self.tmax)
        else:
            raise Exception('Not implemented!')

    def add_delay(self):
        self.new_indices = [(self.timespace[index] + self.delay_arr[index]) if (self.timespace[index] + self.delay_arr[index]) < self.tmax
                            else None for index in range(self.tmax)]

        dict_incidence = dict()
        for index in range(len(self.incidence_arr)):
            if self.new_indices[index] in dict_incidence.keys():
                dict_incidence[self.new_indices[index]
                               ] += self.incidence_arr[index]
            elif self.new_indices[index] is not None:
                dict_incidence[self.new_indices[index]
                               ] = self.incidence_arr[index]
        dict_incidence = dict(sorted(dict_incidence.items()))

        delayed_incidence = [None for _ in range(self.tmax)]
        for key in dict_incidence.keys():
            delayed_incidence[key] = dict_incidence[key]
        # does not work because all incidence data with time index > self.tmax is deleted
        # assert sum(filter(None, self.incidence_arr)) == sum(filter(None, delayed_incidence))
        self.incidence_arr = delayed_incidence
        
    
    def remove_delay(self):
        self.new_indices = [(self.timespace[index] - self.delay_arr[index]) if (self.timespace[index] >= self.delay_arr[index]) 
                            else None for index in range(self.tmax)]

        dict_incidence = dict()
        for index in range(len(self.incidence_arr)):
            if self.new_indices[index] in dict_incidence.keys():
                dict_incidence[self.new_indices[index]
                               ] += self.incidence_arr[index]
            elif self.new_indices[index] is not None:
                dict_incidence[self.new_indices[index]
                               ] = self.incidence_arr[index]
        dict_incidence = dict(sorted(dict_incidence.items()))

        delayed_incidence = [None for _ in range(self.tmax)]
        for key in dict_incidence.keys():
            delayed_incidence[key] = dict_incidence[key]
        # does not work because all incidence data with time index > self.tmax is deleted
        # assert sum(filter(None, self.incidence_arr)) == sum(filter(None, delayed_incidence))
        self.incidence_arr = delayed_incidence

    def add_underreporting(self):
        self.incidence_arr = [self.incidence_arr[index] *
                              self.underreporting_arr[index] for index in range(self.tmax)]

    def add_noise(self):
        self.add_underreporting()
        self.add_delay()
