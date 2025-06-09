import numpy as np
import scipy
from .model_output import SEIRModelOutput


class NaiveErrorModel():
    def __init__(self, time_series: np.array,
                 mean_delay: float = 1/20,
                 mean_underreporting: float = 1/10, 
                 error_mode = 'random'):
        '''
        Takes time series array as input and add noise to data: delay and underreporting.
        The resulting
        '''
        self.tmax = len(time_series)
        self.timespace = np.arange(self.tmax)
        self.incidence_arr = time_series
        
        if error_mode == 'random':
            self.delay_arr = self.generate_random_delay(mean_delay)
            self.underreporting_arr = self.generate_random_underreporting(mean_underreporting)
        elif error_mode == 'fixed':
            self.delay_arr = self.generate_fixed_delay(mean_delay)
            self.underreporting_arr = self.generate_fixed_underreporting(mean_underreporting)
        else:
            raise Exception('Choose correct mode for error: "random" or "fixed".')


    # FIXED ERROR GENERATION
    def generate_fixed_delay(self, delay):
        return [delay for _ in range(self.tmax)]
    
    def generate_fixed_underreporting(self, underreporting):
        return[underreporting for _ in range(self.tmax)]
    
    # RANDOM ERROR GENERATION
    def generate_random_delay(self, mean_delay, mode='geom'):
        if mode == 'geom':
            return scipy.stats.geom.rvs(1/mean_delay, size=self.tmax)
        else:
            raise Exception('Not implemented!')

    def generate_random_underreporting(self, mean_underreporting, mode='uniform'):
        if mode == 'uniform':
            return scipy.stats.uniform.rvs(loc=mean_underreporting, scale=0.02, size=self.tmax)
        else:
            raise Exception('Not implemented!')

    # ADDING ERROR TO THE INCIDENCE TIME SERIES
    def add_delay(self):
        self.new_indices = [(self.timespace[index] + self.delay_arr[index]) if (self.timespace[index] + self.delay_arr[index]) < self.tmax
                            else np.nan for index in range(self.tmax)]

        dict_incidence = dict()
        for index in range(len(self.incidence_arr)):
            if self.new_indices[index] in dict_incidence.keys():
                dict_incidence[self.new_indices[index]
                               ] += self.incidence_arr[index]
            elif self.new_indices[index] is not np.nan:
                dict_incidence[self.new_indices[index]
                               ] = self.incidence_arr[index]
        dict_incidence = dict(sorted(dict_incidence.items()))

        delayed_incidence = [np.nan for _ in range(self.tmax)]
        for key in dict_incidence.keys():
            delayed_incidence[key] = dict_incidence[key]
        # does not work because all incidence data with time index > self.tmax is deleted
        # assert sum(filter(np.nan, self.incidence_arr)) == sum(filter(np.nan, delayed_incidence))
        self.incidence_arr = delayed_incidence

    def add_underreporting(self):
        self.incidence_arr = [self.incidence_arr[index] *
                              self.underreporting_arr[index] for index in range(self.tmax)]
        
    def remove_random_delay(self):
        self.new_indices = [(self.timespace[index] - self.delay_arr[index]) if (self.timespace[index] >= self.delay_arr[index])
                            else np.nan for index in range(self.tmax)]

        dict_incidence = dict()
        for index in range(len(self.incidence_arr)):
            if self.new_indices[index] in dict_incidence.keys():
                dict_incidence[self.new_indices[index]
                               ] += self.incidence_arr[index]
            elif self.new_indices[index] is not np.nan:
                dict_incidence[self.new_indices[index]
                               ] = self.incidence_arr[index]
        dict_incidence = dict(sorted(dict_incidence.items()))

        delayed_incidence = [np.nan for _ in range(self.tmax)]
        for key in dict_incidence.keys():
            delayed_incidence[key] = dict_incidence[key]
        # does not work because all incidence data with time index > self.tmax is deleted
        # assert sum(filter(np.nan, self.incidence_arr)) == sum(filter(np.nan, delayed_incidence))
        self.incidence_arr = delayed_incidence

    def add_noise(self):
        self.add_underreporting()
        self.add_delay()
