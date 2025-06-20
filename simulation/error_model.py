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
        self.mean_delay = mean_delay
        self.mean_underreporting = mean_underreporting
        self.daily_incidence = time_series
        self.error_mode = error_mode
        
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
    def generate_random_delay(self, mean_delay, mode='geom', size=None):
        if size is None:
            size = self.tmax
        if mode == 'geom':
            if mean_delay == 0:
                return np.zeros(size)
            else:
                return scipy.stats.geom.rvs(1/mean_delay, size=size)
        else:
            raise Exception('Not implemented!')

    def generate_random_underreporting(self, mean_underreporting, mode='uniform', size=None):
        if size is None:
            size = self.tmax
        if mode == 'uniform':
            return scipy.stats.uniform.rvs(loc=mean_underreporting, scale=0.02, size=size)
        else:
            raise Exception('Not implemented!')

    # ADDING ERROR TO THE INCIDENCE TIME SERIES
    def add_delay(self):
        self.new_indices = [int(self.timespace[index] + self.delay_arr[index]) for index in range(self.tmax)]
        dict_incidence = dict()
        for index in range(len(self.new_indices)):
            if self.new_indices[index] in dict_incidence.keys():
                dict_incidence[self.new_indices[index]
                               ] += self.daily_incidence[index]
            elif self.new_indices[index] is not np.nan:
                dict_incidence[self.new_indices[index]] = self.daily_incidence[index]
        dict_incidence = dict(sorted(dict_incidence.items()))

        delayed_incidence = [np.nan for _ in range(int(max(self.new_indices)+1))]
        for key in dict_incidence.keys():
            delayed_incidence[key] = dict_incidence[key]
        # assert sum(filter(np.nan, self.incidence_arr)) == sum(filter(np.nan, delayed_incidence))
        self.daily_incidence = delayed_incidence

    def add_underreporting(self):
        self.daily_incidence = [self.daily_incidence[index] *
                              self.underreporting_arr[index] for index in range(self.tmax)]
        
    
    # SUBTRACT DELAY (INVERSE OPERATION TO ADD DELAY)
    def subtract_delay(self, mean_delay=None):
        '''
        This function makes inverse operation to 'add_delay'
        Parameters
        ----------
        mean_delay: expected value for delay in days for error_mode='random' and
        constant value in days for error_mode='fixed'
        '''
        delay_arr = None
        if mean_delay is None:
            mean_delay = self.mean_delay
        if self.error_mode == 'random':
            delay_arr = self.generate_random_delay(mean_delay)
        elif self.error_mode == 'fixed':
            delay_arr = self.generate_fixed_delay(mean_delay)
        else:
            raise Exception('Choose correct mode for error: "random" or "fixed".')
        
        
        self.new_indices = [int(self.timespace[index] - delay_arr[index]) if (self.timespace[index] >= delay_arr[index])
                            else np.nan for index in range(self.tmax)]

        dict_incidence = dict()
        for index in range(self.tmax):
            if self.new_indices[index] in dict_incidence.keys():
                dict_incidence[self.new_indices[index]
                               ] += self.daily_incidence[index]
            elif self.new_indices[index] is not np.nan:
                dict_incidence[self.new_indices[index]
                               ] = self.daily_incidence[index]
        dict_incidence = dict(sorted(dict_incidence.items()))

        delayed_incidence = [np.nan for _ in range(self.tmax)]
        for key in dict_incidence.keys():
            delayed_incidence[key] = dict_incidence[key]
        self.daily_incidence = delayed_incidence

    def add_noise(self):
        self.add_underreporting()
        self.add_delay()
        self.calculate_weekly_incidence()
        
  
    def pad_array_to_multiple_of_seven(self, arr):
        '''
        Auxiliary function used for padding array of daily data by zeroes for converting
        to weekly data.
        '''
        current_size = len(arr)
        new_size = (current_size + 6) // 7 * 7
        padding_needed = new_size - current_size
        padded_array = np.pad(arr, (0, padding_needed),
                            mode='constant', constant_values=0)
        return padded_array

    def calculate_weekly_incidence(self):
        '''
        Calculates weekly newly infected cases using daily incidence data.
        '''
        daily_incidence_padded = self.pad_array_to_multiple_of_seven(self.daily_incidence)
        self.weekly_incidence = np.nansum(daily_incidence_padded.reshape(-1, 7), axis=1)