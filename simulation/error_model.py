import numpy as np
from .model_output import SEIRModelOutput


class NaiveErrorModel():
    def __init__(self, model_output: SEIRModelOutput):
        self.model_output = model_output
        self.under_reporting_scale_arr = None
        self.delay_time_arr = None
    
    def call(self):
        self.model_output.weekly_incidence = np.dot(self.model_output.weekly_incidence, 
                                                    self.under_reporting_scale_arr)