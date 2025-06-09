import math
import scipy
import numpy as np

from .model_output import SEIRModelOutput, SEIRParams
from sklearn.metrics import r2_score


class SEIRModel():
    def __init__(self, population: int):
        self.population = population
        
        # FOLLOWING PARAMETERS ARE EPIDEMICALLY DETERMINED
        # R_0 HERE LIES IN RANGE [1; 2.5]
        self.min_params = SEIRParams(alpha=1/5, beta=1/9, gamma=1/9, init_inf_frac=1e-6, init_rec_frac=1e-2)
        self.max_params = SEIRParams(alpha=1, beta=0.625, gamma=1/4, init_inf_frac=1e-3, init_rec_frac=2e-1)
        self.last_sim_params = None
        
    def __deriv(self, y, t, alpha, beta, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I
        dEdt = beta * S * I - alpha * E
        dIdt = alpha * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def simulate(self, alpha=1/2, beta=1/7*1.5, gamma=1/7, init_inf_frac=1e-4, init_rec_frac=0.15, tmax: int = 150):
        '''
        alpha: rate of progression from exposed to infectious
        beta: transmission rate
        gamma: recovery rate
        init_inf_frac: fraction of initially infected
        init_rec_frac: fraction of initially recovered
        '''
        E0 = 0
        I0 = init_inf_frac
        R0 = init_rec_frac
        S0 = 1 - I0 - R0
        y0 = S0, E0, I0, R0
        t = np.linspace(0, tmax, tmax)
        S, E, I, R = scipy.integrate.odeint(self.__deriv, y0, t,
                                     args=(alpha, beta, gamma)).T * self.population
        self.result = SEIRModelOutput(t, S, E, I, R)
        self.last_sim_params = SEIRParams(alpha, beta, gamma, init_inf_frac, init_rec_frac)
        return self.result
    
    def calibrate(self, time_series):
        '''
        
        return: SEIRParams object
        '''
        tmax = len(time_series)
        not_none_value_indices = [i for i, x in enumerate(time_series) if x is not np.nan]
        def AnnealingModel(x):
            alpha, beta, gamma, init_inf_frac, init_rec_frac = x
            sim = self.simulate(alpha=alpha, beta=beta, gamma=gamma, 
                                init_inf_frac=init_inf_frac, 
                                init_rec_frac=init_rec_frac, 
                                tmax=tmax)
            daily_incidence_sim = sim.daily_incidence
            # ax.plot(daily_incidence_sim, color='RoyalBlue', alpha=0.3)
            return -r2_score(np.array(daily_incidence_sim)[not_none_value_indices], 
                            np.array(time_series)[not_none_value_indices])
            
        lw = [self.min_params.alpha, self.min_params.beta, self.min_params.gamma, 
              self.min_params.init_inf_frac, self.min_params.init_rec_frac]
        up = [self.max_params.alpha, self.max_params.beta, self.max_params.gamma, 
              self.max_params.init_inf_frac, self.max_params.init_rec_frac]
        
        ret = scipy.optimize.dual_annealing(AnnealingModel, bounds=list(zip(lw, up)))
        
        best_params = SEIRParams(*ret.x, tmax)
        return best_params, -ret.fun 
    
    def calculate_rel_error(self, true_params: SEIRParams, estimated_params: SEIRParams):
        true_params_arr = np.array(true_params.as_list())
        estimated_params_arr = np.array(estimated_params.as_list())
        return np.abs(true_params_arr - estimated_params_arr)/true_params_arr
    