import math
import scipy
import numpy as np

from .model_output import SEIRModelOutput


class SEIRModel():
    def __init__(self, population: int):
        self.population = population

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
        return self.result
    