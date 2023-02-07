import os
import pandas as pd
import numpy as np
from epstein_civil_violence.model import EpsteinCivilViolence
from mesa.batchrunner import FixedBatchRunner
from itertools import product

from mesa import DataCollector

fixed_parameters = {
    'width':40,
    'height':40,
    'citizen_density':0.7,
    'cop_density':0.03,
    'citizen_vision':7,
    'cop_vision':7,
    'legitimacy':0.8,
    'max_jail_term':30,
    'active_threshold':0.1,
    'arrest_prob_constant':2.3,
    'movement':True,
    'max_iters':200,
}

params = {
    'activation_type': ["default","linear","quadratic","logistic"],
    'min_proportion': [*np.arange(0.5,1,0.1)]
}

def dict_product(dicts): 
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))

parameters_list = [*dict_product(params)]

batch_run = FixedBatchRunner(EpsteinCivilViolence, parameters_list,
                             fixed_parameters,
                             iterations=10,
                             max_steps=200,
                             model_reporters={
                                "Quiescent": lambda m: m.count_quiescent(m),
                                "Active": lambda m: m.count_active(m),
                                "Jailed": lambda m: m.count_jailed(m),
                                "Average Jailing Term": lambda m: m.get_average_jail_term(m)
                             },
                             )


batch_run.run_all()
batch_end = batch_run.get_model_vars_dataframe()
batch_step_raw = batch_run.get_collector_model()

batch_end.to_csv("output/batch_all.csv")

for key,df in batch_step_raw.items():
    df.to_csv(f'step/activation_{key[0]}_tau_{key[1]}_{key[2]}.csv')