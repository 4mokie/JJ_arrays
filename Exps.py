
from typing import List, Tuple, Dict
from collections.abc import Iterable 
from collections import defaultdict

import matplotlib as mpt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches

import numpy as np
from scipy import optimize
import os

from JJ_data_processing import *
from JJformulas import *

#from tqdm import tqdm, tqdm_notebook
from tqdm.autonotebook import tqdm

from scipy.optimize import curve_fit


import qcodes as qc
from qcodes.dataset.database import initialise_database
from qcodes.dataset.plotting import plot_by_id, get_data_by_id

import pandas as pd

import pprint
import json




class Exps():
    

    
    def __init__(self, sample, folder):
        

    
        #keys ids Bs coss T comm  
        self.keys = ['ids', 'B', 'cos', 'T', 'comm']
        
        self.sample = sample
        self.folder = folder
        
        self.exp_connect()

        self.ids = dict()
        
                    

        
    # self.tabe = { 'id' : {  'B' : Bval , },  }    
    def load_old(self, list_of_exp_dict):
        
        
        table =  dict()
        
        for exp in list_of_exp_dict:
                
            for j, idx in enumerate(exp['ids']):
                
                
                id_param = dict()
         
                for key, val in exp.items():

                    if isinstance(val, Iterable) and not isinstance(val, str):

                        id_param[key] =  val[j] 

                    else:

                        id_param[key] =  val
                        
#                 if 'cos' in exp.keys():
#                     new_line_exp['B'] = self._get_B (idx)
#                 elif 'B' in exp.keys():
#                     new_line_exp['cos'] = self._get_cos (idx)
                    
                table[idx] = id_param
                    
        path = 'file.json'            
        
        self.table= table
        
        with open(path, "w") as write_file:

            json.dump(table, write_file)
                        

    def exp_connect(self):

#         path = self.folder + 'exps_table{}.json'.format(self.sample) 
        path = 'file.json'
        try:
            with open(path, "r") as read_file:
                data = json.load( read_file)
        except FileNotFoundError:
            with open(path, "w") as write_file:
                data = {0 : ''}
                json.dump(data, write_file)
        
        self.table = data
        
    def _get_cos(self, idx):
        
        exp = self.ids[idx]
        cos = np.cos(np.pi*(exp['B'] - exp['ZF'] )/2/ (exp['FF'] - exp['ZF']))
        return  abs( cos )#np.round(abs( cos ), decimals = 4)

    
    def _get_B(self, idx):
        
        exp = self.ids[idx]
        B = 2*(exp['FF'] - exp['ZF'])*np.arccos(exp['cos'])/np.pi + exp['ZF']

        return  abs(B)#np.round(abs(B), decimals = 4)

    def show_all(self):
        
        self.exps_topd = list()
        for exp in self.exps:
            
            exp_topd = dict()
            for key, val in exp.items():
                if isinstance(val, Iterable) and not isinstance(val, str):
                    exp_topd[key] = '{}..{}'.format(eng_string(val[0]), eng_string(val[-1])  )
                else:
                    exp_topd[key] = '{}'.format(eng_string(val) )
                    
            self.exps_topd.append(exp_topd)
                
        df = pd.DataFrame(self.exps_topd)
#         print(df)
        return df
  
            


    def find_by_single_cond( self, param, value ) :

        def isclose_float_or_str(a,b, atol = 1e-8):
            if isinstance(a, str) or isinstance(b, str) :
                if  (a in b)  :
                    return True
            else:
                return np.isclose(a, b, atol = atol)

        cond_ids = set(idx for idx in self.table.keys() if isclose_float_or_str(value, self.table[idx][param]) )


        if len(cond_ids) == 0:   # if there's no exact matching - find closest
            
            
            all_vals = np.array([ exp[param] for exp in self.table.values() ] )

            atol = min( abs( all_vals - value ) )
            cond_ids = set(idx for idx in self.table.keys() 
                           if isclose_float_or_str(self.table[idx][param], value, atol = atol) )
        
        return cond_ids
    
    
     
    def find(self, which ) :  #like ['T':.1, 'cos':[0.1, 0.2]]
        
        def make_iterable( val ):
            
            if isinstance(val, Iterable) and not isinstance(val, str):   # if cannot iterate - make iterable
                vals = val 
            else:
                vals = list([val])
            return vals
        
        
#         self.db_connect()
            
        list_cond_ids = []

        for wh_k, wh_v in which.items():
            
            wh_vs = make_iterable( wh_v )
            
            union_cond_ids = set()
            
            for wh_v_ in wh_vs:
                
                union_cond_ids = union_cond_ids.union(self.find_by_single_cond( wh_k, wh_v_ ))
                    
            list_cond_ids.append( union_cond_ids ) 
            
        out = set.intersection(*list_cond_ids)
            
        return  out
    
    
    
    def _make_label(self, idx, which):
        lab = ''
        for key in which.keys():
            
            val = self.table[idx][key]
            if not isinstance(val, str):
                val = eng_string(val)
#                 val = '{:.2e}'.format(val)
            lab += ' {:s} = {:s};'.format(key, val)
            
        return lab
    
        
    
    def plot(self, which, ax = None, N = None,  **kw ):
        
        if  ax is None:
            fig, ax = plt.subplots()
            
        ids = self.find(which)
        
        for idx in ids:
            
            lab = self._make_label(idx, which)
            
            if N is None:

                plot_by_id(idx, axes = ax, label = lab ,  **kw)
            else:
                I, V = self.get_Is(idx)
                ax.plot(I, V, label = lab ,  **kw)
                
        ax.legend()
        
        return ax
    
    def get_Is(self, idx,  dy = 300e-6, Voff = -0.55e-3):
        
        I, V = xy_by_id(idx)
        V-= Voff
        I, V = cut_dxdy(I, V, dx = 50e-9 ,dy = dy)
        
        return I,V+Voff
        
        
    
    def get_param( self, param, which, **kw  ):
        
        func_dict = {'R0' : get_R0}
        
        ids = self.find(which)
        results = []
        for idx in ids:
            
            x,y = self.get_Is(idx)
            results.append(  func_dict[param] (x,y, **kw) ) 
        
        return np.array(results)
        
        