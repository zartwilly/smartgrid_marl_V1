# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 08:21:09 2021

@author: jwehounou
"""

import os
import time
import numpy as np
import pandas as pd
import itertools as it

import fonctions_auxiliaires as fct_aux

from bokeh.models.tools import HoverTool, PanTool, BoxZoomTool, WheelZoomTool 
from bokeh.models.tools import RedoTool, ResetTool, SaveTool, UndoTool
from bokeh.models import ColumnDataSource, FactorRange
from bokeh.models import Band
from bokeh.plotting import figure, show, output_file, save
from bokeh.layouts import row, column
from bokeh.models import Panel, Tabs, Legend
from bokeh.transform import factor_cmap
from bokeh.transform import dodge

# from bokeh.models import Select
# from bokeh.io import curdoc
# from bokeh.plotting import reset_output
# from bokeh.models.widgets import Slider


# Importing a pallette
from bokeh.palettes import Category20
#from bokeh.palettes import Spectral5 
from bokeh.palettes import Viridis256


from bokeh.models.annotations import Title

#------------------------------------------------------------------------------
#                   definitions of constants
#------------------------------------------------------------------------------
WIDTH = 500;
HEIGHT = 500;
MULT_WIDTH = 2.5;
MULT_HEIGHT = 3.5;

MARKERS = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", 
               "P", "*", "h", "H", "+", "x", "X", "D", "d"]
COLORS = Category20[19] #["red", "yellow", "blue", "green", "rosybrown","darkorange", "fuchsia", "grey", ]

TOOLS = [
            PanTool(),
            BoxZoomTool(),
            WheelZoomTool(),
            UndoTool(),
            RedoTool(),
            ResetTool(),
            SaveTool(),
            HoverTool(tooltips=[
                ("Price", "$y"),
                ("Time", "$x")
                ])
            ]

NAME_RESULT_SHOW_VARS = "resultat_show_variables_pi_plus_{}_pi_minus_{}.html"

#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------

# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> debut
# _____________________________________________________________________________
def get_local_storage_variables(path_to_variable):
    """
    obtain the content of variables stored locally .

    Returns
    -------
     arr_pls_M_T, RUs, B0s, C0s, BENs, CSTs, pi_sg_plus_s, pi_sg_minus_s.
    
    arr_pls_M_T: array of players with a shape M_PLAYERS*T_PERIODS*INDEX_ATTRS
    arr_T_nsteps_vars : array of players with a shape 
                        M_PLAYERS*T_PERIODS*NSTEPS*vars_nstep
                        avec len(vars_nstep)=20
    RUs: array of (M_PLAYERS,)
    BENs: array of M_PLAYERS*T_PERIODS
    CSTs: array of M_PLAYERS*T_PERIODS
    B0s: array of (T_PERIODS,)
    C0s: array of (T_PERIODS,)
    pi_sg_plus_s: array of (T_PERIODS,)
    pi_sg_minus_s: array of (T_PERIODS,)

    pi_hp_plus_s: array of (T_PERIODS,)
    pi_hp_minus_s: array of (T_PERIODS,)
    """

    arr_pl_M_T_K_vars = np.load(os.path.join(path_to_variable, 
                                             "arr_pl_M_T_K_vars.npy"),
                          allow_pickle=True)
    b0_s_T_K = np.load(os.path.join(path_to_variable, "b0_s_T_K.npy"),
                          allow_pickle=True)
    c0_s_T_K = np.load(os.path.join(path_to_variable, "c0_s_T_K.npy"),
                          allow_pickle=True)
    B_is_M = np.load(os.path.join(path_to_variable, "B_is_M.npy"),
                          allow_pickle=True)
    C_is_M = np.load(os.path.join(path_to_variable, "C_is_M.npy"),
                          allow_pickle=True)
    BENs_M_T_K = np.load(os.path.join(path_to_variable, "BENs_M_T_K.npy"),
                          allow_pickle=True)
    CSTs_M_T_K = np.load(os.path.join(path_to_variable, "CSTs_M_T_K.npy"),
                          allow_pickle=True)
    BB_is_M = np.load(os.path.join(path_to_variable, "BB_is_M.npy"),
                          allow_pickle=True)
    CC_is_M = np.load(os.path.join(path_to_variable, "CC_is_M.npy"),
                          allow_pickle=True)
    RU_is_M = np.load(os.path.join(path_to_variable, "RU_is_M.npy"),
                          allow_pickle=True)
    pi_sg_plus_T_K = np.load(os.path.join(path_to_variable, "pi_sg_plus_T_K.npy"),
                          allow_pickle=True)
    pi_sg_minus_T_K = np.load(os.path.join(path_to_variable, "pi_sg_minus_T_K.npy"),
                          allow_pickle=True)
    pi_0_plus_T_K = np.load(os.path.join(path_to_variable, "pi_0_plus_T_K.npy"),
                          allow_pickle=True)
    pi_0_minus_T_K = np.load(os.path.join(path_to_variable, "pi_0_minus_T_K.npy"),
                          allow_pickle=True)
    pi_hp_plus_T = np.load(os.path.join(path_to_variable, "pi_hp_plus_T.npy"),
                          allow_pickle=True)
    pi_hp_minus_T = np.load(os.path.join(path_to_variable, "pi_hp_minus_T.npy"),
                          allow_pickle=True)
    
    return arr_pl_M_T_K_vars, \
            b0_s_T_K, c0_s_T_K, \
            B_is_M, C_is_M, \
            BENs_M_T_K, CSTs_M_T_K, \
            BB_is_M, CC_is_M, RU_is_M, \
            pi_sg_plus_T_K, pi_sg_minus_T_K, \
            pi_0_plus_T_K, pi_0_minus_T_K, \
            pi_hp_plus_T, pi_hp_minus_T
            
def get_tuple_paths_of_arrays(name_dir="tests", name_simu="simu_1811_1754",
                prices=None, algos=None, learning_rates=None, 
                algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                     "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE", 
                                       "MIDDLE-BRUTE-FORCE"], 
                algos_4_showing=["DETERMINIST", "LRI1", "LRI2",
                                 "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE"],
                ext=".npy", 
                exclude_html_files=[NAME_RESULT_SHOW_VARS,"html"]):
    
    tuple_paths = []
    path_2_best_learning_steps = []
    rep_dir_simu = os.path.join(name_dir, name_simu)
    
    prices_new = None
    if prices is None:
        prices_new = os.listdir(rep_dir_simu)
    else:
        prices_new = prices
    prices_new = [x for x in prices_new 
                     if x.split('.')[-1] not in exclude_html_files]
    for price in prices_new:
        path_price = os.path.join(name_dir, name_simu, price)
        algos_new = None
        if algos is None:
            algos_new = os.listdir(path_price)
        else:
            algos_new = algos
        for algo in algos_new:
            if algo in algos_4_showing:
                path_price_algo = os.path.join(name_dir, name_simu, price, algo)
                if algo not in algos_4_no_learning:
                    learning_rates_new = None
                    if learning_rates is None:
                        learning_rates_new = os.listdir(path_price_algo)
                    else:
                        learning_rates_new = learning_rates
                    for learning_rate in learning_rates_new:
                        tuple_paths.append( (name_dir, name_simu, price, 
                                             algo, learning_rate) )
                        if algo is not algos_4_no_learning:
                            path_2_best_learning_steps.append(
                                (name_dir, name_simu, price, 
                                 algo, learning_rate))
                else:
                    tuple_paths.append( (name_dir, name_simu, price, algo) )
                
    return tuple_paths, prices_new, algos_new, \
            learning_rates_new, path_2_best_learning_steps
            
def get_k_stop_4_periods(path_2_best_learning_steps):
    """
     determine the upper k_stop from algos LRI1 and LRI2 for each period

    Parameters
    ----------
    path_2_best_learning_steps : Tuple
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df_LRI_12 = None #pd.DataFrame()
    for tuple_path_2_algo in path_2_best_learning_steps:
        path_2_algo = os.path.join(*tuple_path_2_algo)
        algo = tuple_path_2_algo[3]
        df_al = pd.read_csv(
                    os.path.join(path_2_algo, "best_learning_steps.csv"),
                    index_col=0)
        index_mapper = {"k_stop":algo+"_k_stop"}
        df_al.rename(index=index_mapper, inplace=True)
        if df_LRI_12 is None:
            df_LRI_12 = df_al
        else:
            df_LRI_12 = pd.concat([df_LRI_12, df_al], axis=0)
            
    cols = df_LRI_12.columns.tolist()
    indices = df_LRI_12.index.tolist()
    df_k_stop = pd.DataFrame(columns=cols, index=["k_stop"])
    for col in cols:
        best_index = None
        for index in indices:
            if best_index is None:
                best_index = index
            elif df_LRI_12.loc[best_index, col] < df_LRI_12.loc[index, col]:
                best_index = index
        df_k_stop.loc["k_stop", col] = df_LRI_12.loc[best_index, col]
        
    return df_LRI_12, df_k_stop

def get_array_turn_df_for_t_BON(tuple_paths, t=1, k_steps_args=250, 
                            algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                                 "BEST-BRUTE-FORCE",
                                                 "BAD-BRUTE-FORCE", 
                                                 "MIDDLE-BRUTE-FORCE"], 
                            algos_4_learning=["LRI1", "LRI2"]):
    df_arr_M_T_Ks = []
    df_b0_c0_pisg_pi0_T_K = []
    df_B_C_BB_CC_RU_M = []
    df_ben_cst_M_T_K = []
    for tuple_path in tuple_paths:
        path_to_variable = os.path.join(*tuple_path)
        
        # arr_pl_M_T_K_vars, \
        # b0_s_T_K, c0_s_T_K, \
        # B_is_M, C_is_M, \
        # BENs_M_T_K, CSTs_M_T_K, \
        # BB_is_M, CC_is_M, RU_is_M, \
        # pi_sg_plus_T, pi_sg_minus_T, \
        # pi_0_plus_T, pi_0_minus_T, \
        # pi_hp_plus_s, pi_hp_minus_s \
        #     = get_local_storage_variables(path_to_variable)
        
        arr_pl_M_T_K_vars, \
        b0_s_T_K, c0_s_T_K, \
        B_is_M, C_is_M, B_is_M_T, C_is_M_T,\
        BENs_M_T_K, CSTs_M_T_K, \
        BB_is_M, CC_is_M, RU_is_M, BB_is_M_T, CC_is_M_T, RU_is_M_T,\
        pi_sg_plus_T, pi_sg_minus_T, \
        pi_0_plus_T, pi_0_minus_T, \
        pi_hp_plus_T, pi_hp_minus_T \
            = fct_aux.get_local_storage_variables(
                path_to_variable=path_to_variable)
        
        price = tuple_path[2].split("_")[3]+"_"+tuple_path[2].split("_")[-1]
        algo = tuple_path[3];
        rate = tuple_path[4] if algo in algos_4_learning else 0
        
        m_players = arr_pl_M_T_K_vars.shape[0]
        t_periods = arr_pl_M_T_K_vars.shape[1]
        k_steps = arr_pl_M_T_K_vars.shape[2] if arr_pl_M_T_K_vars.shape == 4 \
                                             else k_steps_args                                    
        #for t in range(0, t_periods):                                     
        t_periods = None; tu_mtk = None; tu_tk = None; tu_m = None
        if t is None:
            t_periods = arr_pl_M_T_K_vars.shape[1]
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     range(0, t_periods), 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    range(0, t_periods), 
                                    range(0, k_steps)))
            t_periods = list(range(0, t_periods))
        elif type(t) is list:
            t_periods = t
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    t_periods, 
                                    range(0, k_steps)))
        elif type(t) is int:
            t_periods = [t]
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    t_periods, 
                                    range(0, k_steps)))
                      
        print('t_periods = {}'.format(t_periods))
        tu_m = list(it.product([algo], [rate], [price], range(0, m_players)))
                    
        variables = list(fct_aux.AUTOMATE_INDEX_ATTRS.keys())
        
        if algo in algos_4_learning:
            arr_pl_M_T_K_vars_t = arr_pl_M_T_K_vars[:, t_periods, :, :]
            ## process of arr_pl_M_T_K_vars 
            arr_pl_M_T_K_vars_2D = arr_pl_M_T_K_vars_t.reshape(
                                        -1, 
                                        arr_pl_M_T_K_vars.shape[3])
            df_lri_x = pd.DataFrame(data=arr_pl_M_T_K_vars_2D, 
                                    index=tu_mtk, 
                                    columns=variables)
            
            df_arr_M_T_Ks.append(df_lri_x)
            
            ## process of df_b0_c0_pisg_pi0_T_K
            b0_s_T_K_2D = []
            c0_s_T_K_2D = []
            pi_0_minus_T_K_2D = []
            pi_0_plus_T_K_2D = []
            pi_sg_minus_T_K_2D = []
            pi_sg_plus_T_K_2D = []
            for tx in t_periods:
                b0_s_T_K_2D.append(list( b0_s_T_K[tx,:].reshape(-1)))
                c0_s_T_K_2D.append(list( c0_s_T_K[tx,:].reshape(-1)))
                pi_0_minus_T_K_2D.append([ pi_0_minus_T[tx] ]*k_steps_args)
                pi_0_plus_T_K_2D.append([ pi_0_plus_T[tx] ]*k_steps_args)
                pi_sg_minus_T_K_2D.append([ pi_sg_minus_T[tx] ]*k_steps_args)
                pi_sg_plus_T_K_2D.append([ pi_sg_plus_T[tx] ]*k_steps_args)
            b0_s_T_K_2D = np.array(b0_s_T_K_2D, dtype=object)
            c0_s_T_K_2D = np.array(c0_s_T_K_2D, dtype=object)
            pi_0_minus_T_K_2D = np.array(pi_0_minus_T_K_2D, dtype=object)
            pi_0_plus_T_K_2D = np.array(pi_0_plus_T_K_2D, dtype=object)
            pi_sg_minus_T_K_2D = np.array(pi_sg_minus_T_K_2D, dtype=object)
            pi_sg_plus_T_K_2D = np.array(pi_sg_plus_T_K_2D, dtype=object)
            
            b0_s_T_K_1D = b0_s_T_K_2D.reshape(-1)
            c0_s_T_K_1D = c0_s_T_K_2D.reshape(-1)
            pi_0_minus_T_K_1D = pi_0_minus_T_K_2D.reshape(-1)
            pi_0_plus_T_K_1D = pi_0_plus_T_K_2D.reshape(-1)
            pi_sg_minus_T_K_1D = pi_sg_minus_T_K_2D.reshape(-1)
            pi_sg_plus_T_K_1D = pi_sg_plus_T_K_2D.reshape(-1)
            
            df_b0_c0_pisg_pi0_T_K_lri \
                = pd.DataFrame({
                        "b0":b0_s_T_K_1D, "c0":c0_s_T_K_1D, 
                        "pi_0_minus":pi_0_minus_T_K_1D, 
                        "pi_0_plus":pi_0_plus_T_K_1D, 
                        "pi_sg_minus":pi_sg_minus_T_K_1D, 
                        "pi_sg_plus":pi_sg_plus_T_K_1D}, 
                    index=tu_tk)
            df_b0_c0_pisg_pi0_T_K.append(df_b0_c0_pisg_pi0_T_K_lri)
            
            ## process of df_ben_cst_M_T_K
            BENs_M_T_K_1D = BENs_M_T_K[:,t_periods,:].reshape(-1)
            CSTs_M_T_K_1D = CSTs_M_T_K[:,t_periods,:].reshape(-1)
            df_ben_cst_M_T_K_lri = pd.DataFrame({
                'ben':BENs_M_T_K_1D, 'cst':CSTs_M_T_K_1D}, index=tu_mtk)
            df_ben_cst_M_T_K.append(df_ben_cst_M_T_K_lri)
            ## process of df_B_C_BB_CC_RU_M
            df_B_C_BB_CC_RU_M_lri \
                = pd.DataFrame({
                        "B":B_is_M, "C":C_is_M, 
                        "BB":BB_is_M, "CC":CC_is_M, "RU":RU_is_M}, 
                    index=tu_m)
            df_B_C_BB_CC_RU_M.append(df_B_C_BB_CC_RU_M_lri)
            ## process of 
            ## process of
            
        elif algo in algos_4_no_learning:
            arr_pl_M_T_K_vars_t = arr_pl_M_T_K_vars[:, t_periods, :]
            ## process of arr_pl_M_T_K_vars 
            # turn array from 3D to 4D
            arrs = []
            for k in range(0, k_steps):
                arrs.append(list(arr_pl_M_T_K_vars_t))
            arrs = np.array(arrs, dtype=object)
            arrs = np.transpose(arrs, [1,2,0,3])
            arr_pl_M_T_K_vars_4D = np.zeros((arrs.shape[0],
                                              arrs.shape[1],
                                              arrs.shape[2],
                                              arrs.shape[3]), 
                                            dtype=object)
            
            arr_pl_M_T_K_vars_4D[:,:,:,:] = arrs.copy()
            # turn in 2D
            arr_pl_M_T_K_vars_2D = arr_pl_M_T_K_vars_4D.reshape(
                                        -1, 
                                        arr_pl_M_T_K_vars_4D.shape[3])
            # turn arr_2D to df_{RD}DET 
            # variables[:-3] = ["Si_minus","Si_plus",
            #        "added column so that columns df_lri and df_det are identicals"]
            df_rd_det = pd.DataFrame(data=arr_pl_M_T_K_vars_2D, 
                                     index=tu_mtk, columns=variables)
            
            df_arr_M_T_Ks.append(df_rd_det)
            
            ## process of df_b0_c0_pisg_pi0_T_K
            # turn array from 1D to 2D
            arrs_b0_2D, arrs_c0_2D = [], []
            arrs_pi_0_plus_2D, arrs_pi_0_minus_2D = [], []
            arrs_pi_sg_plus_2D, arrs_pi_sg_minus_2D = [], []
            # print("shape: b0_s_T_K={}, pi_0_minus_T_K={}".format(
            #     b0_s_T_K.shape, pi_0_minus_T_K.shape))
            for k in range(0, k_steps):
                # print("type: b0_s_T_K={}, b0_s_T_K={}; bool={}".format(type(b0_s_T_K), 
                #      b0_s_T_K.shape, b0_s_T_K.shape == ()))
                if b0_s_T_K.shape == ():
                    arrs_b0_2D.append([b0_s_T_K])
                else:
                    arrs_b0_2D.append(list(b0_s_T_K[t_periods]))
                if c0_s_T_K.shape == ():
                    arrs_c0_2D.append([c0_s_T_K])
                else:
                    arrs_c0_2D.append(list(c0_s_T_K[t_periods]))
                if pi_0_plus_T.shape == ():
                    arrs_pi_0_plus_2D.append([pi_0_plus_T])
                else:
                    arrs_pi_0_plus_2D.append(list(pi_0_plus_T[t_periods]))
                if pi_0_minus_T.shape == ():
                    arrs_pi_0_minus_2D.append([pi_0_minus_T])
                else:
                    arrs_pi_0_minus_2D.append(list(pi_0_minus_T[t_periods]))
                if pi_sg_plus_T.shape == ():
                    arrs_pi_sg_plus_2D.append([pi_sg_plus_T])
                else:
                    arrs_pi_sg_plus_2D.append(list(pi_sg_plus_T[t_periods]))
                if pi_sg_minus_T.shape == ():
                    arrs_pi_sg_minus_2D.append([pi_sg_minus_T])
                else:
                    arrs_pi_sg_minus_2D.append(list(pi_sg_minus_T[t_periods]))
                 #arrs_c0_2D.append(list(c0_s_T_K))
                 #arrs_pi_0_plus_2D.append(list(pi_0_plus_T_K))
                 #arrs_pi_0_minus_2D.append(list(pi_0_minus_T_K))
                 #arrs_pi_sg_plus_2D.append(list(pi_sg_plus_T_K))
                 #arrs_pi_sg_minus_2D.append(list(pi_sg_minus_T_K))
            arrs_b0_2D = np.array(arrs_b0_2D, dtype=object)
            arrs_c0_2D = np.array(arrs_c0_2D, dtype=object)
            arrs_pi_0_plus_2D = np.array(arrs_pi_0_plus_2D, dtype=object)
            arrs_pi_0_minus_2D = np.array(arrs_pi_0_minus_2D, dtype=object)
            arrs_pi_sg_plus_2D = np.array(arrs_pi_sg_plus_2D, dtype=object)
            arrs_pi_sg_minus_2D = np.array(arrs_pi_sg_minus_2D, dtype=object)
            arrs_b0_2D = np.transpose(arrs_b0_2D, [1,0])
            arrs_c0_2D = np.transpose(arrs_c0_2D, [1,0])
            arrs_pi_0_plus_2D = np.transpose(arrs_pi_0_plus_2D, [1,0])
            arrs_pi_0_minus_2D = np.transpose(arrs_pi_0_minus_2D, [1,0])
            arrs_pi_sg_plus_2D = np.transpose(arrs_pi_sg_plus_2D, [1,0])
            arrs_pi_sg_minus_2D = np.transpose(arrs_pi_sg_minus_2D, [1,0])
            # turn array from 2D to 1D
            arrs_b0_1D = arrs_b0_2D.reshape(-1)
            arrs_c0_1D = arrs_c0_2D.reshape(-1)
            arrs_pi_0_minus_1D = arrs_pi_0_minus_2D.reshape(-1)
            arrs_pi_0_plus_1D = arrs_pi_0_plus_2D.reshape(-1)
            arrs_pi_sg_minus_1D = arrs_pi_sg_minus_2D.reshape(-1)
            arrs_pi_sg_plus_1D = arrs_pi_sg_plus_2D.reshape(-1)
            # create dataframe
            df_b0_c0_pisg_pi0_T_K_det \
                = pd.DataFrame({
                    "b0":arrs_b0_1D, 
                    "c0":arrs_c0_1D, 
                    "pi_0_minus":arrs_pi_0_minus_1D, 
                    "pi_0_plus":arrs_pi_0_plus_1D, 
                    "pi_sg_minus":arrs_pi_sg_minus_1D, 
                    "pi_sg_plus":arrs_pi_sg_plus_1D}, index=tu_tk)
            df_b0_c0_pisg_pi0_T_K.append(df_b0_c0_pisg_pi0_T_K_det) 

            ## process of df_ben_cst_M_T_K
            # turn array from 2D to 3D
            arrs_ben_3D, arrs_cst_3D = [], []
            for k in range(0, k_steps):
                 arrs_ben_3D.append(list(BENs_M_T_K[:,t_periods]))
                 arrs_cst_3D.append(list(CSTs_M_T_K[:,t_periods]))
            arrs_ben_3D = np.array(arrs_ben_3D, dtype=object)
            arrs_cst_3D = np.array(arrs_cst_3D, dtype=object)
            arrs_ben_3D = np.transpose(arrs_ben_3D, [1,2,0])
            arrs_cst_3D = np.transpose(arrs_cst_3D, [1,2,0])
    
            # turn array from 3D to 1D
            BENs_M_T_K_1D = arrs_ben_3D.reshape(-1)
            CSTs_M_T_K_1D = arrs_cst_3D.reshape(-1)
            #create dataframe
            df_ben = pd.DataFrame(data=BENs_M_T_K_1D, 
                              index=tu_mtk, columns=['ben'])
            df_cst = pd.DataFrame(data=CSTs_M_T_K_1D, 
                              index=tu_mtk, columns=['cst'])
            df_ben_cst_M_T_K_det = pd.concat([df_ben, df_cst], axis=1)

            df_ben_cst_M_T_K.append(df_ben_cst_M_T_K_det)
            
            
            ## process of df_B_C_BB_CC_RU_M
            df_B_C_BB_CC_RU_M_det = pd.DataFrame({
                "B":B_is_M, "C":C_is_M, 
                "BB":BB_is_M,"CC":CC_is_M,"RU":RU_is_M,}, index=tu_m)
            df_B_C_BB_CC_RU_M.append(df_B_C_BB_CC_RU_M_det)
            ## process of 
            ## process of 
        
    df_arr_M_T_Ks = pd.concat(df_arr_M_T_Ks, axis=0)
    df_ben_cst_M_T_K = pd.concat(df_ben_cst_M_T_K, axis=0)
    df_b0_c0_pisg_pi0_T_K = pd.concat(df_b0_c0_pisg_pi0_T_K, axis=0)
    df_B_C_BB_CC_RU_M = pd.concat(df_B_C_BB_CC_RU_M, axis=0)
    
    # insert index as columns of dataframes
    ###  df_arr_M_T_Ks
    columns_df = df_arr_M_T_Ks.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i","t","k"]
    indices = list(df_arr_M_T_Ks.index)
    df_ind = pd.DataFrame(indices,columns=columns_ind)
    df_arr_M_T_Ks = pd.concat([df_ind.reset_index(), 
                                df_arr_M_T_Ks.reset_index()],
                              axis=1, ignore_index=True)
    df_arr_M_T_Ks.drop(df_arr_M_T_Ks.columns[[0]], axis=1, inplace=True)
    df_arr_M_T_Ks.columns = columns_ind+["old_index"]+columns_df
    df_arr_M_T_Ks.pop("old_index")
    ###  df_ben_cst_M_T_K
    columns_df = df_ben_cst_M_T_K.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i","t","k"]
    indices = list(df_ben_cst_M_T_K.index)
    df_ind = pd.DataFrame(indices,columns=columns_ind)
    df_ben_cst_M_T_K = pd.concat([df_ind.reset_index(), 
                                df_ben_cst_M_T_K.reset_index()],
                              axis=1, ignore_index=True)
    df_ben_cst_M_T_K.drop(df_ben_cst_M_T_K.columns[[0]], axis=1, inplace=True)
    df_ben_cst_M_T_K.columns = columns_ind+["old_index"]+columns_df
    df_ben_cst_M_T_K.pop("old_index")
    df_ben_cst_M_T_K["state_i"] = df_arr_M_T_Ks["state_i"]
    ###  df_b0_c0_pisg_pi0_T_K
    columns_df = df_b0_c0_pisg_pi0_T_K.columns.to_list()
    columns_ind = ["algo","rate","prices","t","k"]
    indices = list(df_b0_c0_pisg_pi0_T_K.index)
    df_ind = pd.DataFrame(indices, columns=columns_ind)
    df_b0_c0_pisg_pi0_T_K = pd.concat([df_ind.reset_index(), 
                                        df_b0_c0_pisg_pi0_T_K.reset_index()],
                                        axis=1, ignore_index=True)
    df_b0_c0_pisg_pi0_T_K.drop(df_b0_c0_pisg_pi0_T_K.columns[[0]], 
                               axis=1, inplace=True)
    df_b0_c0_pisg_pi0_T_K.columns = columns_ind+["old_index"]+columns_df
    df_b0_c0_pisg_pi0_T_K.pop("old_index")
    ###  df_B_C_BB_CC_RU_M
    columns_df = df_B_C_BB_CC_RU_M.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i"]
    indices = list(df_B_C_BB_CC_RU_M.index)
    df_ind = pd.DataFrame(indices, columns=columns_ind)
    df_B_C_BB_CC_RU_M = pd.concat([df_ind.reset_index(), 
                                        df_B_C_BB_CC_RU_M.reset_index()],
                                        axis=1, ignore_index=True)
    df_B_C_BB_CC_RU_M.drop(df_B_C_BB_CC_RU_M.columns[[0]], 
                               axis=1, inplace=True)
    df_B_C_BB_CC_RU_M.columns = columns_ind+["old_index"]+columns_df
    df_B_C_BB_CC_RU_M.pop("old_index")

    return df_arr_M_T_Ks, df_ben_cst_M_T_K, \
            df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M
            
def get_array_turn_df_for_t(tuple_paths, t=1, k_steps_args=250, 
                            algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                                 "BEST-BRUTE-FORCE",
                                                 "BAD-BRUTE-FORCE", 
                                                 "MIDDLE-BRUTE-FORCE"], 
                            algos_4_learning=["LRI1", "LRI2"]):
    df_arr_M_T_Ks = []
    df_b0_c0_pisg_pi0_T_K = []
    df_B_C_BB_CC_RU_M = []
    df_ben_cst_M_T_K = []
    for tuple_path in tuple_paths:
        path_to_variable = os.path.join(*tuple_path)
        
        # arr_pl_M_T_K_vars, \
        # b0_s_T_K, c0_s_T_K, \
        # B_is_M, C_is_M, \
        # BENs_M_T_K, CSTs_M_T_K, \
        # BB_is_M, CC_is_M, RU_is_M, \
        # pi_sg_plus_T, pi_sg_minus_T, \
        # pi_0_plus_T, pi_0_minus_T, \
        # pi_hp_plus_s, pi_hp_minus_s \
        #     = get_local_storage_variables(path_to_variable)
            
        arr_pl_M_T_K_vars, \
        b0_s_T_K, c0_s_T_K, \
        B_is_M, C_is_M, B_is_M_T, C_is_M_T,\
        BENs_M_T_K, CSTs_M_T_K, \
        BB_is_M, CC_is_M, RU_is_M, BB_is_M_T, CC_is_M_T, RU_is_M_T,\
        pi_sg_plus_T, pi_sg_minus_T, \
        pi_0_plus_T, pi_0_minus_T, \
        pi_hp_plus_T, pi_hp_minus_T \
            = fct_aux.get_local_storage_variables(
                path_to_variable=path_to_variable)
        
        price = tuple_path[2].split("_")[3]+"_"+tuple_path[2].split("_")[-1]
        algo = tuple_path[3];
        rate = tuple_path[4] if algo in algos_4_learning else 0
        
        m_players = arr_pl_M_T_K_vars.shape[0]
        k_steps = arr_pl_M_T_K_vars.shape[2] if arr_pl_M_T_K_vars.shape == 4 \
                                             else k_steps_args                                    
        #for t in range(0, t_periods):                                     
        t_periods = None; tu_mtk = None; tu_tk = None; tu_m = None
        if t is None:
            t_periods = arr_pl_M_T_K_vars.shape[1]
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     range(0, t_periods), 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    range(0, t_periods), 
                                    range(0, k_steps)))
            t_periods = list(range(0, t_periods))
        elif type(t) is list:
            t_periods = t
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    t_periods, 
                                    range(0, k_steps)))
        elif type(t) is int:
            t_periods = [t]
            tu_mtk = list(it.product([algo], [rate], [price],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price],
                                    t_periods, 
                                    range(0, k_steps)))
                      
        print('t_periods = {}'.format(t_periods))
        tu_m = list(it.product([algo], [rate], [price], range(0, m_players)))
                    
        variables = list(fct_aux.AUTOMATE_INDEX_ATTRS.keys())
        
        if algo in algos_4_learning:
            arr_pl_M_T_K_vars_t = arr_pl_M_T_K_vars[:, t_periods, :, :]
            ## process of arr_pl_M_T_K_vars 
            arr_pl_M_T_K_vars_2D = arr_pl_M_T_K_vars_t.reshape(
                                        -1, 
                                        arr_pl_M_T_K_vars.shape[3])
            df_lri_x = pd.DataFrame(data=arr_pl_M_T_K_vars_2D, 
                                    index=tu_mtk, 
                                    columns=variables)
            
            df_arr_M_T_Ks.append(df_lri_x)
            
            ## process of df_b0_c0_pisg_pi0_T_K
            b0_s_T_K_2D = []
            c0_s_T_K_2D = []
            pi_0_minus_T_K_2D = []
            pi_0_plus_T_K_2D = []
            pi_sg_minus_T_K_2D = []
            pi_sg_plus_T_K_2D = []
            for tx in t_periods:
                b0_s_T_K_2D.append(list( b0_s_T_K[tx,:].reshape(-1)))
                c0_s_T_K_2D.append(list( c0_s_T_K[tx,:].reshape(-1)))
                pi_0_minus_T_K_2D.append([ pi_0_minus_T[tx] ]*k_steps_args)
                pi_0_plus_T_K_2D.append([ pi_0_plus_T[tx] ]*k_steps_args)
                pi_sg_minus_T_K_2D.append([ pi_sg_minus_T[tx] ]*k_steps_args)
                pi_sg_plus_T_K_2D.append([ pi_sg_plus_T[tx] ]*k_steps_args)
            b0_s_T_K_2D = np.array(b0_s_T_K_2D, dtype=object)
            c0_s_T_K_2D = np.array(c0_s_T_K_2D, dtype=object)
            pi_0_minus_T_K_2D = np.array(pi_0_minus_T_K_2D, dtype=object)
            pi_0_plus_T_K_2D = np.array(pi_0_plus_T_K_2D, dtype=object)
            pi_sg_minus_T_K_2D = np.array(pi_sg_minus_T_K_2D, dtype=object)
            pi_sg_plus_T_K_2D = np.array(pi_sg_plus_T_K_2D, dtype=object)
            
            b0_s_T_K_1D = b0_s_T_K_2D.reshape(-1)
            c0_s_T_K_1D = c0_s_T_K_2D.reshape(-1)
            pi_0_minus_T_K_1D = pi_0_minus_T_K_2D.reshape(-1)
            pi_0_plus_T_K_1D = pi_0_plus_T_K_2D.reshape(-1)
            pi_sg_minus_T_K_1D = pi_sg_minus_T_K_2D.reshape(-1)
            pi_sg_plus_T_K_1D = pi_sg_plus_T_K_2D.reshape(-1)
            
            df_b0_c0_pisg_pi0_T_K_lri \
                = pd.DataFrame({
                        "b0":b0_s_T_K_1D, "c0":c0_s_T_K_1D, 
                        "pi_0_minus":pi_0_minus_T_K_1D, 
                        "pi_0_plus":pi_0_plus_T_K_1D, 
                        "pi_sg_minus":pi_sg_minus_T_K_1D, 
                        "pi_sg_plus":pi_sg_plus_T_K_1D}, 
                    index=tu_tk)
            df_b0_c0_pisg_pi0_T_K.append(df_b0_c0_pisg_pi0_T_K_lri)
            
            ## process of df_ben_cst_M_T_K
            BENs_M_T_K_1D = BENs_M_T_K[:,t_periods,:].reshape(-1)
            CSTs_M_T_K_1D = CSTs_M_T_K[:,t_periods,:].reshape(-1)
            df_ben_cst_M_T_K_lri = pd.DataFrame({
                'ben':BENs_M_T_K_1D, 'cst':CSTs_M_T_K_1D}, index=tu_mtk)
            df_ben_cst_M_T_K.append(df_ben_cst_M_T_K_lri)
            ## process of df_B_C_BB_CC_RU_M
            df_B_C_BB_CC_RU_M_lri \
                = pd.DataFrame({
                        "B":B_is_M, "C":C_is_M, 
                        "BB":BB_is_M, "CC":CC_is_M, "RU":RU_is_M}, 
                    index=tu_m)
            df_B_C_BB_CC_RU_M.append(df_B_C_BB_CC_RU_M_lri)
            ## process of 
            ## process of
            
        elif algo in algos_4_no_learning:
            arr_pl_M_T_K_vars_t = arr_pl_M_T_K_vars[:, t_periods, :]
            ## process of arr_pl_M_T_K_vars 
            # turn array from 3D to 4D
            arrs = []
            for k in range(0, k_steps):
                arrs.append(list(arr_pl_M_T_K_vars_t))
            arrs = np.array(arrs, dtype=object)
            arrs = np.transpose(arrs, [1,2,0,3])
            arr_pl_M_T_K_vars_4D = np.zeros((arrs.shape[0],
                                              arrs.shape[1],
                                              arrs.shape[2],
                                              arrs.shape[3]), 
                                            dtype=object)
            
            arr_pl_M_T_K_vars_4D[:,:,:,:] = arrs.copy()
            # turn in 2D
            arr_pl_M_T_K_vars_2D = arr_pl_M_T_K_vars_4D.reshape(
                                        -1, 
                                        arr_pl_M_T_K_vars_4D.shape[3])
            # turn arr_2D to df_{RD}DET 
            # variables[:-3] = ["Si_minus","Si_plus",
            #        "added column so that columns df_lri and df_det are identicals"]
            df_rd_det = pd.DataFrame(data=arr_pl_M_T_K_vars_2D, 
                                     index=tu_mtk, columns=variables)
            
            df_arr_M_T_Ks.append(df_rd_det)
            
            ## process of df_b0_c0_pisg_pi0_T_K
            # turn array from 1D to 2D
            arrs_b0_2D, arrs_c0_2D = [], []
            arrs_pi_0_plus_2D, arrs_pi_0_minus_2D = [], []
            arrs_pi_sg_plus_2D, arrs_pi_sg_minus_2D = [], []
            # print("shape: b0_s_T_K={}, pi_0_minus_T_K={}".format(
            #     b0_s_T_K.shape, pi_0_minus_T_K.shape))
            for k in range(0, k_steps):
                # print("type: b0_s_T_K={}, b0_s_T_K={}; bool={}".format(type(b0_s_T_K), 
                #      b0_s_T_K.shape, b0_s_T_K.shape == ()))
                if b0_s_T_K.shape == ():
                    arrs_b0_2D.append([b0_s_T_K])
                else:
                    arrs_b0_2D.append(list(b0_s_T_K[t_periods]))
                if c0_s_T_K.shape == ():
                    arrs_c0_2D.append([c0_s_T_K])
                else:
                    arrs_c0_2D.append(list(c0_s_T_K[t_periods]))
                if pi_0_plus_T.shape == ():
                    arrs_pi_0_plus_2D.append([pi_0_plus_T])
                else:
                    arrs_pi_0_plus_2D.append(list(pi_0_plus_T[t_periods]))
                if pi_0_minus_T.shape == ():
                    arrs_pi_0_minus_2D.append([pi_0_minus_T])
                else:
                    arrs_pi_0_minus_2D.append(list(pi_0_minus_T[t_periods]))
                if pi_sg_plus_T.shape == ():
                    arrs_pi_sg_plus_2D.append([pi_sg_plus_T])
                else:
                    arrs_pi_sg_plus_2D.append(list(pi_sg_plus_T[t_periods]))
                if pi_sg_minus_T.shape == ():
                    arrs_pi_sg_minus_2D.append([pi_sg_minus_T])
                else:
                    arrs_pi_sg_minus_2D.append(list(pi_sg_minus_T[t_periods]))
                 #arrs_c0_2D.append(list(c0_s_T_K))
                 #arrs_pi_0_plus_2D.append(list(pi_0_plus_T_K))
                 #arrs_pi_0_minus_2D.append(list(pi_0_minus_T_K))
                 #arrs_pi_sg_plus_2D.append(list(pi_sg_plus_T_K))
                 #arrs_pi_sg_minus_2D.append(list(pi_sg_minus_T_K))
            arrs_b0_2D = np.array(arrs_b0_2D, dtype=object)
            arrs_c0_2D = np.array(arrs_c0_2D, dtype=object)
            arrs_pi_0_plus_2D = np.array(arrs_pi_0_plus_2D, dtype=object)
            arrs_pi_0_minus_2D = np.array(arrs_pi_0_minus_2D, dtype=object)
            arrs_pi_sg_plus_2D = np.array(arrs_pi_sg_plus_2D, dtype=object)
            arrs_pi_sg_minus_2D = np.array(arrs_pi_sg_minus_2D, dtype=object)
            arrs_b0_2D = np.transpose(arrs_b0_2D, [1,0])
            arrs_c0_2D = np.transpose(arrs_c0_2D, [1,0])
            arrs_pi_0_plus_2D = np.transpose(arrs_pi_0_plus_2D, [1,0])
            arrs_pi_0_minus_2D = np.transpose(arrs_pi_0_minus_2D, [1,0])
            arrs_pi_sg_plus_2D = np.transpose(arrs_pi_sg_plus_2D, [1,0])
            arrs_pi_sg_minus_2D = np.transpose(arrs_pi_sg_minus_2D, [1,0])
            # turn array from 2D to 1D
            arrs_b0_1D = arrs_b0_2D.reshape(-1)
            arrs_c0_1D = arrs_c0_2D.reshape(-1)
            arrs_pi_0_minus_1D = arrs_pi_0_minus_2D.reshape(-1)
            arrs_pi_0_plus_1D = arrs_pi_0_plus_2D.reshape(-1)
            arrs_pi_sg_minus_1D = arrs_pi_sg_minus_2D.reshape(-1)
            arrs_pi_sg_plus_1D = arrs_pi_sg_plus_2D.reshape(-1)
            # create dataframe
            df_b0_c0_pisg_pi0_T_K_det \
                = pd.DataFrame({
                    "b0":arrs_b0_1D, 
                    "c0":arrs_c0_1D, 
                    "pi_0_minus":arrs_pi_0_minus_1D, 
                    "pi_0_plus":arrs_pi_0_plus_1D, 
                    "pi_sg_minus":arrs_pi_sg_minus_1D, 
                    "pi_sg_plus":arrs_pi_sg_plus_1D}, index=tu_tk)
            df_b0_c0_pisg_pi0_T_K.append(df_b0_c0_pisg_pi0_T_K_det) 

            ## process of df_ben_cst_M_T_K
            # turn array from 2D to 3D
            arrs_ben_3D, arrs_cst_3D = [], []
            for k in range(0, k_steps):
                 arrs_ben_3D.append(list(BENs_M_T_K[:,t_periods]))
                 arrs_cst_3D.append(list(CSTs_M_T_K[:,t_periods]))
            arrs_ben_3D = np.array(arrs_ben_3D, dtype=object)
            arrs_cst_3D = np.array(arrs_cst_3D, dtype=object)
            arrs_ben_3D = np.transpose(arrs_ben_3D, [1,2,0])
            arrs_cst_3D = np.transpose(arrs_cst_3D, [1,2,0])
    
            # turn array from 3D to 1D
            BENs_M_T_K_1D = arrs_ben_3D.reshape(-1)
            CSTs_M_T_K_1D = arrs_cst_3D.reshape(-1)
            #create dataframe
            df_ben = pd.DataFrame(data=BENs_M_T_K_1D, 
                              index=tu_mtk, columns=['ben'])
            df_cst = pd.DataFrame(data=CSTs_M_T_K_1D, 
                              index=tu_mtk, columns=['cst'])
            df_ben_cst_M_T_K_det = pd.concat([df_ben, df_cst], axis=1)

            df_ben_cst_M_T_K.append(df_ben_cst_M_T_K_det)
            
            
            ## process of df_B_C_BB_CC_RU_M
            df_B_C_BB_CC_RU_M_det = pd.DataFrame({
                "B":B_is_M, "C":C_is_M, 
                "BB":BB_is_M,"CC":CC_is_M,"RU":RU_is_M,}, index=tu_m)
            df_B_C_BB_CC_RU_M.append(df_B_C_BB_CC_RU_M_det)
            ## process of 
            ## process of 
        
    df_arr_M_T_Ks = pd.concat(df_arr_M_T_Ks, axis=0)
    df_ben_cst_M_T_K = pd.concat(df_ben_cst_M_T_K, axis=0)
    df_b0_c0_pisg_pi0_T_K = pd.concat(df_b0_c0_pisg_pi0_T_K, axis=0)
    df_B_C_BB_CC_RU_M = pd.concat(df_B_C_BB_CC_RU_M, axis=0)
    
    # insert index as columns of dataframes
    ###  df_arr_M_T_Ks
    columns_df = df_arr_M_T_Ks.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i","t","k"]
    indices = list(df_arr_M_T_Ks.index)
    df_ind = pd.DataFrame(indices,columns=columns_ind)
    df_arr_M_T_Ks = pd.concat([df_ind.reset_index(), 
                                df_arr_M_T_Ks.reset_index()],
                              axis=1, ignore_index=True)
    df_arr_M_T_Ks.drop(df_arr_M_T_Ks.columns[[0]], axis=1, inplace=True)
    df_arr_M_T_Ks.columns = columns_ind+["old_index"]+columns_df
    df_arr_M_T_Ks.pop("old_index")
    ###  df_ben_cst_M_T_K
    columns_df = df_ben_cst_M_T_K.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i","t","k"]
    indices = list(df_ben_cst_M_T_K.index)
    df_ind = pd.DataFrame(indices,columns=columns_ind)
    df_ben_cst_M_T_K = pd.concat([df_ind.reset_index(), 
                                df_ben_cst_M_T_K.reset_index()],
                              axis=1, ignore_index=True)
    df_ben_cst_M_T_K.drop(df_ben_cst_M_T_K.columns[[0]], axis=1, inplace=True)
    df_ben_cst_M_T_K.columns = columns_ind+["old_index"]+columns_df
    df_ben_cst_M_T_K.pop("old_index")
    df_ben_cst_M_T_K["state_i"] = df_arr_M_T_Ks["state_i"]
    ###  df_b0_c0_pisg_pi0_T_K
    columns_df = df_b0_c0_pisg_pi0_T_K.columns.to_list()
    columns_ind = ["algo","rate","prices","t","k"]
    indices = list(df_b0_c0_pisg_pi0_T_K.index)
    df_ind = pd.DataFrame(indices, columns=columns_ind)
    df_b0_c0_pisg_pi0_T_K = pd.concat([df_ind.reset_index(), 
                                        df_b0_c0_pisg_pi0_T_K.reset_index()],
                                        axis=1, ignore_index=True)
    df_b0_c0_pisg_pi0_T_K.drop(df_b0_c0_pisg_pi0_T_K.columns[[0]], 
                               axis=1, inplace=True)
    df_b0_c0_pisg_pi0_T_K.columns = columns_ind+["old_index"]+columns_df
    df_b0_c0_pisg_pi0_T_K.pop("old_index")
    ###  df_B_C_BB_CC_RU_M
    columns_df = df_B_C_BB_CC_RU_M.columns.to_list()
    columns_ind = ["algo","rate","prices","pl_i"]
    indices = list(df_B_C_BB_CC_RU_M.index)
    df_ind = pd.DataFrame(indices, columns=columns_ind)
    df_B_C_BB_CC_RU_M = pd.concat([df_ind.reset_index(), 
                                        df_B_C_BB_CC_RU_M.reset_index()],
                                        axis=1, ignore_index=True)
    df_B_C_BB_CC_RU_M.drop(df_B_C_BB_CC_RU_M.columns[[0]], 
                               axis=1, inplace=True)
    df_B_C_BB_CC_RU_M.columns = columns_ind+["old_index"]+columns_df
    df_B_C_BB_CC_RU_M.pop("old_index")

    return df_arr_M_T_Ks, df_ben_cst_M_T_K, \
            df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M
# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> fin
# _____________________________________________________________________________ 


# _____________________________________________________________________________
#
#                   distribution by states for periods ---> debut
# _____________________________________________________________________________
def plot_distribution(df_al_pr_ra, algo, rate, price,
                      path_2_best_learning_steps):
    """
    plot the bar plot with key is (t, stateX) (X={1,2,3})
    """
    cols = ["t", "state_i"]
    df_state = df_al_pr_ra.groupby(cols)[["state_i"]].count()
    df_state.rename(columns={"state_i":"nb_players"}, inplace=True)
    df_state = df_state.reset_index()
    df_state["t"] = df_state["t"].astype(str)
    
    x = list(map(tuple,list(df_state[cols].values)))
    nb_players = list(df_state["nb_players"])
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("nb_players", "@nb_players")
                            ]
                        )
    px= figure(x_range=FactorRange(*x), 
               plot_height=350, plot_width = int(WIDTH*MULT_WIDTH),
               title="number of players, ({}, rate={}, price={})".format(
                  algo, rate, price),
                toolbar_location=None, tools=TOOLS)

    data = dict(x=x, nb_players=nb_players)
    
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='nb_players', width=0.9, source=source, 
            fill_color=factor_cmap('x', 
                                   palette=Category20[20], 
                                   factors=list(df_state["t"].unique()), 
                                   start=0, end=1))
    
    px.y_range.start = 0
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    
    return px
    
def plot_distribution_by_states_4_periods(df_arr_M_T_Ks, k_steps_args,
                                          path_2_best_learning_steps):
    """
    plot the distribution by state for each period
    plot is the bar plot with key is (t, stateX) (X={1,2,3})
    
    """
    
    rates = df_arr_M_T_Ks["rate"].unique().tolist(); rate = rates[rates!=0]
    prices = df_arr_M_T_Ks["prices"].unique().tolist()
    algos = df_arr_M_T_Ks["algo"].unique().tolist()
    
    dico_pxs = dict()
    for algo, price in it.product(algos, prices):
        mask_al_pr_ra = ((df_arr_M_T_Ks.rate == str(rate)) 
                                 | (df_arr_M_T_Ks.rate == 0)) \
                            & (df_arr_M_T_Ks.prices == price) \
                            & (df_arr_M_T_Ks.algo == algo) \
                            & (df_arr_M_T_Ks.k == k_steps_args-1)    
        df_al_pr_ra = df_arr_M_T_Ks[mask_al_pr_ra].copy()
        
        pxs_al_pr_ra = plot_distribution(df_al_pr_ra, algo, rate, price,
                                         path_2_best_learning_steps)
        
        if (algo, price, rate) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate)] \
                = [pxs_al_pr_ra]
        else:
            dico_pxs[(algo, price, rate)].append(pxs_al_pr_ra)
        
    rows_dists_ts = list()
    for key, pxs_al_pr_ra in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra)
        rows_dists_ts.append(col_px_sts)
    rows_dists_ts=column(children=rows_dists_ts, 
                            sizing_mode='stretch_both')
    return rows_dists_ts

# _____________________________________________________________________________
#
#                   distribution by states for periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#                   utility of players for periods ---> debut
# _____________________________________________________________________________
def compute_CONS_PROD(df_prod_cons, algo, path_2_best_learning_steps):
    """
    compute CONS, PROD 
    """
    k_stop = None;
    path_2_best_learning = None
    if algo in ["LRI1", "LRI2"]:
        for path_2_best in path_2_best_learning_steps:
            if algo == path_2_best[3]:
                path_2_best_learning = os.path.join(*path_2_best)        
        df_tmp = pd.read_csv(os.path.join(path_2_best_learning,
                                           "best_learning_steps.csv"), 
                             sep=',', index_col=0)
    else:
        k_step_max=df_prod_cons.k.unique().max()
        t_periods = df_prod_cons.t.unique().tolist()
        dico = dict()
        for t in t_periods:
            dico[str(t)] = {"k_stop":k_step_max}
        df_tmp = pd.DataFrame.from_dict(dico)
        #k_stop = df_prod_cons.k.unique().max()
    
    # print("df_tmp: index={}, cols={}".format(df_tmp.index, df_tmp.columns))
    
    list_of_players = df_prod_cons.pl_i.unique().tolist()
    cols = ['pl_i', 'PROD_i', 'CONS_i', 'set']
    df_PROD_CONS = pd.DataFrame(columns=cols, 
                                index=list_of_players)
    for num_pl_i in list_of_players:
        sum_prod_pl_i = 0; sum_cons_pl_i = 0; setX = None
        for t in df_prod_cons.t.unique().tolist():
            k_stop = df_tmp.loc["k_stop",str(t)]
            mask_pli_kstop = (df_prod_cons.t == t) \
                             & (df_prod_cons.k == k_stop) \
                             & (df_prod_cons.pl_i == num_pl_i)
            # print("df_prod_cons[mask_pli_kstop]={}".format(df_prod_cons[mask_pli_kstop].shape))
            # df_prod_cons_mask = df_prod_cons[mask_pli_kstop]
            # print("pl_{}, prod_i={}, value={}".format(num_pl_i, 
            #       df_prod_cons_mask["prod_i"], df_prod_cons_mask["prod_i"].values[0] ))
            # sum_prod_pl_i += df_prod_cons[mask_pli_kstop].loc[num_pl_i,"prod_i"]
            # sum_cons_pl_i += df_prod_cons[mask_pli_kstop].loc[num_pl_i,"cons_i"]
            sum_prod_pl_i += df_prod_cons[mask_pli_kstop]["prod_i"].values[0]
            sum_cons_pl_i += df_prod_cons[mask_pli_kstop]["cons_i"].values[0]
            setX = df_prod_cons[mask_pli_kstop]["set"].values[0]
        # print('pl_{}, sum_prod_pl_i={}, sum_cons_pl_i={}'.format(
        #         num_pl_i, sum_prod_pl_i, sum_cons_pl_i))
        df_PROD_CONS.loc[num_pl_i, "PROD_i"] = sum_prod_pl_i
        df_PROD_CONS.loc[num_pl_i, "CONS_i"] = sum_cons_pl_i
        df_PROD_CONS.loc[num_pl_i, "pl_i"] = num_pl_i
        df_PROD_CONS.loc[num_pl_i, "set"] = setX
        
    return df_PROD_CONS

def plot_CONS_PROD(df_prod_cons, algo, rate, price,
                   path_2_best_learning_steps):
    """
    plot CONS, PROD for each player
    """
    k_stop = None;
    path_2_best_learning = None
    if algo in ["LRI1", "LRI2"]:
        for path_2_best in path_2_best_learning_steps:
            if algo == path_2_best[3]:
                path_2_best_learning = os.path.join(*path_2_best)
    else:
        k_stop = df_prod_cons.k.unique().max()
            
    print("path_2_best_learning={}".format(path_2_best_learning))
    df_tmp = pd.read_csv(os.path.join(path_2_best_learning,
                                       "best_learning_steps.csv"), 
                         sep=',', index_col=0)
    
    list_of_players = df_prod_cons.pl_i.unique().tolist()
    cols = ['pl_i', 'PROD_i', 'CONS_i']
    df_PROD_CONS = pd.DataFrame(columns=cols, 
                                index=list_of_players)
    for num_pl_i in list_of_players:
        sum_prod_pl_i = 0; sum_cons_pl_i = 0
        for t in df_prod_cons.t.unique().tolist():
            k_stop = df_tmp.loc["k_stop",str(t)]
            mask_pli_kstop = (df_prod_cons.t == t) \
                             & (df_prod_cons.k == k_stop) \
                             & (df_prod_cons.pl_i == num_pl_i)
            sum_prod_pl_i += df_prod_cons[mask_pli_kstop]["prod_i"]
            sum_cons_pl_i += df_prod_cons[mask_pli_kstop]["cons_i"]
        df_PROD_CONS.loc[num_pl_i, "PROD_i"] = sum_prod_pl_i
        df_PROD_CONS.loc[num_pl_i, "CONS_i"] = sum_cons_pl_i
        df_PROD_CONS.loc[num_pl_i, "pl_i"] = num_pl_i
        
    # plot
    df_PROD_CONS["pl_i"] = df_PROD_CONS["pl_i"].astype(str)
    idx = df_PROD_CONS["pl_i"].unique().tolist()
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("pl_i", "@pl_i"),
                            ("PROD_i", "@PROD_i"),
                            ("CONS_i", "@CONS_i"),
                            ]
                        )
    
    px = figure(x_range=idx, 
                y_range=(0, df_PROD_CONS[cols[1:]].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "{}: PROD/CONS of players (rate:{}, price={})"\
                .format(algo, rate, price)
    px.title.text = title
           
    source = ColumnDataSource(data = df_PROD_CONS)
    
    width= 0.2 #0.5
    px.vbar(x=dodge('pl_i', -0.3, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#c9d9d3", legend_label=cols[1])
    px.vbar(x=dodge('pl_i', -0.3+width, range=px.x_range), top=cols[2], 
                    width=width, source=source,
                    color="#718dbf", legend_label=cols[2])
    
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "players"
    px.yaxis.axis_label = "values"
    
    return px, df_PROD_CONS
            
def plot_utility_OLD(df_al_pr_ra, algo, rate, price,
                 path_2_best_learning_steps):
    """
    plot the bar plot of each player relying on the real utility RU
    """
    
    df_al_pr_ra["pl_i"] = df_al_pr_ra["pl_i"].astype(str)
    idx = df_al_pr_ra["pl_i"].unique().tolist()
    cols = ['pl_i', 'BB', 'CC', 'RU']
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("RU", "@RU"),
                            ("BB", "@BB"),
                            ("CC", "@CC"),
                            ]
                        )
    
    px = figure(x_range=idx, 
                y_range=(0, df_al_pr_ra[cols[1:]].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "{}: utility of players (rate:{}, price={})".format(algo, rate, price)
    px.title.text = title
           
    source = ColumnDataSource(data = df_al_pr_ra)
    
    width= 0.2 #0.5
    px.vbar(x=dodge('pl_i', -0.3, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#c9d9d3", legend_label=cols[1])
    px.vbar(x=dodge('pl_i', -0.3+width, range=px.x_range), top=cols[2], 
                    width=width, source=source,
                    color="#718dbf", legend_label=cols[2])
    px.vbar(x=dodge('pl_i', -0.3+2*width, range=px.x_range), top=cols[3], 
                   width=width, source=source,
                   color="#e84d60", legend_label=cols[3])
    
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "players"
    px.yaxis.axis_label = "values"
    
    return px
    
def plot_players_utility(df_res, algo, rate, price,
                 path_2_best_learning_steps):
    """
    plot the bar plot of each player relying on the real utility RU
    """
    
    df_res["pl_i"] = df_res["pl_i"].astype(str)
    idx = df_res["pl_i"].unique().tolist()
    cols = ['pl_i', 'BB', 'CC', 'RU', "CONS_i", "PROD_i"]
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("pl_i", "@pl_i"),
                            ("RU", "@RU"),
                            ("BB", "@BB"),
                            ("CC", "@CC"),
                            ("CONS_i", "@CONS_i"),
                            ("PROD_i", "@PROD_i"),
                            ("set", "@set"),
                            ]
                        )
    
    px = figure(x_range=idx, 
                y_range=(0, df_res[cols[1:]].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "{}: utility of players (rate:{}, price={})".format(algo, rate, price)
    px.title.text = title
           
    source = ColumnDataSource(data = df_res)
    
    width= 0.2 #0.5
    px.vbar(x=dodge('pl_i', -0.3, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#2E8B57", legend_label=cols[1])
    px.vbar(x=dodge('pl_i', -0.3+width, range=px.x_range), top=cols[2], 
                    width=width, source=source,
                    color="#718dbf", legend_label=cols[2])
    px.vbar(x=dodge('pl_i', -0.3+2*width, range=px.x_range), top=cols[3], 
                   width=width, source=source,
                   color="#e84d60", legend_label=cols[3])
    px.vbar(x=dodge('pl_i', -0.3+3*width, range=px.x_range), top=cols[4], 
                   width=width, source=source,
                   color="#ddb7b1", legend_label=cols[4])
    px.vbar(x=dodge('pl_i', -0.3+4*width, range=px.x_range), top=cols[5], 
                   width=width, source=source,
                   color="#FFD700", legend_label=cols[5])
    
    px.y_range.start = df_res.RU.min() - 1
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "players"
    px.yaxis.axis_label = "values"
    
    return px
    
def plot_utilities_by_player_4_periods(df_arr_M_T_Ks,
                                       df_B_C_BB_CC_RU_M, 
                                       path_2_best_learning_steps):
    """
    plot the utility RU, CONS and PROD of players.
    for each algorithm, plot the utility of each player 
    """
    rates = df_arr_M_T_Ks.rate.unique().tolist(); rate = rates[rates!=0]
    prices = df_arr_M_T_Ks.prices.unique().tolist()
    algos = df_B_C_BB_CC_RU_M.algo.unique().tolist()
    
    dico_pxs = dict()
    for algo, price in it.product(algos, prices):
        # CONS_is, PROD_is = compute_CONS_PROD(df_arr_M_T_Ks, 
        #                                      algo, 
        #                                      path_2_best_learning_steps)
        # mask_al_pr_ra = ((df_B_C_BB_CC_RU_M.rate == str(rate)) 
        #                          | (df_B_C_BB_CC_RU_M.rate == 0)) \
        #                     & (df_B_C_BB_CC_RU_M.prices == price) \
        #                     & (df_B_C_BB_CC_RU_M.algo == algo)     
        
        # mask_al_pr_ra_prod_cons = ((df_arr_M_T_Ks.rate == str(rate)) 
        #                            | (df_arr_M_T_Ks.rate == 0)) \
        #                             & (df_arr_M_T_Ks.prices == price) \
        #                             & (df_arr_M_T_Ks.algo == algo)  
        
        # df_al_pr_ra = df_B_C_BB_CC_RU_M[mask_al_pr_ra].copy()
        # pxs_al_pr_ra = plot_utility(
        #                     df_al_pr_ra, algo, rate, price,
        #                     path_2_best_learning_steps)
        
        # df_prod_cons = df_arr_M_T_Ks[mask_al_pr_ra_prod_cons].copy()
        # df_PROD_CONS = compute_CONS_PROD(df_prod_cons, algo, 
        #                                  path_2_best_learning_steps)
        
        # pxs_prod_cons, df_PROD_CONS = plot_CONS_PROD(
        #                                 df_prod_cons, algo, rate, price,
        #                                 path_2_best_learning_steps)
        
        ######################################################################
        print("ALGO={}".format(algo))
        mask_al_pr_ra = ((df_B_C_BB_CC_RU_M.rate == str(rate)) 
                                 | (df_B_C_BB_CC_RU_M.rate == 0)) \
                            & (df_B_C_BB_CC_RU_M.prices == price) \
                            & (df_B_C_BB_CC_RU_M.algo == algo)     
        
        mask_al_pr_ra_prod_cons = ((df_arr_M_T_Ks.rate == str(rate)) 
                                   | (df_arr_M_T_Ks.rate == 0)) \
                                    & (df_arr_M_T_Ks.prices == price) \
                                    & (df_arr_M_T_Ks.algo == algo)  
        
        df_al_pr_ra = df_B_C_BB_CC_RU_M[mask_al_pr_ra].copy()
        
        df_prod_cons = df_arr_M_T_Ks[mask_al_pr_ra_prod_cons].copy()
        df_PROD_CONS = compute_CONS_PROD(df_prod_cons, algo, 
                                         path_2_best_learning_steps)
        print("{}: df_PROD_CONS={}, df_prod_cons={} ".format(
            algo, df_PROD_CONS.shape, df_prod_cons.shape))
        # merge on column pl_i
        df_res = pd.merge(df_al_pr_ra, df_PROD_CONS, on="pl_i")
        pxs_al_pr_ra = plot_players_utility(
                            df_res, algo, rate, price,
                            path_2_best_learning_steps)
        pxs_al_pr_ra.legend.click_policy="hide"
        ######################################################################
        
        if (algo, price, rate) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate)] \
                = [pxs_al_pr_ra]
        else:
            dico_pxs[(algo, price, rate)].append(pxs_al_pr_ra)
        
    rows_RU_CONS_PROD_ts = list()
    for key, pxs_al_pr_ra in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra)
        rows_RU_CONS_PROD_ts.append(col_px_sts)
    rows_RU_CONS_PROD_ts=column(children=rows_RU_CONS_PROD_ts, 
                                sizing_mode='stretch_both')
    return rows_RU_CONS_PROD_ts

# _____________________________________________________________________________
#
#                   utility of players for periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#                   utility for periods ---> debut
# _____________________________________________________________________________
    
def plot_utility(df_res, rate, price,
                 path_2_best_learning_steps):
    """
    plot the bar plot of each player relying on the real utility RU
    """
    
    idx = df_res.algo.unique().tolist()
    cols = ['B', 'C', 'BB', 'CC', 'RU']
    df_res[cols] = df_res[cols].astype(float)
    df_grouped = df_res.groupby(by='algo')[cols] \
                    .agg([np.min, np.max, np.std, np.mean])
    df_grouped.columns = ["_".join(x) for x in df_grouped.columns.ravel()]
    df_grouped = df_grouped.reset_index()
    aggs = ["amin", "amax", "std", "mean"]
    tooltips = [("{}_{}".format(col, agg), "@{}_{}".format(col, agg)) 
                for (col, agg) in it.product(cols, aggs)]
    TOOLS[7] = HoverTool(tooltips = tooltips)
    
    new_cols = [col[1].split("@")[1] 
                for col in tooltips if col[1].split("_")[1] == "mean"] 
    
    px = figure(x_range=idx, 
                y_range=(0, df_grouped[new_cols].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "utility (rate:{}, price={})".format(rate, price)
    px.title.text = title
           
    source = ColumnDataSource(data = df_grouped)
    
    width= 0.2 #0.5
    px.vbar(x=dodge('algo', -0.3, range=px.x_range), top=new_cols[0], 
                    width=width, source=source,
                    color="#2E8B57", legend_label=new_cols[0])
    px.vbar(x=dodge('algo', -0.3+1*width, range=px.x_range), top=new_cols[1], 
                    width=width, source=source,
                    color="#718dbf", legend_label=new_cols[1])
    px.vbar(x=dodge('algo', -0.3+2*width, range=px.x_range), top=new_cols[2], 
                   width=width, source=source,
                   color="#e84d60", legend_label=new_cols[2])
    px.vbar(x=dodge('algo', -0.3+3*width, range=px.x_range), top=new_cols[3], 
                   width=width, source=source,
                   color="#ddb7b1", legend_label=new_cols[3])
    px.vbar(x=dodge('algo', -0.3+4*width, range=px.x_range), top=new_cols[4], 
                   width=width, source=source,
                   color="#FFD700", legend_label=new_cols[4])
    
    px.y_range.start = df_grouped.RU_mean.min() - 1
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "algo"
    px.yaxis.axis_label = "values"
    
    return px
    
def plot_utilities_4_periods(df_B_C_BB_CC_RU_M, 
                             path_2_best_learning_steps):
    """
    plot the utility RU, B, C, BB and CC of players.
    plot the utility of each algorithm 
    """
    rates = df_B_C_BB_CC_RU_M.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df_B_C_BB_CC_RU_M.prices.unique().tolist()
    
    dico_pxs = dict()
    for price, rate in it.product( prices, rates):
        
        mask_pr_ra = ((df_B_C_BB_CC_RU_M.rate == str(rate)) 
                                 | (df_B_C_BB_CC_RU_M.rate == 0)) \
                            & (df_B_C_BB_CC_RU_M.prices == price) 
        
        df_res = df_B_C_BB_CC_RU_M[mask_pr_ra].copy()
        
        pxs_pr_ra = plot_utility(
                            df_res, rate, price,
                            path_2_best_learning_steps)
        pxs_pr_ra.legend.click_policy="hide"
        
        if (price, rate) not in dico_pxs.keys():
            dico_pxs[(price, rate)] \
                = [pxs_pr_ra]
        else:
            dico_pxs[(price, rate)].append(pxs_pr_ra)
        
    rows_RU_C_B_CC_BB = list()
    for key, pxs_pr_ra in dico_pxs.items():
        col_px_sts = column(pxs_pr_ra)
        rows_RU_C_B_CC_BB.append(col_px_sts)
    rows_RU_C_B_CC_BB=column(children=rows_RU_C_B_CC_BB, 
                                sizing_mode='stretch_both')
    return rows_RU_C_B_CC_BB

# _____________________________________________________________________________
#
#                   utility for periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#               Evolution of players' PROD, CONS over periods ---> debut
# _____________________________________________________________________________
def compute_CONS_PROD_NEW(df_prod_cons, algo, path_2_best_learning_steps):
    """
    compute the CONS and PROD for players 
    return 3 dfs such as:
        df_PROD_CONS : contains sum of players' PROD and CONS
        df_PROD: contains players' PROD by time, shape: (m_players, t_periods)
        df_CONS: contains players' CONS by time, shape: (m_players, t_periods)
    """
    k_stop = None;
    path_2_best_learning = None
    if algo in ["LRI1", "LRI2"]:
        for path_2_best in path_2_best_learning_steps:
            if algo == path_2_best[3]:
                path_2_best_learning = os.path.join(*path_2_best)        
        df_tmp = pd.read_csv(os.path.join(path_2_best_learning,
                                           "best_learning_steps.csv"), 
                             sep=',', index_col=0)
    else:
        k_step_max=df_prod_cons.k.unique().max()
        t_periods = df_prod_cons.t.unique().tolist()
        dico = dict()
        for t in t_periods:
            dico[str(t)] = {"k_stop":k_step_max}
        df_tmp = pd.DataFrame.from_dict(dico)
        
    t_periods = df_prod_cons.t.unique().tolist()
    list_of_players = df_prod_cons.pl_i.unique().tolist()
    list_of_players_str = list(map(str,list_of_players))
    t_periods_str = list(map(str, t_periods))
    
    cols = ['pl_i', 'PROD_i', 'CONS_i', 'set']
    df_PROD_CONS = pd.DataFrame(columns=cols, 
                                index=list_of_players)
    df_PROD = pd.DataFrame(columns=t_periods_str, 
                           index=list_of_players_str)
    df_CONS = pd.DataFrame(columns=t_periods, 
                           index=list_of_players_str)
    
    for num_pl_i in list_of_players:
        sum_prod_pl_i = 0; sum_cons_pl_i = 0; set_i = None
        for t in t_periods:
            k_stop = df_tmp.loc["k_stop",str(t)]
            mask_pli_kstop = (df_prod_cons.t == t) \
                             & (df_prod_cons.k == k_stop) \
                             & (df_prod_cons.pl_i == num_pl_i)
            prod_i = df_prod_cons[mask_pli_kstop]["prod_i"].values[0]
            cons_i = df_prod_cons[mask_pli_kstop]["cons_i"].values[0]
            set_i = df_prod_cons[mask_pli_kstop]["set"].values[0]
            
            sum_prod_pl_i += prod_i
            sum_cons_pl_i += cons_i
            
            df_PROD.loc[str(num_pl_i), str(t)] = prod_i
            df_CONS.loc[str(num_pl_i), str(t)] = cons_i
        
        df_PROD_CONS.loc[num_pl_i, "PROD_i"] = sum_prod_pl_i
        df_PROD_CONS.loc[num_pl_i, "CONS_i"] = sum_cons_pl_i
        df_PROD_CONS.loc[num_pl_i, "pl_i"] = num_pl_i
        df_PROD_CONS.loc[num_pl_i, "set"] = set_i
        
    # turn df_PROD to dico with key="player_num_pl_i", value=dico of time
    dico_players_PROD, dico_players_CONS = dict(), dict()
    for num_pl_i in df_PROD.index.tolist():
        dico_t_PROD = {"t":[],"PROD_i":[],"pl_i":[]}
        dico_t_CONS = {"t":[],"CONS_i":[],"pl_i":[]}
        for t in df_PROD.columns.tolist(): 
            dico_t_PROD["t"].append(t)
            dico_t_PROD["pl_i"].append(num_pl_i)
            dico_t_PROD["PROD_i"].append(df_PROD.loc[num_pl_i, t])
            
            dico_t_CONS["t"].append(t)
            dico_t_CONS["pl_i"].append(num_pl_i)
            dico_t_CONS["CONS_i"].append(df_CONS.loc[num_pl_i, t])
        dico_players_PROD["player_"+str(num_pl_i)] = dico_t_PROD
        dico_players_CONS["player_"+str(num_pl_i)] = dico_t_CONS
        
            
    return df_PROD_CONS, df_PROD, df_CONS, dico_players_PROD, dico_players_CONS
    
def plot_evolution_PROD_CONS(df_PROD, df_CONS, 
                             dico_players_PROD, dico_players_CONS,
                             algo, rate, price,
                             path_2_best_learning_steps):
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("pl_i", "@pl_i"),
                            ("t", "@t"),
                            ("PROD_i", "@PROD_i"),
                            ("CONS_i", "@CONS_i")
                            ]
                        )
    
    px_PROD = figure(plot_height = int(HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    px_CONS = figure(plot_height = int(HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    
    title_PROD = "{}: evolution PROD' players (rate:{}, price={})".format(
                    algo, rate, price)
    title_CONS = "{}: evolution CONS' players (rate:{}, price={})".format(
                    algo, rate, price)
    px_PROD.title.text = title_PROD
    px_CONS.title.text = title_CONS
           
    print("{}: df_PROD={}, df_CONS={}".format(
          algo, df_PROD.shape, df_CONS.shape))
    
    ind_color = 0
    for player_i, dico_t_PROD in dico_players_PROD.items():
        ind_color += 10
        source_PROD = ColumnDataSource(data = dico_t_PROD)
        px_PROD.line(x="t", y="PROD_i", source=source_PROD, 
                legend_label=player_i,
                line_width=2, color=Viridis256[ind_color%256], 
                line_dash=[0,0])
    ind_color = 0
    for player_i, dico_t_CONS in dico_players_CONS.items():
        ind_color += 10
        source_CONS = ColumnDataSource(data = dico_t_CONS)
        px_CONS.line(x="t", y="CONS_i", source=source_CONS, 
                legend_label=player_i,
                line_width=2, color=Viridis256[ind_color%256], 
                line_dash=[0,0])
        
    px_CONS.legend.click_policy="hide"
    px_PROD.legend.click_policy="hide"
    px_PROD.xaxis.axis_label = "periods"
    px_CONS.xaxis.axis_label = "periods"
    px_PROD.yaxis.axis_label = "prod_i"
    px_CONS.yaxis.axis_label = "cons_i"
    
    return px_PROD, px_CONS
        
    
def plot_evolution_players_PROD_CONS_over_time(df_arr_M_T_Ks,
                                       path_2_best_learning_steps):
    """
    show the evolution of players' PROD, CONS over the time
    """
    
    rates = df_arr_M_T_Ks.rate.unique().tolist(); rate = rates[rates!=0]
    prices = df_arr_M_T_Ks.prices.unique().tolist()
    algos = df_arr_M_T_Ks.algo.unique().tolist()
    
    dico_pxs = dict()
    for algo, price in it.product(algos, prices):
        mask_al_pr_ra_prod_cons = ((df_arr_M_T_Ks.rate == str(rate)) 
                                   | (df_arr_M_T_Ks.rate == 0)) \
                                    & (df_arr_M_T_Ks.prices == price) \
                                    & (df_arr_M_T_Ks.algo == algo)  
        
        df_prod_cons = df_arr_M_T_Ks[mask_al_pr_ra_prod_cons].copy()
        df_PROD_CONS, df_PROD, df_CONS, dico_players_PROD, dico_players_CONS \
            = compute_CONS_PROD_NEW(df_prod_cons, algo, 
                                    path_2_best_learning_steps)
        
        pxs_PROD, pxs_CONS = plot_evolution_PROD_CONS(
                                df_PROD, df_CONS, 
                                dico_players_PROD, dico_players_CONS,
                                algo, rate, price,
                                path_2_best_learning_steps)
        pxs_PROD.legend.click_policy="hide"
        pxs_CONS.legend.click_policy="hide"
        
        if (algo, price, rate) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate)] \
                = [pxs_PROD, pxs_CONS]
        else:
            dico_pxs[(algo, price, rate)].extend([pxs_PROD, pxs_CONS])
        
    rows_CONS_PROD_ts = list()
    for key, pxs_CONS_PROD in dico_pxs.items():
        col_px_sts = column(pxs_CONS_PROD)
        rows_CONS_PROD_ts.append(col_px_sts)
    rows_CONS_PROD_ts=column(children=rows_CONS_PROD_ts, 
                                sizing_mode='stretch_both')
    return rows_CONS_PROD_ts
    
# _____________________________________________________________________________
#
#               Evolution of players' PROD, CONS over periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#               Evolution of PROD, CONS over periods ---> debut
# _____________________________________________________________________________
def plot_evolution_IN_OUT_SG_by_algo(df_IN_OUT_SG,
                                     algo, rate, price):
    df_IN_OUT_SG["t"] = df_IN_OUT_SG["t"].astype(str)
    idx = df_IN_OUT_SG.t.unique().tolist()
    # cols = ['k', 'PROD', 'CONS']
    cols = ['k', 'IN_sg', 'OUT_sg']
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("t", "@t"),
                            ("IN_sg", "@IN_sg"),
                            ("OUT_sg", "@OUT_sg")
                            ]
                        )
    px = figure(x_range=idx, 
                y_range=(0, df_IN_OUT_SG[cols[1:]].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "{}: IN_sg, OUT_sg (rate:{}, price={})".format(algo, rate, price)
    px.title.text = title
           
    source = ColumnDataSource(data = df_IN_OUT_SG)
    
    width= 0.2 #0.5
    px.vbar(x=dodge('t', -0.3, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#2E8B57", legend_label=cols[1])
    px.vbar(x=dodge('t', -0.3+width, range=px.x_range), top=cols[2], 
                    width=width, source=source,
                    color="#718dbf", legend_label=cols[2])
    
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "players"
    px.yaxis.axis_label = "values"
    
    return px

def compute_IN_OUT_SG_by_periods(df_prod_cons, algo, df_LRI_12):
    cols = ['t','k','IN_sg','OUT_sg']
    t_periods = list(df_prod_cons.t.unique())
    df_IN_OUT_SG = pd.DataFrame(index=t_periods, columns=cols)
    for t in t_periods:
        k_max = None
        if algo in ['LRI1', 'LRI2']:
            k_max = df_LRI_12.loc[algo+'_k_stop',str(t)]
        else:
            k_max = df_prod_cons.k.max()
        mask_t_kmax = (df_prod_cons.k == k_max) & (df_prod_cons.t == t)
        df_t_k = df_prod_cons[mask_t_kmax]
        IN_sg = df_t_k.prod_i.sum()
        OUT_sg = df_t_k.cons_i.sum()
        df_IN_OUT_SG.loc[t, cols[0]] = t
        df_IN_OUT_SG.loc[t, cols[1]] = k_max
        df_IN_OUT_SG.loc[t, cols[2]] = IN_sg
        df_IN_OUT_SG.loc[t, cols[3]] = OUT_sg
        
    return df_IN_OUT_SG
        
    
def plot_evolution_IN_OUT_SG_over_time(df_arr_M_T_Ks,
                                       df_LRI_12):
    """
    show the evolution of PROD, CONS over the time
    """
    
    rates = df_arr_M_T_Ks.rate.unique().tolist(); rate = rates[rates!=0]
    prices = df_arr_M_T_Ks.prices.unique().tolist()
    algos = df_arr_M_T_Ks.algo.unique().tolist()
    
    dico_pxs = dict()
    for algo, price in it.product(algos, prices):
        mask_al_pr_ra = ((df_arr_M_T_Ks.rate == str(rate)) 
                                   | (df_arr_M_T_Ks.rate == 0)) \
                            & (df_arr_M_T_Ks.prices == price) \
                            & (df_arr_M_T_Ks.algo == algo)
        df_prod_cons = df_arr_M_T_Ks[mask_al_pr_ra].copy()
        df_IN_OUT_SG = compute_IN_OUT_SG_by_periods(
                            df_prod_cons, algo, 
                            df_LRI_12)
        
        pxs_IN_OUT_SG = plot_evolution_IN_OUT_SG_by_algo(
                            df_IN_OUT_SG,
                            algo, rate, price)
        pxs_IN_OUT_SG.legend.click_policy="hide"
        
        if (algo, price, rate) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate)] \
                = [pxs_IN_OUT_SG]
        else:
            dico_pxs[(algo, price, rate)].extend([pxs_IN_OUT_SG])
        
    rows_IN_OUT_SG_ts = list()
    for key, pxs_CONS_PROD in dico_pxs.items():
        col_px_sts = column(pxs_CONS_PROD)
        rows_IN_OUT_SG_ts.append(col_px_sts)
    rows_IN_OUT_SG_ts=column(children=rows_IN_OUT_SG_ts, 
                             sizing_mode='stretch_both')
    return rows_IN_OUT_SG_ts
    
# _____________________________________________________________________________
#
#               Evolution of PROD, CONS over periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#               Evolution of pi_sg, b0, c0 over periods ---> debut
# _____________________________________________________________________________
def plot_price_by_algo(df_ra_pri_al, df_LRI_12, rate, price, algo):
    cols = df_ra_pri_al.columns[4:]
    t_periods = list(df_ra_pri_al.t.unique())
    df_t = pd.DataFrame(index=t_periods, columns=cols)
    for t in t_periods:
        k_max = None
        if algo in ['LRI1', 'LRI2']:
            k_max = df_LRI_12.loc[algo+'_k_stop',str(t)]
        else:
            k_max = df_ra_pri_al.k.max()
        index = df_ra_pri_al[(df_ra_pri_al.k == k_max) 
                             & (df_ra_pri_al.t == t)].index[0]
        for col in cols:
            df_t.loc[t, col] = df_ra_pri_al.loc[index, col]
    df_t = df_t.reset_index()
    df_t.rename(columns={"k":"k_max", 'index':"t"}, inplace=True)
    df_t["t"] = df_t["t"].astype(str)
    
    TOOLS[7] = HoverTool(tooltips=[
                    ("t", "@t"),
                    ("k", "@k_max"),
                    ("b0", "@b0"), ("c0", "@c0"),
                    ("pi_0_-", "@pi_0_minus"), ("pi_0_+", "@pi_0_plus"),
                    ("pi_sg_-", "@pi_sg_minus"), ("pi_sg_+", "@pi_sg_plus")
                    ]
                )
    idx = df_t.t.unique().tolist()
    px = figure(x_range=idx, 
                y_range=(0, df_t[cols[1:]].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "{}: prices over time (rate:{}, price={})".format(
                    algo, rate, price)
    px.title.text = title
    
    source = ColumnDataSource(data = df_t)
    
    width= 0.1 #0.5
    px.vbar(x=dodge('t', -0.3, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#2E8B57", legend_label=cols[1])
    px.vbar(x=dodge('t', -0.3+1*width, range=px.x_range), top=cols[2], 
                    width=width, source=source,
                    color="#718dbf", legend_label=cols[2])
    px.vbar(x=dodge('t', -0.3+2*width, range=px.x_range), top=cols[3], 
                   width=width, source=source,
                   color="#e84d60", legend_label=cols[3])
    px.vbar(x=dodge('t', -0.3+3*width, range=px.x_range), top=cols[4], 
                   width=width, source=source,
                   color="#ddb7b1", legend_label=cols[4])
    px.vbar(x=dodge('t', -0.3+4*width, range=px.x_range), top=cols[5], 
                   width=width, source=source,
                   color="#FFD700", legend_label=cols[5])
    px.vbar(x=dodge('t', -0.3+5*width, range=px.x_range), top=cols[6], 
                   width=width, source=source,
                   color="#d95f0e", legend_label=cols[6])
    
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "t_periods"
    px.yaxis.axis_label = "values"
    
    return px
    
    
def plot_evolution_over_time_PISG_b0c0(df_arr_M_T_Ks, 
                                       df_b0_c0_pisg_pi0_T_K,
                                       df_LRI_12):
    rates = df_b0_c0_pisg_pi0_T_K["rate"].unique(); rates = rates[rates!=0]
    prices = df_b0_c0_pisg_pi0_T_K["prices"].unique()
    algos = df_b0_c0_pisg_pi0_T_K["algo"].unique()
    
    dico_pxs = dict()
    cpt = 0
    for rate, price, algo in it.product(rates, prices, algos):
        
        mask_ra_pri_al = ((df_b0_c0_pisg_pi0_T_K.rate == rate) 
                                 | (df_b0_c0_pisg_pi0_T_K.rate == 0)) \
                            & (df_b0_c0_pisg_pi0_T_K.prices == price) \
                            & (df_b0_c0_pisg_pi0_T_K.algo == algo)
        df_ra_pri_al = df_b0_c0_pisg_pi0_T_K[mask_ra_pri_al].copy()
        
        px_al = plot_price_by_algo(df_ra_pri_al, df_LRI_12, rate, price, algo)
        px_al.legend.click_policy="hide"
        
        if (rate, price, algo) not in dico_pxs.keys():
            dico_pxs[(rate, price, algo)] \
                = [px_al]
        else:
            dico_pxs[(rate, price, algo)].append(px_al)
        cpt += 1                            
        
    # aggregate by algo i.e each algo is on new column.
    col_px_als = []
    for key, px_als in dico_pxs.items():
        row_px_als = row(px_als)
        col_px_als.append(row_px_als)
    col_px_als=column(children=col_px_als, 
                      sizing_mode='stretch_both')
    return col_px_als
    
# _____________________________________________________________________________
#
#               Evolution of pi_sg, b0, c0 over periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#           affichage Perf_t pour chaque algo  ---> debut
# _____________________________________________________________________________
def plot_Perf_t_all_algos(df_ra_pri, rate, price, t, df_LRI_12):
                                
    """
    plot the Perf_t at each learning step for all states 
    considering all scenarios and all algorithms.
    each figure is for one state, one scenario, one prob_Ci, one learning_rate 
    and one price.
    
    Perf_t = \sum\limits_{1\leq i \leq N} ben_i-cst_i
    
    x-axis : one time t
    y-axis : Perf_t
    """
    algos = df_ra_pri["algo"].unique()
    
    tup_legends = [] 
    
    px = figure(plot_height = int(HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    
    for algo in algos:
        df_al = None
        if algo in ["LRI1", "LRI2"]:
            k_stop_algo = df_LRI_12.loc[algo+"_k_stop", str(t)]
            df_al = df_ra_pri[(df_ra_pri.algo == algo) \
                              & (df_ra_pri.k <= k_stop_algo )]
        else:
            df_al = df_ra_pri[(df_ra_pri.algo == algo)]
        
            
        title = "Perf_t at t={} (rate={}, price={})".format(
                t, rate, price)
        xlabel = "k step of learning" 
        ylabel = "Perf_t" #"ben_i-cst_i"
        label = "{}".format(algo)
        
        px.title.text = title
        px.xaxis.axis_label = xlabel
        px.yaxis.axis_label = ylabel
        TOOLS[7] = HoverTool(tooltips=[
                            ("algo", "@algo"),
                            ("k", "@k"),
                            (ylabel, "$y")
                            ]
                        )
        px.tools = TOOLS
        
        cols = ['ben','cst']
        # TODO lauch warning See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy self[k1] = value[k2]
        df_al[cols] = df_al[cols].apply(pd.to_numeric, 
                                        downcast='float', 
                                        errors='coerce')
                
        df_al_k = df_al.groupby(by=["k","pl_i"])[cols]\
                    .aggregate(np.sum)\
                    .apply(lambda x: x[0]-x[1], axis=1).reset_index()
        df_al_k.rename(columns={0:ylabel}, inplace=True)
        df_al_k = df_al_k.groupby("k")[ylabel].aggregate(np.sum).reset_index()  
        
        df_al_k.loc[:,"algo"] = algo
        df_al_k.loc[:,"t"] = t
        source = ColumnDataSource(data = df_al_k)

        ind_color = 0
        if algo == "LRI1":
            ind_color = 1 #10
        elif algo == "LRI2":
            ind_color = 2 #10
        elif algo == "DETERMINIST":
            ind_color = 3 #10
        elif algo == "RD-DETERMINIST":
            ind_color = 4 #10
        elif algo == "BEST-BRUTE-FORCE":
            ind_color = 5 #10
        elif algo == "BAD-BRUTE-FORCE":
            ind_color = 6 #10
        elif algo == "MIDDLE-BRUTE-FORCE":
            ind_color = 7 #10
        elif algo == fct_aux.ALGO_NAMES_NASH[0]:                                # "BEST-NASH"
            ind_color = 8 #10
        elif algo == fct_aux.ALGO_NAMES_NASH[1]:                                # "BAD-NASH"
            ind_color = 9 #10
        elif algo == fct_aux.ALGO_NAMES_NASH[2]:                                # "MIDDLE-NASH"
            ind_color = 10 #10
            
        r1 = px.line(x="k", y=ylabel, source=source, legend_label=label,
                line_width=2, color=COLORS[ind_color], 
                line_dash=[0,0])
        
        nb_k_steps = len(list(df_al_k['k'].unique()))
        interval = int(nb_k_steps*10/250)
        print(".... nb_k_steps={}, interval={} .... ".format(nb_k_steps, interval))
        # ls = None
        # if nb_k_steps > interval and interval > 0:
        #     ls = range(0,nb_k_steps,interval)
        # if nb_k_steps > interval and interval <= 0:
        #     ls = range(0, nb_k_steps)
        # else:
        #     ls = range(0, nb_k_steps)
        # # ls = range(0,nb_k_steps,interval) \
        # #             if nb_k_steps < interval \
        # #             else range(0, nb_k_steps) 
        # # ls = range(0,nb_k_steps,int(nb_k_steps*10/250))
        # if int(nb_k_steps*10/250) > 0:
        #     ls = range(0,nb_k_steps,int(nb_k_steps*10/250))
        # else:
        #     ls = range(0,nb_k_steps,1)
        # df_al_slice = df_al[df_al.index.isin(ls)]
        # source_slice = ColumnDataSource(data = df_al_slice)
        
        if algo == "LRI1":
            ind_color = 1
            r2 = px.asterisk(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
            # tup_legends.append((label, [r2] ))
        elif algo == "LRI2":
            ind_color = 2
            r2 = px.circle(x="k", y=ylabel, size=7, source=source, 
                      color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
            # tup_legends.append((label, [r2] ))
        elif algo == "DETERMINIST":
            ind_color = 3
            r2 = px.triangle_dot(x="k", y=ylabel, size=7, source=source, 
                      color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
            # tup_legends.append((label, [r2] ))
        elif algo == "RD-DETERMINIST":
            ind_color = 4
            r2 = px.triangle(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
            # tup_legends.append((label, [r2] ))
        elif algo == "BEST-BRUTE-FORCE":
            r2 = px.diamond(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == "BAD-BRUTE-FORCE":
            r2 = px.diamond_cross(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == "MIDDLE-BRUTE-FORCE":
            r2 = px.diamond_dot(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == fct_aux.ALGO_NAMES_NASH[0]:                                # "BEST-NASH"
            r2 = px.square_cross(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == fct_aux.ALGO_NAMES_NASH[1]:                                # "BAD-NASH"
            r2 = px.square_pin(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        elif algo == fct_aux.ALGO_NAMES_NASH[2]:                                # "MIDDLE-NASH"
            r2 = px.square_x(x="k", y=ylabel, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label)
            tup_legends.append((label, [r1,r2] ))
        
    legend = Legend(items= tup_legends, location="center")
    px.legend.label_text_font_size = "8px"
    px.legend.click_policy="hide"
    px.add_layout(legend, 'right') 

    return px               

def plot_Perf_t_players_all_algos(df_ben_cst_M_T_K, t, 
                                  df_LRI_12, df_k_stop):
    """
    plot the Perf_t of players at each learning step k for all prob_Ci, price, 
    learning_rate for any state
    
    Perf_t = \sum\limits_{1\leq i \leq N} ben_i-cst_i
    """
    rates = df_ben_cst_M_T_K["rate"].unique(); rates = rates[rates!=0]
    prices = df_ben_cst_M_T_K["prices"].unique()
    
    dico_pxs = dict()
    cpt = 0
    for rate, price in it.product(rates, prices):
        
        mask_ra_pri_st = ((df_ben_cst_M_T_K.rate == rate) 
                                 | (df_ben_cst_M_T_K.rate == 0)) \
                            & (df_ben_cst_M_T_K.prices == price) \
                            & (df_ben_cst_M_T_K.t == t) \
                            & (df_ben_cst_M_T_K.k <= df_k_stop.loc["k_stop",str(t)])
        df_ra_pri = df_ben_cst_M_T_K[mask_ra_pri_st].copy()
        
        px_st = plot_Perf_t_all_algos(df_ra_pri, rate, price, t, df_LRI_12)
        # return px_scen_st
        px_st.legend.click_policy="hide"
        
        if (rate, price) not in dico_pxs.keys():
            dico_pxs[(rate, price)] \
                = [px_st]
        else:
            dico_pxs[(rate, price)].append(px_st)
        cpt += 1                            
        
    # aggregate by state_i i.e each state_i is on new column.
    col_px_sts = []
    for key, px_sts in dico_pxs.items():
        row_px_sts = row(px_sts)
        col_px_sts.append(row_px_sts)
    col_px_sts=column(children=col_px_sts, 
                      sizing_mode='stretch_both')
    return col_px_sts

    # # aggregate by state_i i.e each state_i is on new column.
    # row_px_sts = []
    # for key, px_sts in dico_pxs.items():
    #     col_px_sts = column(px_sts)
    #     row_px_sts.append(col_px_sts)
    # row_px_sts = row(children=row_px_sts, sizing_mode='stretch_both')
        
    # return  row_px_sts 
# _____________________________________________________________________________
#
#           affichage Perf_t pour chaque state  ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________ 
#
#         moyenne de la grande proba dune mode pour chaque state  ---> debut
# _____________________________________________________________________________ 
              
def plot_max_proba_mode_onestate(df_ra_pri_st, 
                                rate, price, state_i, t, 
                                path_2_best_learning_steps,
                                algos):
    
    tup_legends = [] 
    
    px = figure(plot_height = int(HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    
    algo_df = set(df_ra_pri_st["algo"].unique())
    algos = set(algos).intersection(algo_df)
    
    for algo in algos:
        path_2_best_learning = None
        for path_2_best in path_2_best_learning_steps:
            if algo == path_2_best[3]:
                path_2_best_learning = os.path.join(*path_2_best)
            
        print("path_2_best_learning={}".format(path_2_best_learning))
        df_tmp = pd.read_csv(os.path.join(path_2_best_learning,
                                           "best_learning_steps.csv"), 
                             sep=',', index_col=0)
        k_stop = df_tmp.loc["k_stop",str(t)]
            
        df_al = df_ra_pri_st[(df_ra_pri_st.algo == algo) 
                             & (df_ra_pri_st.k <= k_stop)]
        
        title = "mean of max proba for each mode at {}, t={} (rate={}, price={})".format(
                state_i, t, rate, price)
        xlabel = "k step of learning" 
        ylabel_moyS1 = "moy_S1"; 
        ylabel_moyS2 = "moy_S2";
        label_moyS2 = "moy_S2_{}".format(algo)
        label_moyS1 = "moy_S1_{}".format(algo)
        
        # label_stdS1 = "std_S1_{}".format(algo)
        # label_upperS1 = "upper_S1_{}".format(algo)
        # label_lowerS1 = "lower_S1_{}".format(algo)
        # ylabel_stdS2 = "std_S2"; 
        # ylabel_upperS2 = "upper_S2";
        # ylabel_lowerS2 = "lower_S2";
        # ylabel_moyMaxS12 = "moy_max_S12"
        # ylabel_stdS1 = "std_S1";
        # ylabel_upperS1 = "upper_S1";
        # ylabel_lowerS1 = "lower_S1";
        # label_stdS2 = "std_S2_{}".format(algo)
        # label_upperS2 = "upper_S2_{}".format(algo)
        # label_lowerS2 = "lower_S2_{}".format(algo)
        # label_moyMaxS12 = "moy_max_S12_{}".format(algo)
        
        px.title.text = title
        px.xaxis.axis_label = xlabel
        px.yaxis.axis_label = "moy mode"
        TOOLS[7] = HoverTool(tooltips=[
                            ("algo", "@algo"),
                            ("k", "@k"),
                            ("moy_S1", "@moy_S1"),
                            ("moy_S2", "@moy_S2"),
                            ("lower_S1", "@lower_S1"),
                            ("upper_S1", "@upper_S1"),
                            ("lower_S2", "@lower_S2"),
                            ("upper_S2", "@upper_S2"),
                            ("std_S1", "@std_S1"), 
                            ("std_S2", "@std_S2"),
                            ("min_S1", "@min_S1"), 
                            ("max_S1", "@max_S1"), 
                            ("min_S2", "@min_S2"),
                            ("max_S2", "@max_S2"),
                            ("moy_max_S12", "@moy_max_S12"),
                            ("S1", "@S1"),
                            ("S2", "@S2"),
                            ]
                        )
        px.tools = TOOLS
        
        S1, S2 = None, None
        if state_i == "state1":
            S1 = fct_aux.STATE1_STRATS[0]
            S2 = fct_aux.STATE1_STRATS[1]
        elif state_i == "state2":
            S1 = fct_aux.STATE2_STRATS[0]
            S2 = fct_aux.STATE2_STRATS[1]
        elif state_i == "state3":
            S1 = fct_aux.STATE3_STRATS[0]
            S2 = fct_aux.STATE3_STRATS[1]
        
        df_al["S1_p_i_j_k"] = df_al["S1_p_i_j_k"]\
                                .apply(pd.to_numeric,
                                       downcast='float',
                                       errors='coerce')
        df_al["S2_p_i_j_k"] = df_al["S2_p_i_j_k"]\
                                .apply(pd.to_numeric,
                                       downcast='float',
                                       errors='coerce')
        # probleme ICI car tous les p_i_j_k ne correspondent pas a S1
        df_al.loc[:,"S1"] = df_al["S1_p_i_j_k"].copy()
        df_al.loc[:,"S2"] = df_al["S2_p_i_j_k"].copy()
        df_al.loc[:,"S1"] = df_al["S1"].apply(pd.to_numeric,
                                        downcast='float',
                                        errors='coerce').copy()
        df_al.loc[:,"S2"] = df_al["S2"].apply(pd.to_numeric,
                                        downcast='float',
                                        errors='coerce').copy()
        #print("S1={}".format(df_al["S1"]))
        print("S1={}, df_al={}, rate={}, price={}, state_i={}, t={}, algo={}".format(
                df_al["S1"].shape, df_al.shape, rate, price, state_i, t, algo))
        df_al_k = df_al.groupby("k")[["S1","S2"]]\
                    .aggregate(np.mean).reset_index()
        df_al_k = df_al.groupby("k")[["S1","S2"]]\
                        .agg({"S1": [np.mean, np.std, np.min, np.max], 
                              "S2": [np.mean, np.std, np.min, np.max]})\
                        .reset_index()    
        tuple_cols = list(df_al_k.columns)
        dico_new_cols = dict()
        for tu_col in tuple_cols:
            dico_new_cols[tu_col] = tu_col[1]+"_"+tu_col[0]
        df_al_k.columns = df_al_k.columns.to_flat_index()
        df_al_k.rename(columns=dico_new_cols, inplace=True)  
        
        df_al_k['lower_S1'] = df_al_k.mean_S1 - df_al_k.std_S1
        df_al_k['upper_S1'] = df_al_k.mean_S1 + df_al_k.std_S1
        
        df_al_k['lower_S2'] = df_al_k.mean_S2 - df_al_k.std_S2
        df_al_k['upper_S2'] = df_al_k.mean_S2 + df_al_k.std_S2
        
        dico_rename = {"_k":"k","mean_S1":"moy_S1", "mean_S2":"moy_S2",
                       "std_S1":"std_S1", "std_S2":"std_S2",
                       "amin_S1":"min_S1", "amin_S2":"min_S2",
                       "amax_S1":"max_S1", "amax_S2":"max_S2"
                       }
        df_al_k.rename(columns=dico_rename, inplace=True)
        
        df_al_k.loc[:,"algo"] = algo
        df_al_k.loc[:,"t"] = t
        df_al_k.loc[:,"S1"] = S1
        df_al_k.loc[:,"S2"] = S2
        
        source = ColumnDataSource(data = df_al_k)
        
        ind_color = 0
        r1 = px.line(x="k", y=ylabel_moyS1, source=source, 
                     legend_label=label_moyS1,
                     line_width=2, color=COLORS[ind_color], 
                     line_dash=[0,0])
        ind_color = 1
        r2 = px.line(x="k", y=ylabel_moyS2, source=source, 
                     legend_label=label_moyS2,
                     line_width=2, color=COLORS[ind_color], 
                     line_dash=[0,0])
        band_S1 = Band(base='k', lower='lower_S1', upper='upper_S1', 
                source=source, level='underlay',
                fill_alpha=1.0, line_width=1, line_color='black')
        band_S2 = Band(base='k', lower='lower_S2', upper='upper_S2', 
                source=source, level='underlay',
                fill_alpha=1.0, line_width=1, line_color='black')
        #px.add_layout(band)
        px.add_layout(band_S1)
        px.add_layout(band_S2)
                        
        
        if algo == "LRI1":
            ind_color = 3
            r4 = px.asterisk(x="k", y=ylabel_moyS1, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label_moyS1)
            r5 = px.asterisk(x="k", y=ylabel_moyS2, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label_moyS2)
            # r6 = px.asterisk(x="k", y=ylabel_moyMaxS12, size=7, source=source, 
            #             color=COLORS[ind_color], legend_label=label_moyMaxS12)
            tup_legends.append((algo, [r1,r2,r4,r5] ))
            # tup_legends.append((algo, [r1,r2,r3,r4,r5,r6] ))
        elif algo == "LRI2":
            ind_color = 4
            r4 = px.asterisk(x="k", y=ylabel_moyS1, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label_moyS1)
            r5 = px.asterisk(x="k", y=ylabel_moyS2, size=7, source=source, 
                        color=COLORS[ind_color], legend_label=label_moyS2)
            # r6 = px.asterisk(x="k", y=ylabel_moyMaxS12, size=7, source=source, 
            #             color=COLORS[ind_color], legend_label=label_moyMaxS12)
            tup_legends.append((algo, [r1,r2,r4,r5] ))
            # tup_legends.append((algo, [r1,r2,r3,r4,r5,r6] ))
        
    legend = Legend(items= tup_legends, location="center")
    px.legend.label_text_font_size = "8px"
    px.legend.click_policy="hide"
    px.add_layout(legend, 'right') 

    return px                
        
def plot_max_proba_mode(df_arr_M_T_Ks, t, path_2_best_learning_steps, 
                        algos=['LRI1','LRI2']):
    """
    plot the mean of the max probability of one mode. 
    The steps of process are:
        * select all players having state_i = stateX, X in {1,2,3}
        * compute the mean of each mode for each k_step moy_S_1_X, moy_S_2_X
        * select the max of each mode for each k_step max_S_1_X, max_S_2_X
        * compute the mean of max of each mode for each k_step ie
            moy_max_S_12_X = (max_S_1_X - max_S_2_X)/2
        * plot the curves of moy_S_1_X, moy_S_2_X and moy_max_S_12_X
        
    x-axis : k_step
    y-axis : moy

    """
    rates = df_arr_M_T_Ks["rate"].unique(); rates = rates[rates!=0]
    prices = df_arr_M_T_Ks["prices"].unique()
    states = df_arr_M_T_Ks["state_i"].unique()
    
    dico_pxs = dict()
    cpt = 0
    for rate, price, state_i\
        in it.product(rates, prices, states):
        
        mask_ra_pri = ((df_arr_M_T_Ks.rate == rate) 
                                         | (df_arr_M_T_Ks.rate == 0)) \
                            & (df_arr_M_T_Ks.prices == price) \
                            & (df_arr_M_T_Ks.t == t) \
                            & (df_arr_M_T_Ks.state_i == state_i)    
        df_ra_pri = df_arr_M_T_Ks[mask_ra_pri].copy()
        
        if df_ra_pri.shape[0] != 0:
            px_st_mode = plot_max_proba_mode_onestate(
                            df_ra_pri, 
                            rate, price, state_i, 
                            t, path_2_best_learning_steps, algos)
            px_st_mode.legend.click_policy="hide"
    
            if (rate, price, state_i) not in dico_pxs.keys():
                dico_pxs[(rate, price, state_i)] \
                    = [px_st_mode]
            else:
                dico_pxs[(rate, price, state_i)].append(px_st_mode)
            cpt += 1
        
    # aggregate by state_i i.e each state_i is on new column.
    col_px_st_S1S2s = []
    for key, px_st_mode in dico_pxs.items():
        row_pxs = row(px_st_mode)
        col_px_st_S1S2s.append(row_pxs)
    col_px_st_S1S2s=column(children=col_px_st_S1S2s, 
                           sizing_mode='stretch_both')  
    # show(col_px_scen_st_S1S2s)
    
    return col_px_st_S1S2s
     
# _____________________________________________________________________________
#
#        moyenne de la grande proba dune mode pour chaque state  ---> fin
# _____________________________________________________________________________


# _____________________________________________________________________________
#
#        representation de moyenne de Vi a chaque periode  ---> debut
# _____________________________________________________________________________
# def create_dataframe_mean_Vi_for(df_ben_cst_M_T_K, df_LRI_12, k_steps_args, 
#                                  algos_4_learning):
#     """
#     create a dataframe containing the mean of Vi by for loop
#     for each algo at each period
    
#     """
#     df_ben_cst_M_T_K["Vi"] = df_ben_cst_M_T_K["ben"] - df_ben_cst_M_T_K["cst"]
    
#     algos = df_ben_cst_M_T_K.algo.unique().tolist()
#     t_periods = df_ben_cst_M_T_K.t.unique().tolist()
#     rates = df_ben_cst_M_T_K["rate"].unique(); rates = rates[rates!=0]
#     prices = df_ben_cst_M_T_K["prices"].unique()
    
#     dico_algo_t_periods = dict()
#     dico_algo_t_periods["algo"] = []
#     dico_algo_t_periods["t"] = []
#     dico_algo_t_periods["rate"] = []
#     dico_algo_t_periods["price"] = []
#     dico_algo_t_periods["moy_Vi"] = []
    
#     # TODO : l es valeurs de determinist sont tous a NAN
#     for (rate,price,algo) in it.product(rates,prices,algos):
#         for t in t_periods:
#             kmax = df_LRI_12.loc[algo+"_k_stop", str(t)] \
#                         if algo in algos_4_learning \
#                         else k_steps_args-1
#             mask_algo_kmax = (df_ben_cst_M_T_K.algo == algo) \
#                                 & (df_ben_cst_M_T_K.rate == rate) \
#                                 & (df_ben_cst_M_T_K.prices == price) \
#                                 & (df_ben_cst_M_T_K.k == kmax)
#             df_pl_is = df_ben_cst_M_T_K[mask_algo_kmax]
#             mean_Vi = df_pl_is['Vi'].mean()
#             dico_algo_t_periods["algo"].append(algo)
#             dico_algo_t_periods["t"].append(str(t))
#             dico_algo_t_periods["moy_Vi"].append(mean_Vi)
#             dico_algo_t_periods["rate"].append(rate)
#             dico_algo_t_periods["price"].append(price)
            
#         print("mean_Vi_for: {}, price={},rate={} TERMINE".format(algo,price,rate))
        
#     df_algo_t_periods_moyVi = pd.DataFrame.from_dict(dico_algo_t_periods)
    
#     return df_algo_t_periods_moyVi
def create_dataframe_mean_Vi_for(df_ben_cst_M_T_K, df_LRI_12, k_steps_args, 
                                 algos_4_learning):
    """
    create a dataframe containing the mean of Vi by for loop
    for each algo at each period
    
    """
    df_ben_cst_M_T_K["Vi"] = df_ben_cst_M_T_K["ben"] - df_ben_cst_M_T_K["cst"]
    
    algos = df_ben_cst_M_T_K.algo.unique().tolist()
    t_periods = df_ben_cst_M_T_K.t.unique().tolist()
    rates = df_ben_cst_M_T_K["rate"].unique(); rates = rates[rates!=0]
    prices = df_ben_cst_M_T_K["prices"].unique()
    
    dico_algo_t_periods = dict()
    dico_algo_t_periods["algo"] = []
    dico_algo_t_periods["t"] = []
    dico_algo_t_periods["rate"] = []
    dico_algo_t_periods["prices"] = []
    dico_algo_t_periods["moy_Vi"] = []
    dico_algo_t_periods["std_Vi"] = []
    
    for (rate,price,algo) in it.product(rates,prices,algos):
        for t in t_periods:
            kmax = df_LRI_12.loc[algo+"_k_stop", str(t)] \
                        if algo in algos_4_learning \
                        else k_steps_args-1
            mask_algo_kmax = (df_ben_cst_M_T_K.algo == algo) \
                                & ((df_ben_cst_M_T_K.rate == rate)| 
                                   (df_ben_cst_M_T_K.rate == 0))\
                                & (df_ben_cst_M_T_K.prices == price) \
                                & (df_ben_cst_M_T_K.k == kmax) \
                                & (df_ben_cst_M_T_K.t == t)
            df_pl_is = df_ben_cst_M_T_K[mask_algo_kmax]
            mean_Vi = df_pl_is.Vi.mean()
            std_Vi = df_pl_is.Vi.std()
            dico_algo_t_periods["algo"].append(algo)
            dico_algo_t_periods["t"].append(str(t))
            dico_algo_t_periods["moy_Vi"].append(mean_Vi)
            dico_algo_t_periods["rate"].append(rate)
            dico_algo_t_periods["prices"].append(price)
            dico_algo_t_periods["std_Vi"].append(std_Vi)
            
        print("mean_Vi_for: {}, prices={},rate={} TERMINE".format(algo,price,rate))
        
    df_algo_t_periods_moyVi = pd.DataFrame.from_dict(dico_algo_t_periods)
    
    return df_algo_t_periods_moyVi

def plot_bar_meanVi_over_time_one_algo(df_ra_pr, price, rate):
    """
    draw a bar plot of Vi mean.
    the bar plot has period t and algo on x-axis ie key = (t, algo)
        and moyVi on y-axis

    """
    
    cols = ["t", "algo"]
    x = list(map(tuple,list(df_ra_pr[cols].values)))
    moy_Vi = list(df_ra_pr["moy_Vi"])
    TOOLS[7] = HoverTool(tooltips=[
                            ("algo", "@algo"),
                            ("t", "@t"),
                            ("moy_Vi", "@moy_Vi"),
                            ("std_Vi", "@std_Vi")
                            ]
                        )
    
    px= figure(x_range=FactorRange(*x), 
                plot_height=int(HEIGHT*MULT_HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH),
                title="mean of Vi over time (price={},rate={}".format(price,rate),
                toolbar_location="above", tools=TOOLS)

    data = dict(x=x, moy_Vi=moy_Vi, std_Vi=df_ra_pr.std_Vi.tolist(),
                t=df_ra_pr.t.tolist(),
                algo=df_ra_pr.algo.tolist())
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='moy_Vi', width=0.9, source=source, 
            fill_color=factor_cmap('x', 
                                    palette=Category20[20], 
                                    factors=list(df_ra_pr.algo.unique()), 
                                    start=1, end=2))
    
    print("min(moy_Vi)={}".format( df_ra_pr.moy_Vi.min() ))
    px.y_range.start = df_ra_pr.moy_Vi.min() - 1
    #px.y_range.end = df_ra_pr.moy_Vi.max()
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    px.xaxis.axis_label = 't_periods'
    px.yaxis.axis_label = 'moy_Vi'
    
    return px
    

def plot_bar_meanVi_over_time(df_algo_t_periods_moyVi):
    
    algos = df_algo_t_periods_moyVi.algo.unique().tolist()
    rates = df_algo_t_periods_moyVi["rate"].unique(); rates = rates[rates!=0]
    prices = df_algo_t_periods_moyVi["prices"].unique()
    
    dico_pxs = dict()
    for (rate, price) in it.product(rates, prices):
        mask_ra_pr = (df_algo_t_periods_moyVi.rate == rate) \
                            & (df_algo_t_periods_moyVi.prices == price)
        df_ra_pr = df_algo_t_periods_moyVi[mask_ra_pr]
        
        px = plot_bar_meanVi_over_time_one_algo(df_ra_pr, price, rate)
        if (price, rate) not in dico_pxs:
            dico_pxs[(price, rate)] = [px]
        else:
            dico_pxs[(price, rate)].append(px)
            
    col_pxs = []
    for key, pxs in dico_pxs.items():
        row_px_sts = row(pxs)
        col_pxs.append(row_px_sts)
    col_pxs=column(children=col_pxs, 
                   sizing_mode='stretch_both')
    return col_pxs   
# _____________________________________________________________________________
#
#        representation de moyenne de Vi a chaque periode  ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#           distribution de moyenne de Vi a chaque periode  ---> debut
# _____________________________________________________________________________
def plot_distribution_algo(df_al_pr_ra, algo, price, rate):
    """
    """
    
    moy_Vi = df_al_pr_ra.moy_Vi.tolist()
    TOOLS[7] = HoverTool(tooltips=[
                            ("algo", "@algo"),
                            ("t", "@t"),
                            ("moy_Vi", "@moy_Vi"),
                            ("std_Vi", "@std_Vi")
                            ]
                        )
    
    title = "{}: distribution of moy_Vi over time (price={},rate={}"\
                .format(algo, price, rate)
    ts = df_al_pr_ra.t.tolist()
    px= figure(x_range=ts, 
                plot_height=int(HEIGHT*MULT_HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH),
                title=title,
                toolbar_location="above", tools=TOOLS)

    data = dict(x=ts, moy_Vi=moy_Vi, std_Vi=df_al_pr_ra.std_Vi.tolist(),
                t=df_al_pr_ra.t.tolist(),
                algo=df_al_pr_ra.algo.tolist())
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='moy_Vi', width=0.9, source=source, 
            fill_color=factor_cmap('x', 
                                    palette=Category20[20], 
                                    factors=list(df_al_pr_ra.t.unique()), 
                                    start=1, end=2))
    
    px.y_range.start = df_al_pr_ra.moy_Vi.min() - 1
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    px.xaxis.axis_label = 't_periods'
    px.yaxis.axis_label = 'moy_Vi'
    
    return px
    
def plot_distribution_over_time(df_algo_t_periods_moyVi, 
                                algos=["LRI1","LRI2"]):
    """
    distribution of moy_Vi over the time for algo LRIx
    """
    rates = df_algo_t_periods_moyVi.rate.unique(); 
    rates = rates[rates!=0].tolist()
    prices = df_algo_t_periods_moyVi.prices.unique().tolist();
    
    dico_pxs = dict()
    for (algo, price, rate) in it.product(algos, prices, rates):
        mask_al_pr_ra = (df_algo_t_periods_moyVi.rate == rate) \
                            & (df_algo_t_periods_moyVi.prices == price) \
                            & (df_algo_t_periods_moyVi.algo == algo)
        df_al_pr_ra = df_algo_t_periods_moyVi[mask_al_pr_ra]
        
        px = plot_distribution_algo(df_al_pr_ra, algo, price, rate)
        
        if (price, rate, algo) not in dico_pxs:
            dico_pxs[(price, rate, algo)] = [px]
        else:
            dico_pxs[(price, rate, algo)].append(px)
            
    col_pxs = []
    for key, pxs in dico_pxs.items():
        row_px_sts = row(pxs)
        col_pxs.append(row_px_sts)
    col_pxs=column(children=col_pxs, 
                   sizing_mode='stretch_both')
    return col_pxs   
                            
# _____________________________________________________________________________
#
#           distribution de moyenne de Vi a chaque periode  ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#           representation de moyenne de Vi par algo  ---> debut
# _____________________________________________________________________________
def plot_bar_meanVi_all_algos(df_ra_pr, price, rate):
    """
    draw a bar plot of Vi mean for all algo.
    the bar plot has algo on x-axis ie key = (algo)
        and moyVi on y-axis

    """
    cols = ["algo"]
    df_groupby_algo = df_ra_pr.groupby(cols)["moy_Vi"]\
                                .aggregate([np.sum, np.mean, np.std])\
                                .reset_index()
    df_groupby_algo.rename(columns={"mean":"moy_Vi", 
                                    "sum":"sum_Vi", 
                                    "std":"std_Vi"}, 
                                   inplace=True)
    algos = df_groupby_algo[cols].values.tolist() 
    algos = df_groupby_algo['algo'].values.tolist()
    moy_Vi = list(df_groupby_algo["moy_Vi"])
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("moy_Vi", "@moy_Vi"),
                            ("sum_Vi", "@sum_Vi"),
                            ("std_Vi", "@std_Vi"),
                            ("algo", "@algo"),
                            ]
                        )
    px= figure(x_range=algos, 
               plot_height=int(HEIGHT*MULT_HEIGHT), 
               plot_width = int(WIDTH*MULT_WIDTH),
               title="mean of Vi by algorithm",
                toolbar_location="above", tools=TOOLS)

    data = dict(x=algos, moy_Vi=moy_Vi,
                sum_Vi=df_groupby_algo.sum_Vi.tolist(),
                std_Vi=df_groupby_algo.std_Vi.tolist(),
                algo=df_groupby_algo.algo.tolist())
    
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='moy_Vi', width=0.9, source=source, 
            fill_color=factor_cmap('x', 
                                    palette=Category20[20], 
                                    factors=list(df_groupby_algo\
                                                 .algo.unique()), 
                                    start=0, end=1))
    
    px.y_range.start = df_groupby_algo.moy_Vi.min() # 0
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    px.xaxis.axis_label = 'algorithms'
    px.yaxis.axis_label = 'moy_Vi'
    
    col_px = column(px)
    col_px = column(children=[col_px], 
                    sizing_mode='stretch_both')
    return col_px

def plot_bar_meanVi_by_algo(df_algo_t_periods_moyVi):
    algos = df_algo_t_periods_moyVi.algo.unique().tolist()
    rates = df_algo_t_periods_moyVi["rate"].unique(); rates = rates[rates!=0]
    prices = df_algo_t_periods_moyVi["prices"].unique()
    
    dico_pxs = dict()
    for (rate, price) in it.product(rates, prices):
        mask_ra_pr = (df_algo_t_periods_moyVi.rate == rate) \
                            & (df_algo_t_periods_moyVi.prices == price)
        df_ra_pr = df_algo_t_periods_moyVi[mask_ra_pr]
        
        px = plot_bar_meanVi_all_algos(df_ra_pr, price, rate)
        if (price, rate) not in dico_pxs:
            dico_pxs[(price, rate)] = [px]
        else:
            dico_pxs[(price, rate)].append(px)
            
    col_pxs = []
    for key, pxs in dico_pxs.items():
        row_px_sts = row(pxs)
        col_pxs.append(row_px_sts)
    col_pxs=column(children=col_pxs, 
                   sizing_mode='stretch_both')
    return col_pxs   
# _____________________________________________________________________________
#
#           representation de moyenne de Vi par algo  ---> fin
# _____________________________________________________________________________


# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> debut
# _____________________________________________________________________________
def group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                        df_B_C_BB_CC_RU_M,
                        df_b0_c0_pisg_pi0_T_K,
                        t, k_steps_args, name_dir,
                        df_LRI_12, df_k_stop,
                        algos_4_learning,
                        path_2_best_learning_steps, 
                        NAME_RESULT_SHOW_VARS):
    
    rows_dists_ts = plot_distribution_by_states_4_periods(
                        df_arr_M_T_Ks, k_steps_args,
                        path_2_best_learning_steps)
    tab_dists_ts = Panel(child=rows_dists_ts, title="distribution by state")
    print("Distribution of players: TERMINEE")
    
    rows_RU_CONS_PROD_ts = plot_utilities_by_player_4_periods(
                            df_arr_M_T_Ks,
                            df_B_C_BB_CC_RU_M, 
                            path_2_best_learning_steps)
    tab_RU_CONS_PROD_ts = Panel(child=rows_RU_CONS_PROD_ts, 
                                title="Players utility of players")
    print("Players Utility of RU: TERMINEE")
    
    rows_RU_C_B_CC_BB = plot_utilities_4_periods(
                            df_B_C_BB_CC_RU_M, 
                            path_2_best_learning_steps)
    tab_RU_C_B_CC_BB = Panel(child=rows_RU_C_B_CC_BB, 
                                title="utility by algo")
    print("Players Utility of RU: TERMINEE")
    
    ##### to add to group_plot_on_panel with plot_bar_meanVi_over_time() #####
    df_algo_t_periods_moyVi = create_dataframe_mean_Vi_for(
                                        df_ben_cst_M_T_K, 
                                        df_LRI_12, 
                                        k_steps_args, 
                                        algos_4_learning)
    
    cols_distri_over_time = plot_distribution_over_time(
                                df_algo_t_periods_moyVi,
                                algos=["LRI1","LRI2","DETERMINIST"])
    tab_distri_over_time = Panel(child=cols_distri_over_time, 
                                  title="distribution of moy_Vi over time")
    print("distribution of moy_Vi over time: TERMINEE")

    cols_meanVi_over_time = plot_bar_meanVi_over_time(df_algo_t_periods_moyVi)
    tab_meanVi_over_time = Panel(child=cols_meanVi_over_time, 
                                  title="mean of Vi over time")
    print("mean of Vi over time moy_Vi: TERMINEE")
    
    
    cols_meanVi_by_algo = plot_bar_meanVi_by_algo(df_algo_t_periods_moyVi)
    tab_meanVi_by_algo = Panel(child=cols_meanVi_by_algo, 
                                  title="mean of Vi by algo")
    print("mean of Vi over ALGO moy_Vi: TERMINEE")
    
    
    rows_pls_CONS_PROD_ts = plot_evolution_players_PROD_CONS_over_time(
                                    df_arr_M_T_Ks, 
                                    path_2_best_learning_steps)
    tab_pls_CONS_PROD_ts = Panel(child=rows_pls_CONS_PROD_ts, 
                                title="evolution of PROD and CONS by players")
    print("Evolution of CONS and PROD by players over time: TERMINEE")
    
    
    rows_OUT_IN_SG_ts = plot_evolution_IN_OUT_SG_over_time(
                            df_arr_M_T_Ks,
                            df_LRI_12)
    tab_OUT_IN_SG_ts = Panel(child=rows_OUT_IN_SG_ts, 
                              title="evolution of In_sg and OUT_sg over time")
    print("Evolution of OUT_sg and IN_sg by time: TERMINEE")
    
    
    rows_PISG_b0c0_ts = plot_evolution_over_time_PISG_b0c0(df_arr_M_T_Ks, 
                                        df_b0_c0_pisg_pi0_T_K,
                                        df_LRI_12)
    tab_PISG_b0c0_ts = Panel(child=rows_PISG_b0c0_ts, 
                              title="evolution of pi_sg,b0,c0")
    print("Evolution of PI_SG, b0, c0: TERMINEE")
    
    
    # col_pxs_Pref_t = plot_Perf_t_players_all_states_for_scenarios(
    #                     df_ben_cst_M_T_K, t)
    t = 0
    col_pxs_Pref_algo_t = plot_Perf_t_players_all_algos(
                            df_ben_cst_M_T_K, t, 
                            df_LRI_12, df_k_stop)
    tab_Pref_algo_t=Panel(child=col_pxs_Pref_algo_t, title="Pref_t")
    print("Performance Perf_t all algos: TERMINEE")
    
    
    col_px_scen_st_S1S2s = plot_max_proba_mode(df_arr_M_T_Ks, t, 
                                                path_2_best_learning_steps, 
                                                algos=['LRI1','LRI2'])
    tab_S1S2=Panel(child=col_px_scen_st_S1S2s, title="mean_S1_S2")
    print("affichage S1, S2 p_ijk : TERMINEE")
    
    
    # col_pxs_in_out = plot_in_out_sg_ksteps_for_scenarios(df_arr_M_T_Ks, t)
    # col_pxs_ben_cst = plot_mean_ben_cst_players_all_states_for_scenarios(
    #                     df_ben_cst_M_T_K, t)
    # col_px_scen_sts = histo_states(df_arr_M_T_Ks, t)
    # col_px_scen_mode_S1S2_nbplayers = plot_histo_strategies(df_arr_M_T_Ks, t)
    # col_playing_players = plot_histo_playing(df_arr_M_T_Ks, t)
    
    # # # tab_Pref_t=Panel(child=col_pxs_Pref_t, title="Pref_t by state")
    # # tab_inout=Panel(child=col_pxs_in_out, title="In_sg-Out_sg")
    # # tab_bencst=Panel(child=col_pxs_ben_cst, title="mean(ben-cst)")
    # # tab_sts=Panel(child=col_px_scen_sts, title="number players")
    # # tab_mode_S1S2_nbplayers=Panel(child=col_px_scen_mode_S1S2_nbplayers, 
    # #                               title="number players by strategies")
    # # tab_playing=Panel(child=col_playing_players, 
    # #              title="number players playing/not_playing")
    
    tabs = Tabs(tabs= [ 
                        tab_dists_ts,
                        tab_RU_CONS_PROD_ts,
                        tab_RU_C_B_CC_BB,
                        tab_distri_over_time,
                        tab_meanVi_over_time,
                        tab_meanVi_by_algo,
                        tab_pls_CONS_PROD_ts,
                        tab_OUT_IN_SG_ts,
                        tab_Pref_algo_t,
                        tab_S1S2,
                        tab_PISG_b0c0_ts, 
                        
                        #tab_Pref_t, 
                        #tab_Pref_algo_t,
                        #tab_S1S2,
                        #tab_inout, 
                        #tab_bencst,
                        #tab_sts, 
                        #tab_mode_S1S2_nbplayers, 
                        #tab_playing
                        ])
    
    output_file( os.path.join(name_dir, NAME_RESULT_SHOW_VARS)  )
    save(tabs)
    show(tabs)
    
    return df_algo_t_periods_moyVi
    
# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> fin
# _____________________________________________________________________________

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    MULT_WIDTH = 2.5;
    MULT_HEIGHT = 1.2;
    # fct_aux.N_DECIMALS = 7;
    
    pi_hp_plus = 0.2*pow(10,-3); pi_hp_minus = 0.33
    NAME_RESULT_SHOW_VARS = NAME_RESULT_SHOW_VARS.format(pi_hp_plus, pi_hp_minus)
    
    debug = True #False#True
    
    t = 1
    
    name_simu =  "simu_2701_1259"; k_steps_args = 250
    name_simu =  "simu_DDMM_HHMM_scenario1_T50"; k_steps_args = 50#2000#250
    
    #name_simu = "simu_DDMM_HHMM_scenario1_T20"
    #name_simu = "simu_DDMM_HHMM_scenario2_T20"
    name_simu = "simu_DDMM_HHMM_scenario2_T20"; k_steps_args = 250 #350 #2000#250
    name_simu = "simu_DDMM_HHMM_scenario3_T3gammaV1"; k_steps_args = 5
    
    name_simu = "simu_DDMM_HHMM_scenario2_T10gammaV4"; k_steps_args = 250; 
    name_simu = "simu_DDMM_HHMM_scenario3_T50gammaV4"; k_steps_args = 250;
    
    name_simu = "simu_DDMM_HHMM_scenario2_T20gammaV4"; k_steps_args = 250;
    name_simu = os.path.join("gamma_V0_V1_V2_V3_V4_T20_kstep250_setACsetAB1B2C", name_simu)
    
    name_simu = "simu_DDMM_HHMM_scenario2_T5gammaV4"; k_steps_args = 250;
    name_simu = os.path.join("gamma_V0_V1_V2_V3_V4_T5_kstep75_setACsetAB1B2C", name_simu)
    
    algos_4_no_learning = ["DETERMINIST","RD-DETERMINIST"] \
                            + fct_aux.ALGO_NAMES_BF \
                            + fct_aux.ALGO_NAMES_NASH
    algos_4_learning = ["LRI1", "LRI2"]
    # algos_4_showing = ["DETERMINIST", "LRI1", "LRI2"] \
    #                     + [fct_aux.ALGO_NAMES_BF[0], fct_aux.ALGO_NAMES_BF[1]]
    algos_4_showing = ["DETERMINIST", "LRI1", "LRI2"] 
                        
    tuple_paths, prices, algos, learning_rates, path_2_best_learning_steps \
        = get_tuple_paths_of_arrays(
            name_simu=name_simu, 
            algos_4_no_learning=algos_4_no_learning, 
            algos_4_showing = algos_4_showing
            )
    #print("tuple_paths:{}".format(tuple_paths))
    #print("path_2_best_learning_steps:{}".format(path_2_best_learning_steps))
    print("get_tuple_paths_of_arrays: TERMINE")    
        
    
    dico_k_stop = dict()
    path_2_best_learning_steps = list(set(path_2_best_learning_steps))
    df_LRI_12, df_k_stop = get_k_stop_4_periods(path_2_best_learning_steps)
    print("get_k_stop_4_periods: TERMINE") 
    
    tuple_paths = list(set(tuple_paths))
    df_arr_M_T_Ks, \
    df_ben_cst_M_T_K, \
    df_b0_c0_pisg_pi0_T_K, \
    df_B_C_BB_CC_RU_M \
        = get_array_turn_df_for_t(tuple_paths, t=None, k_steps_args=k_steps_args, 
                                  algos_4_no_learning=algos_4_no_learning, 
                                  algos_4_learning=algos_4_learning)
    print("size t={}, df_arr_M_T_Ks={} Mo, df_ben_cst_M_T_K={} Mo, df_b0_c0_pisg_pi0_T_K={} Mo, df_B_C_BB_CC_RU_M={} Mo".format(
                t, 
              round(df_arr_M_T_Ks.memory_usage().sum()/(1024*1024), 2),  
              round(df_ben_cst_M_T_K.memory_usage().sum()/(1024*1024), 2),
              round(df_b0_c0_pisg_pi0_T_K.memory_usage().sum()/(1024*1024), 2),
              round(df_B_C_BB_CC_RU_M.memory_usage().sum()/(1024*1024), 4)
              ))
    print("get_array_turn_df_for_t: TERMINE")
    
    ## -- plot figures
    name_dir = os.path.join("tests", name_simu)
    df_algo_t_periods_moyVi = group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                        df_B_C_BB_CC_RU_M,
                        df_b0_c0_pisg_pi0_T_K,
                        t, k_steps_args, name_dir, 
                        df_LRI_12, df_k_stop,
                        algos_4_learning,
                        path_2_best_learning_steps, 
                        NAME_RESULT_SHOW_VARS)
    
    
    # ##### to add to group_plot_on_panel with plot_bar_meanVi_over_time() #####
    # df_algo_t_periods_moyVi = create_dataframe_mean_Vi_for(
    #                                     df_ben_cst_M_T_K, 
    #                                     df_LRI_12, 
    #                                     k_steps_args, 
    #                                     algos_4_learning)
    
    # cols_meanVi_over_time = plot_bar_meanVi_over_time(df_algo_t_periods_moyVi)
    # tab_meanVi_over_time = Panel(child=cols_meanVi_over_time, 
    #                               title="mean of Vi over time")
    
    # cols_meanVi_by_algo = plot_bar_meanVi_by_algo(df_algo_t_periods_moyVi)
    # tab_meanVi_by_algo = Panel(child=cols_meanVi_by_algo, 
    #                               title="mean of Vi by algo")
       
    # tabs = Tabs(tabs= [ 
    #                         # tab_meanVi_over_time,
    #                         tab_meanVi_by_algo
    #                         ])
        
    # show(tabs)
   
    




