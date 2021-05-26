# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 11:37:24 2021

@author: jwehounou
"""
import os
import time
import numpy as np
import pandas as pd
import itertools as it
import fonctions_auxiliaires as fct_aux
import visu_bkh_automate_v1 as autoVizGameV1

from pathlib import Path

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
MULT_WIDTH = 3.5;
MULT_HEIGHT = 1.5;

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

#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------

# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> debut
# _____________________________________________________________________________

def get_tuple_paths_of_arrays_4_many_simu(name_dir="tests", 
                                          n_instances=50,
                                          t_periods=1,
                prices=None, algos=None, learning_rates=None, 
                algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                     "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE", 
                                       "MIDDLE-BRUTE-FORCE"], 
                algos_4_showing=["DETERMINIST", "LRI1", "LRI2",
                                 "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE"],
                ext=".npy", 
                exclude_html_files=[autoVizGameV1.NAME_RESULT_SHOW_VARS,"html"]):
    
    tuple_paths = []
    path_2_best_learning_steps = []
    prices_new, algos_new, learning_rates_new = [], [], []
    rep_simu_dir = os.path.join(name_dir)
    name_simus_5 = [simu_ for simu_ in os.listdir(rep_simu_dir)
                    if len(simu_.split("_")) == 6]
    name_simus = [simu_ for simu_ in name_simus_5
                  if int(simu_.split("_")[3]) < n_instances 
                      and simu_.split("_")[5] == str(t_periods)]
    print("name_simus_5={}, name_simus={}".format(len(name_simus_5), 
                                                  len(name_simus)))
    cpt_NASH, cpt_BF_MID = 0, 0
    for name_simu in name_simus:
        num_instance = name_simu.split("_")[3]
        tuple_paths_sim, prices_new_sim, algos_new_sim, \
        learning_rates_new_sim, \
        path_2_best_learning_steps_sim \
            = autoVizGameV1.get_tuple_paths_of_arrays(
                name_dir=name_dir, name_simu=name_simu,
                prices=prices, algos=algos, learning_rates=learning_rates, 
                algos_4_no_learning=algos_4_no_learning, 
                algos_4_showing=algos_4_showing,
                ext=ext, 
                exclude_html_files=exclude_html_files
                )
            
        for tuple_path in tuple_paths_sim:
            if tuple_path[3] in fct_aux.ALGO_NAMES_NASH:
                cpt_NASH += 1
            if tuple_path[3] in [fct_aux.ALGO_NAMES_BF[2]]:
                cpt_BF_MID += 1
        tuple_paths.extend(tuple_paths_sim)
        path_2_best_learning_steps.extend(path_2_best_learning_steps_sim)
        prices_new.extend(prices_new_sim)
        algos_new.extend(algos_new_sim) 
        learning_rates_new.extend(learning_rates_new_sim)
        
        
    print('cpt_NASH={}, cpt_BF_MID={}'.format(cpt_NASH, cpt_BF_MID))
    return tuple_paths, prices_new, algos_new, learning_rates_new, \
            path_2_best_learning_steps
       
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
        #     = autoVizGameV1.get_local_storage_variables(path_to_variable)
            
        arr_pl_M_T_K_vars, \
        b0_s_T_K, c0_s_T_K, \
        B_is_M, C_is_M, B_is_M_T, C_is_M_T,\
        BENs_M_T_K, CSTs_M_T_K, \
        BB_is_M, CC_is_M, RU_is_M, BB_is_M_T, CC_is_M_T, RU_is_M_T, \
        pi_sg_plus_T, pi_sg_minus_T, \
        pi_0_plus_T, pi_0_minus_T, \
        pi_hp_plus_T, pi_hp_minus_T \
            = fct_aux.get_local_storage_variables(path_to_variable)
        
        instance = tuple_path[1].split("_")[3] \
                    if len(tuple_path[1].split("_")) >= 3 else str(0)
        price = tuple_path[2].split("_")[3]+"_"+tuple_path[2].split("_")[-1]
        algo = tuple_path[3];
        rate = tuple_path[4] if algo in algos_4_learning else 0
        
        m_players = arr_pl_M_T_K_vars.shape[0]
        k_steps = arr_pl_M_T_K_vars.shape[2] if arr_pl_M_T_K_vars.shape == 4 \
                                             else k_steps_args                                    
        #for t in range(0, t_periods):                                     
        t_periods = None; tu_mtk = None; tu_tk = None; tu_m = None
        if t is None:
            t_periods = arr_pl_M_T_K_vars.shape[1] - 1
            tu_mtk = list(it.product([algo], [rate], [price], [instance],
                                     range(0, m_players), 
                                     range(0, t_periods), 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [instance],
                                    range(0, t_periods), 
                                    range(0, k_steps)))
            t_periods = list(range(0, t_periods))
        elif type(t) is list:
            t_periods = t
            tu_mtk = list(it.product([algo], [rate], [price], [instance],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [instance],
                                    t_periods, 
                                    range(0, k_steps)))
        elif type(t) is int:
            t_periods = [t]
            tu_mtk = list(it.product([algo], [rate], [price], [instance],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [instance],
                                    t_periods, 
                                    range(0, k_steps)))
                      
        print('t_periods = {}'.format(t_periods))
        tu_m = list(it.product([algo], [rate], [price], [instance], 
                               range(0, m_players)))
                    
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
    columns_ind = ["algo","rate","prices","num_instance","pl_i","t","k"]
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
    columns_ind = ["algo","rate","prices","num_instance","pl_i","t","k"]
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
    columns_ind = ["algo","rate","prices","num_instance","t","k"]
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
    columns_ind = ["algo","rate","prices","num_instance","pl_i"]
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
def plot_bar_Perf_t_all_algos(df_ra_pri, rate, price, t):
                                
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
    nb_instances = df_ra_pri.num_instance.unique().tolist()
    num_pl_is = df_ra_pri.pl_i.unique().tolist()
    tup_legends = [] 
    
    dico_algo_instance = dict()
    dico_algo_instance["pl_i"] = []
    dico_algo_instance["k"] = []
    dico_algo_instance["ben"] = []
    dico_algo_instance["cst"] = []
    dico_algo_instance["num_instance"] = []
    dico_algo_instance["algo"] = []
    for algo in algos:
        
        mask_algo = (df_ra_pri.algo == algo)
        df_ra_pr_al = df_ra_pri[mask_algo]
        # delete rows with nan values of columns = [ben, cst]
        df_ra_pr_al = df_ra_pr_al[df_ra_pr_al['ben'].notna() 
                                  & df_ra_pr_al['cst'].notna() ]
        # for each instance, select the k_stop and put it to a new dataframe.
        num_instances = df_ra_pr_al.num_instance.unique().tolist()
        for num_instance in num_instances:
            for num_pl_i in num_pl_is:
                mask_inst_pl_i = (df_ra_pr_al.pl_i==num_pl_i) \
                                    & (df_ra_pr_al.num_instance==num_instance)
                k_max = df_ra_pr_al[mask_inst_pl_i].k.max()
                mask_inst_pl_i_k_max = (df_ra_pr_al.pl_i==num_pl_i) \
                                    & (df_ra_pr_al.num_instance==num_instance) \
                                    & (df_ra_pr_al.k==k_max)
                index = df_ra_pr_al[ mask_inst_pl_i_k_max ].index[0]
                ben = df_ra_pr_al[mask_inst_pl_i].loc[index, 'ben']
                cst = df_ra_pr_al[mask_inst_pl_i].loc[index, 'cst']
                dico_algo_instance["pl_i"].append(num_pl_i)
                dico_algo_instance["k"].append(k_max)
                dico_algo_instance["ben"].append(ben)
                dico_algo_instance["cst"].append(cst)
                dico_algo_instance["num_instance"].append(num_instance)
                dico_algo_instance["algo"].append(algo)
    
    df = pd.DataFrame.from_dict(dico_algo_instance)
    
    df["Vi"] = df["ben"] - df["cst"]
    
    df_res = df.groupby(['algo'])['Vi'].aggregate([np.sum, np.mean, np.std])\
                .reset_index()
    
    TOOLS[7] = HoverTool(tooltips=[
                    ("algo", "@algo"),
                    ("mean", "@mean"),
                    ("sum", "@sum"), 
                    ("std", "@std")
                    ]
                )
    idx = df_res.algo.unique().tolist()
    cols = ["algo","mean","std","sum"]
    px = figure(x_range=idx, 
                y_range=(df_res[cols[1]].values.min(), 
                         df_res[cols[1]].values.max() ), 
                plot_height = int(HEIGHT*MULT_HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
    title = "Average of Vi over 50 executions (rate:{}, price={})".format(
                rate, price)
    px.title.text = title
    
    source = ColumnDataSource(data = df_res)
    
    width= 0.1 #0.5
    px.vbar(x=dodge('algo', -0.0, range=px.x_range), top=cols[1], 
                    width=width, source=source,
                    color="#2E8B57", legend_label=cols[1])
    # px.vbar(x=dodge('algo', -0.3+1*width, range=px.x_range), top=cols[2], 
    #                 width=width, source=source,
    #                 color="#718dbf", legend_label=cols[2])
    # px.vbar(x=dodge('t', -0.3+2*width, range=px.x_range), top=cols[3], 
    #                width=width, source=source,
    #                color="#e84d60", legend_label=cols[3])
    
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "t_periods"
    px.yaxis.axis_label = "values"
    
    return px

def plot_Perf_t_players_all_algos(df_ben_cst_M_T_K, t):
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
                            & (df_ben_cst_M_T_K.t == t) 
        df_ra_pri = df_ben_cst_M_T_K[mask_ra_pri_st].copy()
        
        px_st = plot_bar_Perf_t_all_algos(df_ra_pri, rate, price, t)
        
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


# _____________________________________________________________________________
#
#            create dataframe with only k_max of each player  ---> debut
# _____________________________________________________________________________
def create_dataframe_kmax(df_ben_cst_M_T_K):
    """
    create a dataframe with only k_max of each player
    """
    algos = df_ben_cst_M_T_K["algo"].unique().tolist()
    
    dico_algo_instance = dict()
    dico_algo_instance["pl_i"] = []
    dico_algo_instance["k_max"] = []
    dico_algo_instance["ben"] = []
    dico_algo_instance["cst"] = []
    dico_algo_instance["num_instance"] = []
    dico_algo_instance["algo"] = []
    dico_algo_instance["state_i"] = []
    for algo in algos:
        
        mask_algo = (df_ben_cst_M_T_K.algo == algo)
        df_al = df_ben_cst_M_T_K[mask_algo]
        # delete rows with nan values of columns = [ben, cst]
        df_al = df_al[df_al['ben'].notna() & df_al['cst'].notna()]
        # for each instance, select the k_stop and put it to a new dataframe.
        num_instances = df_al.num_instance.unique().tolist()
        num_pl_is = df_al.pl_i.unique().tolist()
        for num_instance in num_instances:
            for num_pl_i in num_pl_is:
                mask_inst_pl_i = (df_al.pl_i == num_pl_i) \
                                    & (df_al.num_instance == num_instance)
                k_max = df_al[mask_inst_pl_i].k.max()
                mask_inst_pl_i_k_max = (df_al.pl_i==num_pl_i) \
                                    & (df_al.num_instance==num_instance) \
                                    & (df_al.k==k_max)
                index = df_al[ mask_inst_pl_i_k_max ].index[0]
                ben = df_al[mask_inst_pl_i].loc[index, 'ben']
                cst = df_al[mask_inst_pl_i].loc[index, 'cst']
                state_i = df_al[mask_inst_pl_i].loc[index, 'state_i']
                dico_algo_instance["pl_i"].append(num_pl_i)
                dico_algo_instance["k_max"].append(k_max)
                dico_algo_instance["ben"].append(ben)
                dico_algo_instance["cst"].append(cst)
                dico_algo_instance["num_instance"].append(num_instance)
                dico_algo_instance["algo"].append(algo)
                dico_algo_instance["state_i"].append(state_i)
        print("k_max: {} TERMINE".format(algo))
    
    df_ben_cst_M_T_Kmax = pd.DataFrame.from_dict(dico_algo_instance)
    
    df_ben_cst_M_T_Kmax["Vi"] = df_ben_cst_M_T_Kmax["ben"] \
                                - df_ben_cst_M_T_Kmax["cst"]
    
    return df_ben_cst_M_T_Kmax
    
    
# _____________________________________________________________________________
#
#            create dataframe with only k_max of each player  ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#               create dataframe belonging mean of Vi  ---> debut
# _____________________________________________________________________________
def create_dataframe_mean_Vi_for(df_ben_cst_M_T_Kmax):
    """
    create a dataframe containing the mean of Vi by for loop
    for each algo, each state and each instance
    
    created dataframe is named df_algo_instance_state_moyVi
    columns of df_algo_inst_state_moyVi are algo, instance, state, moy_Vi
    """    
    
    algos = df_ben_cst_M_T_Kmax.algo.unique().tolist()
    dico_algo_instance_state = dict()
    dico_algo_instance_state["algo"] = []
    dico_algo_instance_state["num_instance"] = []
    dico_algo_instance_state["state_i"] = []
    dico_algo_instance_state["moy_Vi"] = []
    dico_algo_instance_state["nb_players"] = []
    dico_algo_instance_state["id_players"] = []
    
    for algo in algos:
        mask_algo = (df_ben_cst_M_T_Kmax.algo == algo)
        num_instances = df_ben_cst_M_T_Kmax[mask_algo]\
                            .num_instance.unique().tolist()
        for num_instance in num_instances:
            mask_algo_instance = (df_ben_cst_M_T_Kmax.algo == algo) \
                                & (df_ben_cst_M_T_Kmax.num_instance\
                                       == num_instance)
            state_is = df_ben_cst_M_T_Kmax[mask_algo_instance]\
                        .state_i.unique().tolist()
            for state_i in state_is:
                mask_algo_instance_state \
                    = (df_ben_cst_M_T_Kmax.algo == algo) \
                        & (df_ben_cst_M_T_Kmax.num_instance == num_instance) \
                        & (df_ben_cst_M_T_Kmax.state_i == state_i)
                df_pl_is = df_ben_cst_M_T_Kmax[mask_algo_instance_state]
                nb_players = df_pl_is.count().values[0]
                id_players = df_pl_is.pl_i.tolist()
                mean_state_i = df_pl_is['Vi'].mean()
                dico_algo_instance_state["algo"].append(algo)
                dico_algo_instance_state["num_instance"].append(num_instance)
                dico_algo_instance_state["state_i"].append(state_i)
                dico_algo_instance_state["moy_Vi"].append(mean_state_i)
                dico_algo_instance_state["nb_players"].append(nb_players)
                dico_algo_instance_state["id_players"].append(id_players)
        print("mean_Vi_for: {} TERMINE".format(algo))
        
    df_algo_instance_state_moyVi = pd.DataFrame.from_dict(
                                    dico_algo_instance_state)
    
    return df_algo_instance_state_moyVi
    

def create_dataframe_mean_Vi_groupby(df_ben_cst_M_T_Kmax):
    """
    create a dataframe containing the mean of Vi by for loop
    for each algo, each state and each instance
    
    created dataframe is named df_algo_inst_state_moyVi
    columns of df_algo_inst_state_moyVi are algo, instance, state, moy_Vi
    """    
    pass
# _____________________________________________________________________________
#
#               create dataframe belonging mean of Vi  ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#        plot bar of mean of Vi (moy_Vi) for each algo, instance, state   ---> debut
# _____________________________________________________________________________
def plot_bar_moyVi_4_one_algo_many_state_instance(df_algo, algo):
    """
    bar plot Vi mean for one algo knowing states and instances
    plot is the bar plot with key is (instance, stateX) (X={1,2,3})
    
    on x-axis, there are instance and state
    on y-axis, there is Vi mean (moyVi)
    """
    # order df_algo by num_instance
    
    
    cols = ["num_instance", "state_i"]
    x = list(map(tuple,list(df_algo[cols].values)))
    moy_Vi = list(df_algo["moy_Vi"])
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("nb_players", "@nb_players"),
                            ("id_players", "@id_players"),
                            ("state_i", "@state_i"),
                            ("moy_Vi", "@moy_Vi"),
                            ("instance", "@num_instance"),
                            ("algo", "@algo"),
                            ]
                        )
    print("HEIGHT={},MULT_HEIGHT{},WIDTH={},MULT_WIDTH={}".format(
            HEIGHT, MULT_HEIGHT, WIDTH, MULT_WIDTH))
    px= figure(x_range=FactorRange(*x), 
                plot_height=int(HEIGHT*MULT_HEIGHT), 
                plot_width = int(WIDTH*MULT_WIDTH),
                title="mean of Vi by instances for {}".format(algo),
                toolbar_location="above", tools=TOOLS)
    # px= figure(x_range=FactorRange(*x), 
    #            plot_height=int(500), 
    #            plot_width = int(1500),
    #            title="mean of Vi by instances for {}".format(algo),
    #            toolbar_location="above", tools=TOOLS)

    data = dict(x=x, moy_Vi=moy_Vi, nb_players=df_algo.nb_players.tolist(),
                id_players=df_algo.id_players.tolist(),
                state_i=df_algo.state_i.tolist(),
                num_instance=df_algo.num_instance.tolist(),
                algo=df_algo.algo.tolist())
    
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='moy_Vi', width=0.9, source=source, 
            fill_color=factor_cmap('x', 
                                    palette=Category20[20], 
                                    factors=list(df_algo.state_i.unique()), 
                                    start=1, end=2))
    
    
    px.y_range.start = df_algo.moy_Vi.min() # 0
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    px.xaxis.axis_label = 'instances'
    px.yaxis.axis_label = 'moy_Vi'
    
    return px
    
def plot_bar_moyVi_4_algo_state_instance(df_algo_instance_state_moyVi):
    """
    for each row of plot, draw a bar plot of Vi mean for each algo.
    the bar plot has instance and state on x-axis ie key = (instance, stateX) (X={1,2,3})
        and moyVi on y-axis

    Parameters
    ----------
    df_algo_instance_state_moyVi : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    dico_pxs = dict()
    
    algos = df_algo_instance_state_moyVi.algo.unique().tolist()
    for algo in algos:
        mask_algo = (df_algo_instance_state_moyVi.algo == algo)    
        df_algo = df_algo_instance_state_moyVi[mask_algo].copy()
        df_algo["num_instance"] = df_algo["num_instance"].astype(int)
        df_algo = df_algo.sort_values(['num_instance', 'state_i'], 
                                      ascending=[True, True])
        df_algo["num_instance"] = df_algo["num_instance"].astype(str)
        
        pxs_algo = plot_bar_moyVi_4_one_algo_many_state_instance(df_algo, algo)
        
        if algo not in dico_pxs.keys():
            dico_pxs[algo] \
                = [pxs_algo]
        else:
            dico_pxs[algo].append(pxs_algo)
        
    col_algos = list()
    for key, pxs_al in dico_pxs.items():
        col_px_algo = column(pxs_al)
        col_algos.append(col_px_algo)
    col_algos = column(children=col_algos, 
                       sizing_mode='stretch_both')
    return col_algos
    
# _____________________________________________________________________________
#
#        plot bar of mean of Vi (moy_Vi) for each state and algo   ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#        plot bar of mean of Vi (moy_Vi) for each state and algo   ---> debut
# _____________________________________________________________________________
def plot_bar_moyVi_4_algo_state(df_algo_instance_state_moyVi):
    """
    for each row of plot, draw a bar plot of Vi mean for all algos.
    the bar plot has algo and state on x-axis ie key = (algo, stateX) (X={1,2,3})
        and moyVi on y-axis

    Parameters
    ----------
    df_algo_instance_state_moyVi : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    dico_pxs = dict()
    cols = ["algo", "state_i"]
    df_groupby_algo_state_i = df_algo_instance_state_moyVi\
                                .groupby(cols)["moy_Vi"]\
                                .aggregate([np.sum, np.mean, np.std])\
                                .reset_index()
    df_groupby_algo_state_i.rename(columns={"mean":"moy_Vi", 
                                            "sum":"sum_Vi", 
                                            "std":"std_Vi"}, 
                                   inplace=True)
    x = list(map(tuple,list(df_groupby_algo_state_i[cols].values)))
    moy_Vi = list(df_groupby_algo_state_i["moy_Vi"])
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("state_i", "@state_i"),
                            ("moy_Vi", "@moy_Vi"),
                            ("sum_Vi", "@sum_Vi"),
                            ("std_Vi", "@std_Vi"),
                            ("algo", "@algo"),
                            ]
                        )
    px= figure(x_range=FactorRange(*x), 
               plot_height=int(HEIGHT*MULT_HEIGHT), 
               plot_width = int(WIDTH*MULT_WIDTH),
               title="mean of Vi by algorithm and states",
                toolbar_location="above", tools=TOOLS)

    data = dict(x=x, moy_Vi=moy_Vi,
                sum_Vi=df_groupby_algo_state_i.sum_Vi.tolist(),
                std_Vi=df_groupby_algo_state_i.std_Vi.tolist(),
                state_i=df_groupby_algo_state_i.state_i.tolist(),
                algo=df_groupby_algo_state_i.algo.tolist())
    
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='moy_Vi', width=0.9, source=source, 
            fill_color=factor_cmap('x', 
                                    palette=Category20[20], 
                                    factors=list(df_groupby_algo_state_i\
                                                 .state_i.unique()), 
                                    start=1, end=2))
    
    px.y_range.start = df_groupby_algo_state_i.moy_Vi.min() # 0
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    px.xaxis.axis_label = 'algorithms'
    px.yaxis.axis_label = 'moy_Vi'
    
    col_px = column(px)
    col_px = column(children=[col_px], 
                    sizing_mode='stretch_both')
    return col_px
    
# _____________________________________________________________________________
#
#        plot bar of mean of Vi (moy_Vi) for each state and algo   ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> debut
# _____________________________________________________________________________
def group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                        df_B_C_BB_CC_RU_M,
                        df_b0_c0_pisg_pi0_T_K,
                        df_algo_instance_state_moyVi,
                        t, k_steps_args, path_to_save,
                        path_2_best_learning_steps, 
                        NAME_RESULT_SHOW_VARS):
    
    cols_mean_Perf_ts = plot_Perf_t_players_all_algos(df_ben_cst_M_T_K, t)
    tab_mean_ts = Panel(child=cols_mean_Perf_ts, title="Average of Perf_t")
    print("Average of Perf_t: TERMINEE")
    
    global MULT_WIDTH
    global MULT_HEIGHT
    
    MULT_WIDTH = 3.5;
    MULT_HEIGHT = 1.0;
    cols_meanVi_algo_instance_state = plot_bar_moyVi_4_algo_state_instance(
                                        df_algo_instance_state_moyVi)
    tab_meanVi_algo_instance_state \
        = Panel(child = cols_meanVi_algo_instance_state, 
                title="Vi Average for each algo")
    print("Vi Average plot for each algo: TERMINEE")
    
    cols_meanVi_algo_state = plot_bar_moyVi_4_algo_state(
                                        df_algo_instance_state_moyVi)
    tab_meanVi_algo_state \
        = Panel(child = cols_meanVi_algo_state, 
                title="Vi Average for all algos")
    print("Vi Average plot for all algos: TERMINEE")
    
    
    # rows_RU_CONS_PROD_ts = plot_utilities_by_player_4_periods(
    #                         df_arr_M_T_Ks,
    #                         df_B_C_BB_CC_RU_M, 
    #                         path_2_best_learning_steps)
    # print("Utility of RU: TERMINEE")
    
    # rows_pls_CONS_PROD_ts = plot_evolution_players_PROD_CONS_over_time(
    #                                 df_arr_M_T_Ks, 
    #                                 path_2_best_learning_steps)
    # print("Evolution of CONS and PROD by players over time: TERMINEE")
    
        
    tabs = Tabs(tabs= [ 
                        tab_meanVi_algo_instance_state,
                        tab_meanVi_algo_state,   
                        tab_mean_ts,
                        # tab_RU_CONS_PROD_ts,
                        # tab_pls_CONS_PROD_ts,
                        ])
    
    output_file( os.path.join(path_to_save, NAME_RESULT_SHOW_VARS)  )
    save(tabs)
    show(tabs)
    
# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> fin
# _____________________________________________________________________________
#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    
    algos_4_showing=["DETERMINIST", "LRI1", "LRI2"] \
                    + fct_aux.ALGO_NAMES_BF \
                    + fct_aux.ALGO_NAMES_NASH
    algos_4_showing=["DETERMINIST", "LRI1", "LRI2"] \
                    + [fct_aux.ALGO_NAMES_BF[0]] \
                    + [fct_aux.ALGO_NAMES_NASH[0]]
    algos_4_no_learning=["DETERMINIST","RD-DETERMINIST"] \
                         + fct_aux.ALGO_NAMES_BF \
                         + fct_aux.ALGO_NAMES_NASH
                         
    name_dir = "tests"
    name_dir_oneperiod = os.path.join(name_dir,"OnePeriod_50instances")
    name_dir_oneperiod = os.path.join(name_dir,"OnePeriod_50instancesGammaV4")
    
    phi_name = "A1B1"#"A1.2B0.8"#"A1B1"  # "A1.2B0.8", "A1B1"
    name_dir = "tests"; sub1_name_dir = "OnePeriod_50instances"; gamma_version = "gammaV-1"
    name_dir_oneperiod = os.path.join(
                            name_dir,
                            phi_name+sub1_name_dir, 
                            sub1_name_dir+gamma_version)
    
    tuple_paths, \
    prices_new, \
    algos_new, \
    learning_rates_new, \
    path_2_best_learning_steps = get_tuple_paths_of_arrays_4_many_simu(
                                    name_dir=name_dir_oneperiod,
                                    algos_4_showing=algos_4_showing,
                                    algos_4_no_learning=algos_4_no_learning)
    print("tuple_paths = {}".format(len(tuple_paths)))

    t = 0
    k_steps_args = 250
    algos_4_learning = ["LRI1", "LRI2"]
    df_arr_M_T_Ks, \
    df_ben_cst_M_T_K, \
    df_b0_c0_pisg_pi0_T_K, \
    df_B_C_BB_CC_RU_M \
        = get_array_turn_df_for_t(tuple_paths, t=t, k_steps_args=k_steps_args, 
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
    
    ti_ = time.time()
    df_ben_cst_M_T_Kmax = create_dataframe_kmax(df_ben_cst_M_T_K)
    print("create_dataframe_k_max, df_ben_cst_M_T_kmax:{}, runtime={} TERMINE"\
          .format(df_ben_cst_M_T_Kmax.shape, time.time()-ti_ ))
    
    ti_ = time.time()
    df_algo_instance_state_moyVi = create_dataframe_mean_Vi_for(
                                        df_ben_cst_M_T_Kmax)
    print("create_dataframe_mean_Vi: df_algo_instance_state_moyVi={}, runtime={},TERMINE"\
          .format(df_algo_instance_state_moyVi.shape, time.time()-ti_))
    
    path_to_save = os.path.join("tests", "AVERAGE_RESULTS")
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    group_plot_on_panel(df_arr_M_T_Ks, df_ben_cst_M_T_K, 
                        df_B_C_BB_CC_RU_M,
                        df_b0_c0_pisg_pi0_T_K,
                        df_algo_instance_state_moyVi,
                        t, k_steps_args, path_to_save,
                        path_2_best_learning_steps, 
                        NAME_RESULT_SHOW_VARS)
    
    print("runtime = {}".format(time.time() - ti))