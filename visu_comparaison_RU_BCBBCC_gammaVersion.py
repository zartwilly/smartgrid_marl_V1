# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 21:10:15 2021

@author: jwehounou

visu_dbg
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

name_dirs = ["tests"]
exclude_dirs_files = ["html", "AVERAGE_RESULTS", "AUTOMATE_INSTANCES_GAMES",
                      "gamma", "npy", "csv"]

algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                    "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE", 
                      "MIDDLE-BRUTE-FORCE"]
algos_4_showing=["DETERMINIST", "LRI1", "LRI2",
                 "BEST-BRUTE-FORCE", "BAD-BRUTE-FORCE"]

#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------


# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> debut
# _____________________________________________________________________________

def get_tuple_paths_of_arrays(name_dirs=["tests"], nb_sub_dir=1,
                algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                     "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE", 
                                       "MIDDLE-BRUTE-FORCE"], 
                algos_4_showing=["DETERMINIST", "LRI1", "LRI2",
                                 "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE"],
                exclude_dirs_files = ["html", "AVERAGE_RESULTS", 
                                      "AUTOMATE_INSTANCES_GAMES",
                                      "gamma", "npy", "csv", 
                                      NAME_RESULT_SHOW_VARS]):
    """
    autre version plus rapide 
    https://stackoverflow.com/a/59803793/2441026
    def run_fast_scandir(dir, exclude_dirs_files):    # dir: str, ext: list
        # https://stackoverflow.com/a/59803793/2441026
    
        subfolders, files = [], []
    
        for f in os.scandir(dir):
            if f.is_dir() \
                and f.name.split("_")[0] not in exclude_dirs_files \
                and f.name not in exclude_dirs_files:
                subfolders.append(f.path)
            # if f.is_file():
            #     if os.path.splitext(f.name)[1].lower() not in exclude_dirs_files:
            #         files.append(f.path)
            if f.is_file():
                if f.name.split(".")[-1] not in exclude_dirs_files:
                    files.append(f.path)
    
    
        for dir in list(subfolders):
            sf, f = run_fast_scandir(dir, exclude_dirs_files)
            subfolders.extend(sf)
            files.extend(f)
        return subfolders, files
    """
    lis_old = list()
    for name_dir in name_dirs:
        dirs = [rep for rep in os.listdir(name_dir) \
                if rep not in exclude_dirs_files \
                    and rep.split('_')[0] not in exclude_dirs_files]
        for dir_ in dirs:
            lis_old.extend([os.path.join(dp)
                        for dp, dn, fn in os.walk(os.path.join(name_dir,dir_)) 
                            for f in fn
                                ])
                        
    tuple_paths = list(); path_2_best_learning_steps = list()
    for dp in lis_old:
        if len(dp.split(os.sep)) > nb_sub_dir+1:
            tuple_paths.append( tuple(dp.split(os.sep)) )
            #print(tuple(dp.split(os.sep)))
            if dp.split(os.sep)[nb_sub_dir+2] in ["LRI1", "LRI2"]:
                path_2_best_learning_steps.append(tuple(dp.split(os.sep)))
            
    return tuple_paths, path_2_best_learning_steps

def get_tuple_paths_of_arrays_SelectGammaVersion(
                name_dirs=["tests"], nb_sub_dir=1,
                dico_SelectGammaVersion={"DETERMINIST": [1,3], 
                                         "LRI1": [1,3],
                                         "LRI2": [1,3]},
                algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                     "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE", 
                                       "MIDDLE-BRUTE-FORCE"], 
                algos_4_showing=["DETERMINIST", "LRI1", "LRI2",
                                 "BEST-BRUTE-FORCE","BAD-BRUTE-FORCE"],
                exclude_dirs_files = ["html", "AVERAGE_RESULTS", 
                                      "AUTOMATE_INSTANCES_GAMES",
                                      "gamma", "npy", "csv", 
                                      NAME_RESULT_SHOW_VARS]):
    """
    autre version plus rapide 
    https://stackoverflow.com/a/59803793/2441026
    def run_fast_scandir(dir, exclude_dirs_files):    # dir: str, ext: list
        # https://stackoverflow.com/a/59803793/2441026
    
        subfolders, files = [], []
    
        for f in os.scandir(dir):
            if f.is_dir() \
                and f.name.split("_")[0] not in exclude_dirs_files \
                and f.name not in exclude_dirs_files:
                subfolders.append(f.path)
            # if f.is_file():
            #     if os.path.splitext(f.name)[1].lower() not in exclude_dirs_files:
            #         files.append(f.path)
            if f.is_file():
                if f.name.split(".")[-1] not in exclude_dirs_files:
                    files.append(f.path)
    
    
        for dir in list(subfolders):
            sf, f = run_fast_scandir(dir, exclude_dirs_files)
            subfolders.extend(sf)
            files.extend(f)
        return subfolders, files
    """
    lis_old = list()
    for name_dir in name_dirs:
        dirs = [rep for rep in os.listdir(name_dir) \
                if rep not in exclude_dirs_files \
                    and rep.split('_')[0] not in exclude_dirs_files]
        for dir_ in dirs:
            lis_old.extend([os.path.join(dp)
                        for dp, dn, fn in os.walk(os.path.join(name_dir,dir_)) 
                            for f in fn
                                ])
                        
    tuple_paths = list(); path_2_best_learning_steps = list()
    for dp in lis_old:
        tuple_path = dp.split(os.sep)
        algo = tuple_path[nb_sub_dir+2] if len(tuple_path) > nb_sub_dir+1 else ""
        gamma_version = int(list(tuple_path[nb_sub_dir].split("_")[-1])[-1])
        
        if len(tuple_path) > nb_sub_dir+1 \
            and algo in dico_SelectGammaVersion.keys() \
            and gamma_version in dico_SelectGammaVersion[algo] :
            tuple_paths.append( tuple(dp.split(os.sep)) )
            if tuple_path[nb_sub_dir+2] in ["LRI1", "LRI2"]:
                path_2_best_learning_steps.append(tuple(dp.split(os.sep)))
            
    return tuple_paths, path_2_best_learning_steps

def get_k_stop_4_periods(path_2_best_learning_steps, nb_sub_dir):
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
    index_name_df_kstop = None
    
    df_LRI_12_stop = None #pd.DataFrame()
    for tuple_path_2_algo in path_2_best_learning_steps:
        path_2_algo = os.path.join(*tuple_path_2_algo)
        algo = tuple_path_2_algo[nb_sub_dir+2]
        scenario = tuple_path_2_algo[nb_sub_dir].split("_")[3]
        gamma_version = "".join(
                        list(tuple_path_2_algo[nb_sub_dir].split("_")[4])[3:]
                        )
        df_al = pd.read_csv(
                    os.path.join(path_2_algo, "best_learning_steps.csv"),
                    index_col=0)
        index_name_df_kstop = scenario+"_"+gamma_version+"_"+algo+"_k_stop"
        index_mapper = {"k_stop":index_name_df_kstop}
        df_al.rename(index=index_mapper, inplace=True)
        if df_LRI_12_stop is None:
            df_LRI_12_stop = df_al
        else:
            df_LRI_12_stop = pd.concat([df_LRI_12_stop, df_al], axis=0)
            
    print("{}".format(df_LRI_12_stop))
    cols = df_LRI_12_stop.columns.tolist()
    indices = df_LRI_12_stop.index.tolist()
    df_k_stop = pd.DataFrame(
                    columns=cols, 
                    index=[index_name_df_kstop])
    # print("indices={} \n cols={}".format(indices, cols))
    for col in cols:
        best_index = None
        for index in indices:
            # print("best_index={}, index={}, col={}".format(best_index, index, col))
            if best_index is None:
                best_index = index
            elif df_LRI_12_stop.loc[best_index, col] < df_LRI_12_stop.loc[index, col]:
                best_index = index
                
            # print("df_LRI_12.loc[best_index, col]={}, df_LRI_12.loc[index, col]={}".format(
            #         df_LRI_12.loc[best_index, col], df_LRI_12.loc[index, col]))
                
        df_k_stop.loc[index_name_df_kstop, col] = df_LRI_12_stop.loc[best_index, col]
        
    return df_LRI_12_stop, df_k_stop

def DBG_get_array_turn_df_for_t(tuple_path_det_scen1, t=1, k_steps_args=250, nb_sub_dir=1,
                            algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                                 "BEST-BRUTE-FORCE",
                                                 "BAD-BRUTE-FORCE", 
                                                 "MIDDLE-BRUTE-FORCE"], 
                            algos_4_learning=["LRI1", "LRI2"]):
    
    df_arr_M_T_Ks = []
    df_b0_c0_pisg_pi0_T_K = []
    df_B_C_BB_CC_RU_M = []
    df_ben_cst_M_T_K = []
    
    algo_name = tuple_path_det_scen1[nb_sub_dir+2]
    path_to_variable = os.path.join(*tuple_path_det_scen1)
    
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
        
    return algo_name, \
            arr_pl_M_T_K_vars, \
            b0_s_T_K, c0_s_T_K, \
            B_is_M, C_is_M, B_is_M_T, C_is_M_T,\
            BENs_M_T_K, CSTs_M_T_K, \
            BB_is_M, CC_is_M, RU_is_M, BB_is_M_T, CC_is_M_T, RU_is_M_T,\
            pi_sg_plus_T, pi_sg_minus_T, \
            pi_0_plus_T, pi_0_minus_T
    
    
def get_tuple_from_vars_4_columns(nb_t_periods, t, algo_name, rate, price, 
                                  gamma_version, m_players, t_periods, k_steps, 
                                  scenario_name):
    if t is None:
        t_periods = nb_t_periods
        tu_mtk = list(it.product([algo_name], [rate], [price], [gamma_version],
                                 [scenario_name], 
                                 range(0, m_players), 
                                 range(0, t_periods), 
                                 range(0, k_steps)))
        tu_tk = list(it.product([algo_name], [rate], [price], [gamma_version],
                                [scenario_name],
                                range(0, t_periods), 
                                range(0, k_steps)))
        t_periods = list(range(0, t_periods))
    elif type(t) is list:
        t_periods = t
        tu_mtk = list(it.product([algo_name], [rate], [price], [gamma_version],
                                 [scenario_name],
                                 range(0, m_players), 
                                 t_periods, 
                                 range(0, k_steps)))
        tu_tk = list(it.product([algo_name], [rate], [price], [gamma_version],
                                [scenario_name],
                                t_periods, 
                                range(0, k_steps)))
    elif type(t) is int:
        t_periods = [t]
        tu_mtk = list(it.product([algo_name], [rate], [price], [gamma_version],
                                 [scenario_name],
                                 range(0, m_players), 
                                 t_periods, 
                                 range(0, k_steps)))
        tu_tk = list(it.product([algo_name], [rate], [price], [gamma_version],
                                [scenario_name],
                                t_periods, 
                                range(0, k_steps)))
                  
    print('t_periods = {}, nb_t_peirods={}'.format(t_periods, nb_t_periods))
    tu_m = list(it.product([algo_name], [rate], [price], [gamma_version],
                           [scenario_name],
                           range(0, m_players)))
    tu_mt = list(it.product([algo_name], [rate], [price], [gamma_version],
                           [scenario_name],
                           range(0, m_players), 
                           range(0, nb_t_periods)
                           ))
                
    variables = list(fct_aux.AUTOMATE_INDEX_ATTRS.keys())
    
    return t_periods, tu_mtk, tu_tk, tu_m, tu_mt, variables

def turn_arrays_2_2D_learning_algos(arr_pl_M_T_K_vars, 
                                    arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW, 
                                    b0_s_T_K, c0_s_T_K,
                                    B_is_M, C_is_M,
                                    BENs_M_T_K, CSTs_M_T_K,
                                    BB_is_M, CC_is_M, RU_is_M,
                                    pi_sg_plus_T, pi_sg_minus_T,
                                    pi_0_plus_T, pi_0_minus_T,
                                    t_periods, k_steps, 
                                    tu_mtk, tu_tk, tu_m, tu_mt, variables):
    
    arr_pl_M_T_K_vars_t = arr_pl_M_T_K_vars[:, t_periods, :, :]
    ## process of arr_pl_M_T_K_vars 
    arr_pl_M_T_K_vars_2D = arr_pl_M_T_K_vars_t.reshape(
                                -1, 
                                arr_pl_M_T_K_vars.shape[3])
    df_lri_x = pd.DataFrame(data=arr_pl_M_T_K_vars_2D, 
                            index=tu_mtk, 
                            columns=variables)
        
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
        pi_0_minus_T_K_2D.append([ pi_0_minus_T[tx] ]*k_steps)
        pi_0_plus_T_K_2D.append([ pi_0_plus_T[tx] ]*k_steps)
        pi_sg_minus_T_K_2D.append([ pi_sg_minus_T[tx] ]*k_steps)
        pi_sg_plus_T_K_2D.append([ pi_sg_plus_T[tx] ]*k_steps)
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
    
    ## process of df_ben_cst_M_T_K
    BENs_M_T_K_1D = BENs_M_T_K[:,t_periods,:].reshape(-1)
    CSTs_M_T_K_1D = CSTs_M_T_K[:,t_periods,:].reshape(-1)
    df_ben_cst_M_T_K_lri = pd.DataFrame({
        'ben':BENs_M_T_K_1D, 'cst':CSTs_M_T_K_1D}, index=tu_mtk)
    
    ## process of df_B_C_BB_CC_RU_M
    df_B_C_BB_CC_RU_M_lri \
        = pd.DataFrame({
                "B":B_is_M, "C":C_is_M, 
                "BB":BB_is_M, "CC":CC_is_M, "RU":RU_is_M}, 
            index=tu_m)
        
    # ajouter un dataframe pour df_B_C_BB_CC_RU_M_lri contenant les VARS suivantes
    # ["k_stop", "PROD", "CONS", "b0", "c0", 
    #    "pi_sg_plus","pi_sg_minus", "B", "C", "BB", "CC", "RU"]
    # create dataframe df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T_lri_x named df_M_T_lri_x
    # tu_mt
    selected_cols = ["state_i","k_stop", "PROD", "CONS", "b0", "c0", 
                     "pi_sg_plus","pi_sg_minus", "B", "C", "BB", "CC", "RU", 
                     "Si", "mode_i", "ben", "cst", "Cicum", "Picum", "set"]
    id_cols = [ AUTOMATE_INDEX_ATTRS_NEW[col] for col in selected_cols]
    arr_pl_M_T_KSTOP_vars_2D = arr_pl_M_T_KSTOP_vars[:,:, id_cols]\
                                .reshape(-1, len(id_cols))
    df_M_T_lri_x = None
    df_M_T_lri_x = pd.DataFrame(data=arr_pl_M_T_KSTOP_vars_2D, 
                                index=tu_mt, 
                                columns=selected_cols)
    
    return df_lri_x, df_b0_c0_pisg_pi0_T_K_lri, \
            df_ben_cst_M_T_K_lri, df_B_C_BB_CC_RU_M_lri, \
            df_M_T_lri_x

def turn_arrays_2_2D_4_not_learning_algos(arr_pl_M_T_K_vars, 
                                          arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW, 
                                          b0_s_T_K, c0_s_T_K,
                                          B_is_M, C_is_M,
                                          BENs_M_T_K, CSTs_M_T_K,
                                          BB_is_M, CC_is_M, RU_is_M,
                                          pi_sg_plus_T, pi_sg_minus_T,
                                          pi_0_plus_T, pi_0_minus_T,
                                          t_periods, k_steps,
                                          tu_mtk, tu_tk, tu_m, tu_mt, variables):
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
    
    ## process of df_B_C_BB_CC_RU_M
    df_B_C_BB_CC_RU_M_det = pd.DataFrame({
        "B":B_is_M, "C":C_is_M, 
        "BB":BB_is_M,"CC":CC_is_M,"RU":RU_is_M,}, index=tu_m)
    
    ## process of 
    ## process of 
    
    # ajouter un dataframe pour df_B_C_BB_CC_RU_M_lri contenant les VARS suivantes
    # ["k_stop", "PROD", "CONS", "b0", "c0", 
    #    "pi_sg_plus","pi_sg_minus", "B", "C", "BB", "CC", "RU"]
    # create dataframe df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T_lri
    # tu_mt
    selected_cols = ["state_i","k_stop", "PROD", "CONS", "b0", "c0", 
                     "pi_sg_plus","pi_sg_minus", "B", "C", "BB", "CC", "RU", 
                     "Si", "mode_i", "ben", "cst", "Cicum", "Picum", "set"]
    id_cols = [ AUTOMATE_INDEX_ATTRS_NEW[col] for col in selected_cols]
    arr_pl_M_T_KSTOP_vars_2D = arr_pl_M_T_KSTOP_vars[:,:, id_cols]\
                                .reshape(-1, len(id_cols))
    df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T_det \
        = pd.DataFrame(data=arr_pl_M_T_KSTOP_vars_2D, 
                       index=tu_mt, 
                       columns=selected_cols)
    print("df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T_det={}, type={}".format(
            df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T_det.shape, 
            type(df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T_det) ))
    
    return df_rd_det, df_b0_c0_pisg_pi0_T_K_det, \
            df_ben_cst_M_T_K_det, \
            df_B_C_BB_CC_RU_M_det, \
            df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T_det 
    
def insert_index_as_df_columns(df_arr_M_T_Ks, columns_ind):
    """
    insert index as columns of dataframes
    """
    columns_df = df_arr_M_T_Ks.columns.to_list()
    indices = list(df_arr_M_T_Ks.index)
    df_ind = pd.DataFrame(indices,columns=columns_ind)
    df_arr_M_T_Ks = pd.concat([df_ind.reset_index(), 
                                df_arr_M_T_Ks.reset_index()],
                              axis=1, ignore_index=True)
    df_arr_M_T_Ks.drop(df_arr_M_T_Ks.columns[[0]], axis=1, inplace=True)
    df_arr_M_T_Ks.columns = columns_ind+["old_index"]+columns_df
    df_arr_M_T_Ks.pop("old_index")
    return df_arr_M_T_Ks
    
def get_array_turn_df_for_t(tuple_paths, df_LRI_12_stop, 
                            t=1, k_steps_args=250, nb_sub_dir=1,
                            algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                                 "BEST-BRUTE-FORCE",
                                                 "BAD-BRUTE-FORCE", 
                                                 "MIDDLE-BRUTE-FORCE"], 
                            algos_4_learning=["LRI1", "LRI2"]):
    
    df_arr_M_T_Ks = []
    df_b0_c0_pisg_pi0_T_K = []
    df_ben_cst_M_T_K = []
    df_B_C_BB_CC_RU_M = []
    df_B_C_BB_CC_RU_M_T = []
    
    cpt = 0
    for tuple_path in tuple_paths:
        algo_name = tuple_path[nb_sub_dir+2]
        path_to_variable = os.path.join(*tuple_path)
        
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
        
    
        price = tuple_path[nb_sub_dir+1].split("_")[3] \
                +"_"+tuple_path[nb_sub_dir+1].split("_")[-1]
        algo_name = tuple_path[nb_sub_dir+2];
        rate = tuple_path[nb_sub_dir+3] if algo_name in algos_4_learning else 0
        gamma_version = "".join(list(tuple_path[nb_sub_dir].split("_")[4])[3:])
        scenario_name = tuple_path[nb_sub_dir].split("_")[3]
        
        # array of M_PLAYERS, T_PERIODS, VARS
        # VARS = AUTOMATE_INDEX_ATTRS 
        #        + ["k_stop", "PROD", "CONS", "b0", "c0", 
        #           "pi_sg_plus","pi_sg_minus", "B", "C", "BB", "CC", "RU"]
        arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW \
            = add_new_vars_2_arr(
                algo_name=algo_name, 
                scenario_name=scenario_name, 
                gamma_version=gamma_version,
                df_LRI_12_kstop=df_LRI_12_stop,
                arr_pl_M_T_K_vars=arr_pl_M_T_K_vars,
                b0_s_T_K=b0_s_T_K, c0_s_T_K=c0_s_T_K,
                B_is_M=B_is_M, C_is_M=C_is_M, B_is_M_T=B_is_M_T, C_is_M_T=C_is_M_T,
                BENs_M_T_K=BENs_M_T_K, CSTs_M_T_K=CSTs_M_T_K,
                BB_is_M=BB_is_M, CC_is_M=CC_is_M, RU_is_M=RU_is_M,
                BB_is_M_T=BB_is_M_T, CC_is_M_T=CC_is_M_T, RU_is_M_T=RU_is_M_T,
                pi_sg_plus_T=pi_sg_plus_T, pi_sg_minus_T=pi_sg_minus_T,
                pi_0_plus_T=pi_0_plus_T, pi_0_minus_T=pi_0_minus_T, 
                algos_4_no_learning=algos_4_no_learning)
        
        m_players = arr_pl_M_T_K_vars.shape[0]
        k_steps = arr_pl_M_T_K_vars.shape[2] if arr_pl_M_T_K_vars.shape == 4 \
                                             else k_steps_args
                    
        t_periods = None; tu_mtk = None; tu_tk = None; tu_m = None
        nb_t_periods = arr_pl_M_T_K_vars.shape[1]
        t_periods, tu_mtk, tu_tk, tu_m, tu_mt, variables \
            = get_tuple_from_vars_4_columns(nb_t_periods, t, algo_name, rate, price, 
                                  gamma_version, m_players, t_periods, k_steps, 
                                  scenario_name)
            
        if algo_name in algos_4_learning:
            df_lri_x, df_b0_c0_pisg_pi0_T_K_lri = None, None
            df_ben_cst_M_T_K_lri, df_B_C_BB_CC_RU_M_lri = None, None
            df_M_T_lri_x = None
            
            df_lri_x, df_b0_c0_pisg_pi0_T_K_lri, \
            df_ben_cst_M_T_K_lri, df_B_C_BB_CC_RU_M_lri, \
            df_M_T_lri_x \
                = turn_arrays_2_2D_learning_algos(
                    arr_pl_M_T_K_vars, 
                    arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW, 
                    b0_s_T_K, c0_s_T_K,
                    B_is_M, C_is_M,
                    BENs_M_T_K, CSTs_M_T_K,
                    BB_is_M, CC_is_M, RU_is_M,
                    pi_sg_plus_T, pi_sg_minus_T,
                    pi_0_plus_T, pi_0_minus_T,
                    t_periods, k_steps, 
                    tu_mtk, tu_tk, tu_m, tu_mt, variables)
                
            df_arr_M_T_Ks.append(df_lri_x)
            df_b0_c0_pisg_pi0_T_K.append(df_b0_c0_pisg_pi0_T_K_lri)
            df_ben_cst_M_T_K.append(df_ben_cst_M_T_K_lri)
            df_B_C_BB_CC_RU_M.append(df_B_C_BB_CC_RU_M_lri)
            df_B_C_BB_CC_RU_M_T.append(df_M_T_lri_x)
            
        else:
            df_rd_det, df_b0_c0_pisg_pi0_T_K_det = None, None
            df_ben_cst_M_T_K_det, df_B_C_BB_CC_RU_M_det = None, None
            df_M_T_det = None
            
            df_rd_det, df_b0_c0_pisg_pi0_T_K_det, \
            df_ben_cst_M_T_K_det, \
            df_B_C_BB_CC_RU_M_det, \
            df_M_T_det \
                = turn_arrays_2_2D_4_not_learning_algos(
                    arr_pl_M_T_K_vars, 
                    arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW, 
                    b0_s_T_K, c0_s_T_K,
                    B_is_M, C_is_M,
                    BENs_M_T_K, CSTs_M_T_K,
                    BB_is_M, CC_is_M, RU_is_M,
                    pi_sg_plus_T, pi_sg_minus_T,
                    pi_0_plus_T, pi_0_minus_T,
                    t_periods, k_steps,
                    tu_mtk, tu_tk, tu_m, tu_mt, variables
                    )
                
            df_arr_M_T_Ks.append(df_rd_det)    
            df_b0_c0_pisg_pi0_T_K.append(df_b0_c0_pisg_pi0_T_K_det)
            df_ben_cst_M_T_K.append(df_ben_cst_M_T_K_det)
            df_B_C_BB_CC_RU_M.append(df_B_C_BB_CC_RU_M_det)
            df_B_C_BB_CC_RU_M_T.append(df_M_T_det)
            
        cpt += 1 
     
    # merge all dataframes
    df_arr_M_T_Ks = pd.concat(df_arr_M_T_Ks, axis=0)
    df_ben_cst_M_T_K = pd.concat(df_ben_cst_M_T_K, axis=0)
    df_b0_c0_pisg_pi0_T_K = pd.concat(df_b0_c0_pisg_pi0_T_K, axis=0)
    df_B_C_BB_CC_RU_M = pd.concat(df_B_C_BB_CC_RU_M, axis=0)
    df_B_C_BB_CC_RU_M_T = pd.concat(df_B_C_BB_CC_RU_M_T, axis=0)
        
        
    # insert index as columns of dataframes
    ###  df_arr_M_T_Ks
    columns_ind = ["algo","rate","prices","gamma_version","scenario","pl_i","t","k"]
    df_arr_M_T_Ks = insert_index_as_df_columns(df_arr_M_T_Ks, columns_ind)
    ###  df_ben_cst_M_T_K
    columns_ind = ["algo","rate","prices","gamma_version","scenario","pl_i","t","k"]
    df_ben_cst_M_T_K = insert_index_as_df_columns(df_ben_cst_M_T_K, columns_ind)
    df_ben_cst_M_T_K["state_i"] = df_arr_M_T_Ks["state_i"]
    ###  df_b0_c0_pisg_pi0_T_K
    columns_ind = ["algo","rate","prices","gamma_version","scenario","t","k"]
    df_b0_c0_pisg_pi0_T_K = insert_index_as_df_columns(df_b0_c0_pisg_pi0_T_K, columns_ind)
    ###  df_B_C_BB_CC_RU_M
    columns_ind = ["algo","rate","prices","gamma_version","scenario","pl_i"]
    df_B_C_BB_CC_RU_M = insert_index_as_df_columns(df_B_C_BB_CC_RU_M, columns_ind)
    ### df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T
    columns_ind = ["algo","rate","prices","gamma_version","scenario","pl_i","t"]
    df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T \
        = insert_index_as_df_columns(df_B_C_BB_CC_RU_M_T, 
                                     columns_ind)
    
    
        
    return df_arr_M_T_Ks, df_ben_cst_M_T_K, \
            df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M, \
            df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T
            
    
# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> fin
# _____________________________________________________________________________ 

# _____________________________________________________________________________ 
#               
#               add new variables to array of players  --> debut
#                   pi_sg_{+,-}, b0, c0, B, C, BB, CC, RU 
# _____________________________________________________________________________ 
def add_new_vars_2_arr(algo_name, scenario_name, gamma_version,
                       df_LRI_12_kstop,
                       arr_pl_M_T_K_vars,
                       b0_s_T_K, c0_s_T_K,
                       B_is_M, C_is_M, B_is_M_T, C_is_M_T,
                       BENs_M_T_K, CSTs_M_T_K,
                       BB_is_M, CC_is_M, RU_is_M, BB_is_M_T, CC_is_M_T, RU_is_M_T,
                       pi_sg_plus_T, pi_sg_minus_T,
                       pi_0_plus_T, pi_0_minus_T, 
                       algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                             "BEST-BRUTE-FORCE",
                                             "BAD-BRUTE-FORCE", 
                                             "MIDDLE-BRUTE-FORCE"]):
    """
    Version compute B,BB, C, CC
    add new variables to array of players  --> debut
                   pi_sg_{+,-}, b0, c0, B, C, BB, CC, RU 
    """
    vars_2_add = ["k_stop", "PROD", "CONS", 
                  "b0", "c0", "pi_sg_plus","pi_sg_minus", 
                  "B", "C", "BB", "CC", "RU", "ben", "cst", "Cicum", "Picum"]
    dico_vars2Add = dict()
    for i in range(0, len(vars_2_add)):
        nb_attrs = len(fct_aux.AUTOMATE_INDEX_ATTRS)
        dico_vars2Add[vars_2_add[i]] = nb_attrs+i
    
    AUTOMATE_INDEX_ATTRS_NEW = {**fct_aux.AUTOMATE_INDEX_ATTRS, 
                                **dico_vars2Add}
    
    arr_pl_M_T_KSTOP_vars = None
    arr_pl_M_T_KSTOP_vars = np.zeros((arr_pl_M_T_K_vars.shape[0],
                                      arr_pl_M_T_K_vars.shape[1],
                                      len(AUTOMATE_INDEX_ATTRS_NEW)), 
                                dtype=object)
    
    t_periods = arr_pl_M_T_K_vars.shape[1]
    for t in range(0, t_periods):
        ben_M, cst_M = None, None
        k_stop = None
        if algo_name in algos_4_no_learning:
            arr_pl_M_T_KSTOP_vars[:,t,list(fct_aux.AUTOMATE_INDEX_ATTRS.values()) ] \
                = arr_pl_M_T_K_vars[:,t,:]
            ben_M = BENs_M_T_K[:,t]
            cst_M = CSTs_M_T_K[:,t]
        else:
            index_kstop = scenario_name+"_"+gamma_version+"_"+algo_name+"_"+"k_stop"
            k_stop = df_LRI_12_kstop.loc[index_kstop, str(t)]
            arr_pl_M_T_KSTOP_vars[:,t,list(fct_aux.AUTOMATE_INDEX_ATTRS.values()) ] \
                = arr_pl_M_T_K_vars[:,t,k_stop,:]
            ben_M = BENs_M_T_K[:,t, k_stop]
            cst_M = CSTs_M_T_K[:,t, k_stop]
            
        arr_pl_M_T_KSTOP_vars[:,t,AUTOMATE_INDEX_ATTRS_NEW["ben"]] = ben_M
        arr_pl_M_T_KSTOP_vars[:,t,AUTOMATE_INDEX_ATTRS_NEW["cst"]] = cst_M
                
        b0_t, c0_t = None, None
        b0_0_t_minus_1, c0_0_t_minus_1 = None, None
        pi_sg_plus_t, pi_sg_minus_t = None, None
        if algo_name in algos_4_no_learning:
            b0_t, c0_t = b0_s_T_K[t], c0_s_T_K[t]
            b0_0_t_minus_1 = b0_s_T_K[:t+1]
            c0_0_t_minus_1 = c0_s_T_K[:t+1]
            pi_sg_plus_t = pi_sg_plus_T[t]
            pi_sg_minus_t = pi_sg_minus_T[t]
        else:
            b0_t, c0_t = b0_s_T_K[t, k_stop], c0_s_T_K[t, k_stop]
            b0_0_t_minus_1 = b0_s_T_K[:t+1, k_stop]
            c0_0_t_minus_1 = c0_s_T_K[:t+1, k_stop]
            pi_sg_plus_t = pi_sg_plus_T[t]
            pi_sg_minus_t = pi_sg_minus_T[t]
        
        for num_pl_i in range(arr_pl_M_T_K_vars.shape[0]):
            PROD_i_0_t_minus_1, CONS_i_0_t_minus_1 = None, None
            Pi_0_t_minus_1, Ci_0_t_minus_1 = None, None
            if algo_name in algos_4_no_learning:
                PROD_i_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, 
                                           fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
                CONS_i_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, 
                                           fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
                Pi_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, 
                                           fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
                Ci_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, 
                                           fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
                ben_M = BENs_M_T_K[:,t]
                cst_M = CSTs_M_T_K[:,t]
            else:
                PROD_i_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, k_stop,
                                           fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
                CONS_i_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, k_stop,
                                           fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
                Pi_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1,  k_stop,
                                           fct_aux.AUTOMATE_INDEX_ATTRS["Pi"]]
                Ci_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1,  k_stop,
                                           fct_aux.AUTOMATE_INDEX_ATTRS["Ci"]]
                ben_M = BENs_M_T_K[:,t, k_stop]
                cst_M = CSTs_M_T_K[:,t, k_stop]
            
            Pi_cum = np.sum(Pi_0_t_minus_1, axis=0)
            Ci_cum = np.sum(Ci_0_t_minus_1, axis=0)
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["Picum"]] = Pi_cum
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["Cicum"]] = Ci_cum    
            
            PROD_i = np.sum(PROD_i_0_t_minus_1, axis=0) 
            CONS_i = np.sum(CONS_i_0_t_minus_1, axis=0)
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["PROD"]] = PROD_i
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["CONS"]] = CONS_i
            
            Bi_0_t_minus_1 = np.sum(b0_0_t_minus_1 * PROD_i_0_t_minus_1, axis=0)
            Ci_0_t_minus_1 = np.sum(c0_0_t_minus_1 * CONS_i_0_t_minus_1, axis=0)
            BBi_0_t_minus_1 = pi_sg_plus_t * PROD_i
            CCi_0_t_minus_1 = pi_sg_minus_t * CONS_i
            RUi_0_t_minus_1 = BBi_0_t_minus_1 - CCi_0_t_minus_1
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["B"]] = B_is_M_T[num_pl_i, t]
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["C"]] = C_is_M_T[num_pl_i, t]
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["BB"]] = BB_is_M_T[num_pl_i, t]
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["CC"]] = CC_is_M_T[num_pl_i, t]
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["RU"]] = RU_is_M_T[num_pl_i, t]
            
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["b0"]] = b0_t
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["c0"]] = c0_t
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                    AUTOMATE_INDEX_ATTRS_NEW["pi_sg_plus"]] = pi_sg_plus_t
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                    AUTOMATE_INDEX_ATTRS_NEW["pi_sg_minus"]] = pi_sg_minus_t
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                    AUTOMATE_INDEX_ATTRS_NEW["k_stop"]] = k_stop
            
    # checkout BB, CC, RU
    print("\n _________ Resume {} {}, gamma ={}_________".format(
            algo_name, scenario_name, gamma_version))
    cpt, cpt_BB_OK, cpt_CC_OK, cpt_RU_OK = 0, 0, 0, 0
    for num_pl_i in range(arr_pl_M_T_KSTOP_vars.shape[0]):
        BBi = BB_is_M[num_pl_i]
        CCi = CC_is_M[num_pl_i]
        RUi = RU_is_M[num_pl_i]
        BBi_0_T_minus_1 = arr_pl_M_T_KSTOP_vars[num_pl_i, t_periods-1,
                                                AUTOMATE_INDEX_ATTRS_NEW["BB"]]
        CCi_0_T_minus_1 = arr_pl_M_T_KSTOP_vars[num_pl_i, t_periods-1,
                                                AUTOMATE_INDEX_ATTRS_NEW["CC"]]
        RUi_0_T_minus_1 = arr_pl_M_T_KSTOP_vars[num_pl_i, t_periods-1,
                                                AUTOMATE_INDEX_ATTRS_NEW["RU"]]
        if np.abs(BBi - BBi_0_T_minus_1) < pow(10,-1):
            cpt_BB_OK += 1
        if np.abs(CCi - CCi_0_T_minus_1) < pow(10,-1):
            cpt_CC_OK += 1
        else:
            print("player {}, CCi={}, CCi_0_T_minus_1={}".format(num_pl_i, CCi, CCi_0_T_minus_1 ))
            pass
        if np.abs(RUi - RUi_0_T_minus_1) < pow(10,-1):
            cpt_RU_OK += 1
        cpt += 1
    print("BBis OK?: {}, CCis OK?: {}, RUis OK?: {},".format(
            round(cpt_BB_OK/cpt, 2), round(cpt_CC_OK/cpt, 2), 
            round(cpt_RU_OK/cpt, 2)))
            
    return arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW

def add_new_vars_2_arr_OLD_VERSION(algo_name, scenario_name, gamma_version,
                       df_LRI_12_kstop,
                       arr_pl_M_T_K_vars,
                       b0_s_T_K, c0_s_T_K,
                       B_is_M, C_is_M,
                       BENs_M_T_K, CSTs_M_T_K,
                       BB_is_M, CC_is_M, RU_is_M,
                       pi_sg_plus_T, pi_sg_minus_T,
                       pi_0_plus_T, pi_0_minus_T, 
                       algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                             "BEST-BRUTE-FORCE",
                                             "BAD-BRUTE-FORCE", 
                                             "MIDDLE-BRUTE-FORCE"]):
    """
    Version compute B,BB, C, CC
    add new variables to array of players  --> debut
                   pi_sg_{+,-}, b0, c0, B, C, BB, CC, RU 
    """
    vars_2_add = ["k_stop", "PROD", "CONS", 
                  "b0", "c0", "pi_sg_plus","pi_sg_minus", 
                  "B", "C", "BB", "CC", "RU"]
    dico_vars2Add = dict()
    for i in range(0, len(vars_2_add)):
        nb_attrs = len(fct_aux.AUTOMATE_INDEX_ATTRS)
        dico_vars2Add[vars_2_add[i]] = nb_attrs+i
    
    AUTOMATE_INDEX_ATTRS_NEW = {**fct_aux.AUTOMATE_INDEX_ATTRS, 
                                **dico_vars2Add}
    
    arr_pl_M_T_KSTOP_vars = None
    arr_pl_M_T_KSTOP_vars = np.zeros((arr_pl_M_T_K_vars.shape[0],
                                      arr_pl_M_T_K_vars.shape[1],
                                      len(AUTOMATE_INDEX_ATTRS_NEW)), 
                                dtype=object)
    
    # arr_pl_M_T_KSTOP_vars[:,:,list(fct_aux.AUTOMATE_INDEX_ATTRS.values()) ] \
    #     = arr_pl_M_T_K_vars[:,:,:]
        
    
    t_periods = arr_pl_M_T_K_vars.shape[1]
    for t in range(0, t_periods):
        k_stop = None
        if algo_name in algos_4_no_learning:
            arr_pl_M_T_KSTOP_vars[:,t,list(fct_aux.AUTOMATE_INDEX_ATTRS.values()) ] \
                = arr_pl_M_T_K_vars[:,t,:] 
        else:
            index_kstop = scenario_name+"_"+gamma_version+"_"+algo_name+"_"+"k_stop"
            k_stop = df_LRI_12_stop.loc[index_kstop, str(t)]
            arr_pl_M_T_KSTOP_vars[:,t,list(fct_aux.AUTOMATE_INDEX_ATTRS.values()) ] \
                = arr_pl_M_T_K_vars[:,t,k_stop,:]
            
            
        # index_kstop = scenario_name+"_"+gamma_version+"_"+algo_name+"_"+"k_stop"
        # k_stop = df_LRI_12_stop.loc[index_kstop, str(t)]
        # arr_pl_M_T_KSTOP_vars[:,t,list(fct_aux.AUTOMATE_INDEX_ATTRS.values()) ] \
        #     = arr_pl_M_T_K_vars[:,t,:] \
        #         if algo_name in algos_4_no_learning \
        #         else arr_pl_M_T_K_vars[:,t,k_stop,:]
                
                
        b0_t, c0_t = None, None
        b0_0_t_minus_1, c0_0_t_minus_1 = None, None
        pi_sg_plus_t, pi_sg_minus_t = None, None
        if algo_name in algos_4_no_learning:
            b0_t, c0_t = b0_s_T_K[t], c0_s_T_K[t]
            b0_0_t_minus_1 = b0_s_T_K[:t+1]
            c0_0_t_minus_1 = c0_s_T_K[:t+1]
            pi_sg_plus_t = pi_sg_plus_T[t]
            pi_sg_minus_t = pi_sg_minus_T[t]
        else:
            b0_t, c0_t = b0_s_T_K[t, k_stop], c0_s_T_K[t, k_stop]
            b0_0_t_minus_1 = b0_s_T_K[:t+1, k_stop]
            c0_0_t_minus_1 = c0_s_T_K[:t+1, k_stop]
            pi_sg_plus_t = pi_sg_plus_T[t]
            pi_sg_minus_t = pi_sg_minus_T[t]
        
        for num_pl_i in range(arr_pl_M_T_K_vars.shape[0]):
            PROD_i_0_t_minus_1, CONS_i_0_t_minus_1 = None, None
            if algo_name in algos_4_no_learning:
                PROD_i_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, 
                                           fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
                CONS_i_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, 
                                           fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
            else:
                PROD_i_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, k_stop,
                                           fct_aux.AUTOMATE_INDEX_ATTRS["prod_i"]]
                CONS_i_0_t_minus_1 = arr_pl_M_T_K_vars[num_pl_i, :t+1, k_stop,
                                           fct_aux.AUTOMATE_INDEX_ATTRS["cons_i"]]
                
            PROD_i = np.sum(PROD_i_0_t_minus_1, axis=0) 
            CONS_i = np.sum(CONS_i_0_t_minus_1, axis=0)
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["PROD"]] = PROD_i
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["CONS"]] = CONS_i
            
            Bi_0_t_minus_1 = np.sum(b0_0_t_minus_1 * PROD_i_0_t_minus_1, axis=0)
            Ci_0_t_minus_1 = np.sum(c0_0_t_minus_1 * CONS_i_0_t_minus_1, axis=0)
            BBi_0_t_minus_1 = pi_sg_plus_t * PROD_i
            CCi_0_t_minus_1 = pi_sg_minus_t * CONS_i
            RUi_0_t_minus_1 = BBi_0_t_minus_1 - CCi_0_t_minus_1
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["B"]] = Bi_0_t_minus_1
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["C"]] = Ci_0_t_minus_1
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["BB"]] = BBi_0_t_minus_1
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["CC"]] = CCi_0_t_minus_1
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["RU"]] = RUi_0_t_minus_1
            
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["b0"]] = b0_t
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                                  AUTOMATE_INDEX_ATTRS_NEW["c0"]] = c0_t
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                    AUTOMATE_INDEX_ATTRS_NEW["pi_sg_plus"]] = pi_sg_plus_t
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                    AUTOMATE_INDEX_ATTRS_NEW["pi_sg_minus"]] = pi_sg_minus_t
            arr_pl_M_T_KSTOP_vars[num_pl_i, t,
                    AUTOMATE_INDEX_ATTRS_NEW["k_stop"]] = k_stop
            
    # checkout BB, CC, RU
    print("\n _________ Resume {} {}, gamma ={}_________".format(
            algo_name, scenario_name, gamma_version))
    cpt, cpt_BB_OK, cpt_CC_OK, cpt_RU_OK = 0, 0, 0, 0
    for num_pl_i in range(arr_pl_M_T_KSTOP_vars.shape[0]):
        BBi = BB_is_M[num_pl_i]
        CCi = CC_is_M[num_pl_i]
        RUi = RU_is_M[num_pl_i]
        BBi_0_T_minus_1 = arr_pl_M_T_KSTOP_vars[num_pl_i, t_periods-1,
                                                AUTOMATE_INDEX_ATTRS_NEW["BB"]]
        CCi_0_T_minus_1 = arr_pl_M_T_KSTOP_vars[num_pl_i, t_periods-1,
                                                AUTOMATE_INDEX_ATTRS_NEW["CC"]]
        RUi_0_T_minus_1 = arr_pl_M_T_KSTOP_vars[num_pl_i, t_periods-1,
                                                AUTOMATE_INDEX_ATTRS_NEW["RU"]]
        if np.abs(BBi - BBi_0_T_minus_1) < pow(10,-1):
            cpt_BB_OK += 1
        if np.abs(CCi - CCi_0_T_minus_1) < pow(10,-1):
            cpt_CC_OK += 1
        else:
            print("player {}, CCi={}, CCi_0_T_minus_1={}".format(num_pl_i, CCi, CCi_0_T_minus_1 ))
            pass
        if np.abs(RUi - RUi_0_T_minus_1) < pow(10,-1):
            cpt_RU_OK += 1
        cpt += 1
    print("BBis OK?: {}, CCis OK?: {}, RUis OK?: {},".format(
            round(cpt_BB_OK/cpt, 2), round(cpt_CC_OK/cpt, 2), 
            round(cpt_RU_OK/cpt, 2)))
            
          
    return arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW
# _____________________________________________________________________________ 
#               
#               add new variables to array of players  --> fin
#                   PROD, CONS, pi_sg_{+,-}, b0, c0, B, C, BB, CC, RU 
# _____________________________________________________________________________ 


###############################################################################
#
#               representation des variables : debut
#
###############################################################################

# _____________________________________________________________________________ 
#               
#                   plot B,C 4 various gamma_version 4 each scenario
#                                       --> debut
# _____________________________________________________________________________ 
def plot_gamma_version_BC(df_ra_pr, rate, price, scenario):
    
    cols_2_group = ["algo","gamma_version"]
    
    cols = ["B","C"]; 
    df_res = df_ra_pr.groupby(cols_2_group)[cols]\
                .agg({"B": [np.mean, np.std, np.min, np.max], 
                      "C": [np.mean, np.std, np.min, np.max]
                      })
    df_res.columns = ["_".join(x) for x in df_res.columns.ravel()]
    df_res = df_res.reset_index()
    
    aggs = ["amin", "amax", "std", "mean"]
    tooltips = [("{}_{}".format(col, agg), "@{}_{}".format(col, agg)) 
                for (col, agg) in it.product(cols, aggs)]
    TOOLS[7] = HoverTool(tooltips = tooltips)
    
    new_cols = [col[1].split("@")[1] 
                for col in tooltips if col[1].split("_")[1] == "mean"]
    print('new_cols={}, df_res.cols={}'.format(new_cols, df_res.columns))
    
    x = list(map(tuple,list(df_res[cols_2_group].values)))
    px = figure(x_range=FactorRange(*x), 
                y_range=(0, df_res[new_cols].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
           
    data = dict(x = x, 
                B_mean=df_res.B_mean.tolist(), C_mean=df_res.C_mean.tolist(), 
                B_std=df_res.B_std.tolist(), C_std=df_res.C_std.tolist(),
                B_amin=df_res.B_amin.tolist(), C_amin=df_res.C_amin.tolist(),
                B_amax=df_res.B_amax.tolist(), C_amax=df_res.C_amax.tolist()
                )

    print("data keys={}".format(data.keys()))
    source = ColumnDataSource(data = data)
    
    width= 0.2 #0.5
    
    px.vbar(x=dodge('x', -0.3+0*width, range=px.x_range), top=new_cols[0], 
                    width=width, source=source, legend_label=new_cols[0], 
                    color="#c9d9d3")
    px.vbar(x=dodge('x', -0.3+1*width, range=px.x_range), top=new_cols[1], 
                    width=width, source=source, legend_label=new_cols[1], 
                    color="#718dbf")
    
    title = "comparison Gamma_version B,C({},rate:{}, price={})".format(scenario, rate, price)
    px.title.text = title
    px.y_range.start = min(0, df_res.B_mean.min() - 1, 
                           df_res.C_mean.min() - 1)
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "algo"
    px.yaxis.axis_label = "values"
    
    return px
    
def plot_comparaison_gamma_version_BC(df_B_C_BB_CC_RU_M):
    rates = df_B_C_BB_CC_RU_M.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df_B_C_BB_CC_RU_M.prices.unique().tolist()
    scenarios = df_B_C_BB_CC_RU_M.scenario.unique().tolist()
    
    dico_pxs = dict()
    for rate, price, scenario in it.product(rates, prices, scenarios):
        mask_ra_pr = ((df_B_C_BB_CC_RU_M.rate == rate) \
                      | (df_B_C_BB_CC_RU_M.rate == 0)) \
                        & (df_B_C_BB_CC_RU_M.prices == price) \
                        & (df_B_C_BB_CC_RU_M.scenario == scenario) 
        df_ra_pr = df_B_C_BB_CC_RU_M[mask_ra_pr].copy()
        
        pxs_pr_ra_sc = plot_gamma_version_BC(df_ra_pr, rate, price, scenario)
        pxs_pr_ra_sc.legend.click_policy="hide"
        
        if (price, rate, scenario) not in dico_pxs.keys():
            dico_pxs[(price, rate, scenario)] \
                = [pxs_pr_ra_sc]
        else:
            dico_pxs[(price, rate, scenario)].append(pxs_pr_ra_sc)
        
    rows_RU_C_B_CC_BB = list()
    for key, pxs_pr_ra_sc in dico_pxs.items():
        col_px_sts = column(pxs_pr_ra_sc)
        rows_RU_C_B_CC_BB.append(col_px_sts)
    rows_RU_C_B_CC_BB=column(children=rows_RU_C_B_CC_BB, 
                             sizing_mode='stretch_both')
    return rows_RU_C_B_CC_BB
# _____________________________________________________________________________ 
#               
#                   plot B,C 4 various gamma_version 4 each scenario 
#                                       --> fin
# _____________________________________________________________________________

# _____________________________________________________________________________ 
#               
#                   plot RU 4 various gamma_version 4 each scenario 
#                            --> debut
# _____________________________________________________________________________ 
def OLD_plot_gamma_version_RU(df_ra_pr, rate, price, scenario):
    
    cols_2_group = ["algo","gamma_version"]
    
    cols = ["B","C","BB","CC","RU"]; 
    df_res = df_ra_pr.groupby(cols_2_group)[cols]\
                .agg({"B": [np.mean, np.std, np.min, np.max], 
                      "C": [np.mean, np.std, np.min, np.max], 
                      "BB":[np.mean, np.std, np.min, np.max],
                      "CC":[np.mean, np.std, np.min, np.max],
                      "RU":[np.mean, np.std, np.min, np.max]})
    df_res.columns = ["_".join(x) for x in df_res.columns.ravel()]
    df_res = df_res.reset_index()
    
    aggs = ["amin", "amax", "std", "mean"]
    tooltips = [("{}_{}".format(col, agg), "@{}_{}".format(col, agg)) 
                for (col, agg) in it.product(cols, aggs)]
    TOOLS[7] = HoverTool(tooltips = tooltips)
    
    new_cols = [col[1].split("@")[1] 
                for col in tooltips if col[1].split("_")[1] == "mean"]
    print('new_cols={}, df_res.cols={}'.format(new_cols, df_res.columns))
    
    x = list(map(tuple,list(df_res[cols_2_group].values)))
    px = figure(x_range=FactorRange(*x), 
                y_range=(0, df_res[new_cols].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
           
    data = dict(x = x, 
                B_mean=df_res.B_mean.tolist(), 
                C_mean=df_res.C_mean.tolist(), 
                BB_mean=df_res.BB_mean.tolist(), 
                CC_mean=df_res.CC_mean.tolist(), 
                RU_mean=df_res.RU_mean.tolist(), 
                B_std=df_res.B_std.tolist(), 
                C_std=df_res.C_std.tolist(), 
                BB_std=df_res.BB_std.tolist(), 
                CC_std=df_res.CC_std.tolist(), 
                RU_std=df_res.RU_std.tolist(),
                B_amin=df_res.B_amin.tolist(), 
                C_amin=df_res.C_amin.tolist(), 
                BB_amin=df_res.BB_amin.tolist(), 
                CC_amin=df_res.CC_amin.tolist(), 
                RU_amin=df_res.RU_amin.tolist(), 
                B_amax=df_res.B_amax.tolist(), 
                C_amax=df_res.C_amax.tolist(), 
                BB_amax=df_res.BB_amax.tolist(), 
                CC_amax=df_res.CC_amax.tolist(), 
                RU_amax=df_res.RU_amax.tolist()
                )

    print("data keys={}".format(data.keys()))
    source = ColumnDataSource(data = data)
    
    width= 0.2 #0.5
    # px.vbar(x='x', top=new_cols[4], width=0.9, source=source, color="#c9d9d3")
            
    # px.vbar(x='x', top=new_cols[0], width=0.9, source=source, color="#718dbf")
    
    px.vbar(x=dodge('x', -0.3+0*width, range=px.x_range), top=new_cols[0], 
                    width=width, source=source, legend_label=new_cols[0], 
                    color="#c9d9d3")
    px.vbar(x=dodge('x', -0.3+1*width, range=px.x_range), top=new_cols[1], 
                    width=width, source=source, legend_label=new_cols[1], 
                    color="#718dbf")
    px.vbar(x=dodge('x', -0.3+2*width, range=px.x_range), top=new_cols[2], 
                    width=width, source=source, legend_label=new_cols[2], 
                    color="#e84d60")
    px.vbar(x=dodge('x', -0.3+3*width, range=px.x_range), top=new_cols[3], 
                    width=width, source=source, legend_label=new_cols[3], 
                    color="#ddb7b1")
    px.vbar(x=dodge('x', -0.3+4*width, range=px.x_range), top=new_cols[4], 
                    width=width, source=source, legend_label=new_cols[4], 
                    color="#FFD700")
    
    title = "comparison Gamma_version ({},rate:{}, price={})".format(scenario, rate, price)
    px.title.text = title
    px.y_range.start = df_res.RU_mean.min() - 1
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "algo"
    px.yaxis.axis_label = "values"
    
    return px
    

def plot_gamma_version_RU(df_ra_pr, rate, price, scenario):
    
    cols_2_group = ["algo","gamma_version"]
    
    cols = ["BB","CC","RU"]; 
    df_res = df_ra_pr.groupby(cols_2_group)[cols]\
                .agg({"BB":[np.mean, np.std, np.min, np.max],
                      "CC":[np.mean, np.std, np.min, np.max],
                      "RU":[np.mean, np.std, np.min, np.max]})
    df_res.columns = ["_".join(x) for x in df_res.columns.ravel()]
    df_res = df_res.reset_index()
    
    aggs = ["amin", "amax", "std", "mean"]
    tooltips = [("{}_{}".format(col, agg), "@{}_{}".format(col, agg)) 
                for (col, agg) in it.product(cols, aggs)]
    TOOLS[7] = HoverTool(tooltips = tooltips)
    
    new_cols = [col[1].split("@")[1] 
                for col in tooltips if col[1].split("_")[1] == "mean"]
    print('new_cols={}, df_res.cols={}'.format(new_cols, df_res.columns))
    
    x = list(map(tuple,list(df_res[cols_2_group].values)))
    px = figure(x_range=FactorRange(*x), 
                y_range=(0, df_res[new_cols].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
           
    data = dict(x = x, 
                BB_mean=df_res.BB_mean.tolist(), 
                CC_mean=df_res.CC_mean.tolist(), 
                RU_mean=df_res.RU_mean.tolist(), 
                BB_std=df_res.BB_std.tolist(), 
                CC_std=df_res.CC_std.tolist(), 
                RU_std=df_res.RU_std.tolist(),
                BB_amin=df_res.BB_amin.tolist(), 
                CC_amin=df_res.CC_amin.tolist(), 
                RU_amin=df_res.RU_amin.tolist(), 
                BB_amax=df_res.BB_amax.tolist(), 
                CC_amax=df_res.CC_amax.tolist(), 
                RU_amax=df_res.RU_amax.tolist()
                )

    print("data keys={}".format(data.keys()))
    source = ColumnDataSource(data = data)
    
    width= 0.2 #0.5
    # px.vbar(x='x', top=new_cols[4], width=0.9, source=source, color="#c9d9d3")
            
    # px.vbar(x='x', top=new_cols[0], width=0.9, source=source, color="#718dbf")
    
    px.vbar(x=dodge('x', -0.3+0*width, range=px.x_range), top=new_cols[0], 
                    width=width, source=source, legend_label=new_cols[0], 
                    color="#ddb7b1")
    px.vbar(x=dodge('x', -0.3+1*width, range=px.x_range), top=new_cols[1], 
                    width=width, source=source, legend_label=new_cols[1], 
                    color="#e84d60")
    px.vbar(x=dodge('x', -0.3+2*width, range=px.x_range), top=new_cols[2], 
                    width=width, source=source, legend_label=new_cols[2], 
                    color="#FFD700")
    
    title = "comparison Gamma_version ({},rate:{}, price={})".format(scenario, rate, price)
    px.title.text = title
    px.y_range.start = df_res.RU_mean.min() - 1
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "algo"
    px.yaxis.axis_label = "values"
    
    return px
    

def plot_comparaison_gamma_version_RU(df_B_C_BB_CC_RU_M):
    rates = df_B_C_BB_CC_RU_M.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df_B_C_BB_CC_RU_M.prices.unique().tolist()
    scenarios = df_B_C_BB_CC_RU_M.scenario.unique().tolist()
    
    dico_pxs = dict()
    for rate, price, scenario in it.product(rates, prices, scenarios):
        mask_ra_pr = ((df_B_C_BB_CC_RU_M.rate == rate) \
                      | (df_B_C_BB_CC_RU_M.rate == 0)) \
                        & (df_B_C_BB_CC_RU_M.prices == price) \
                        & (df_B_C_BB_CC_RU_M.scenario == scenario) 
        df_ra_pr = df_B_C_BB_CC_RU_M[mask_ra_pr].copy()
        
        pxs_pr_ra_sc = plot_gamma_version_RU(df_ra_pr, rate, price, scenario)
        pxs_pr_ra_sc.legend.click_policy="hide"
        
        if (price, rate, scenario) not in dico_pxs.keys():
            dico_pxs[(price, rate, scenario)] \
                = [pxs_pr_ra_sc]
        else:
            dico_pxs[(price, rate, scenario)].append(pxs_pr_ra_sc)
        
    rows_RU_C_B_CC_BB = list()
    for key, pxs_pr_ra_sc in dico_pxs.items():
        col_px_sts = column(pxs_pr_ra_sc)
        rows_RU_C_B_CC_BB.append(col_px_sts)
    rows_RU_C_B_CC_BB=column(children=rows_RU_C_B_CC_BB, 
                             sizing_mode='stretch_both')
    return rows_RU_C_B_CC_BB
# _____________________________________________________________________________ 
#               
#                   plot RU 4 various gamma_version 4 each scenario
#                                       --> fin
# _____________________________________________________________________________ 
# _____________________________________________________________________________ 
#               
#                   plot RU 4 various gamma_version 4 all scenarios 
#                            --> debut
# _____________________________________________________________________________ 
def plot_gamma_version_all_scenarios(df_ra_pr, rate, price):
    
    cols_2_group = ["algo","gamma_version"]
    
    cols = ["B","C","BB","CC","RU"]; 
    df_res = df_ra_pr.groupby(cols_2_group)[cols]\
                .agg({"B": [np.mean, np.std, np.min, np.max], 
                      "C": [np.mean, np.std, np.min, np.max], 
                      "BB":[np.mean, np.std, np.min, np.max],
                      "CC":[np.mean, np.std, np.min, np.max],
                      "RU":[np.mean, np.std, np.min, np.max]})
    df_res.columns = ["_".join(x) for x in df_res.columns.ravel()]
    df_res = df_res.reset_index()
    
    aggs = ["amin", "amax", "std", "mean"]
    tooltips = [("{}_{}".format(col, agg), "@{}_{}".format(col, agg)) 
                for (col, agg) in it.product(cols, aggs)]
    TOOLS[7] = HoverTool(tooltips = tooltips)
    
    new_cols = [col[1].split("@")[1] 
                for col in tooltips if col[1].split("_")[1] == "mean"]
    print('new_cols={}, df_res.cols={}'.format(new_cols, df_res.columns))
    
    x = list(map(tuple,list(df_res[cols_2_group].values)))
    px = figure(x_range=FactorRange(*x), 
                y_range=(0, df_res[new_cols].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
           
    data = dict(x = x, 
                B_mean=df_res.B_mean.tolist(), 
                C_mean=df_res.C_mean.tolist(), 
                BB_mean=df_res.BB_mean.tolist(), 
                CC_mean=df_res.CC_mean.tolist(), 
                RU_mean=df_res.RU_mean.tolist(), 
                B_std=df_res.B_std.tolist(), 
                C_std=df_res.C_std.tolist(), 
                BB_std=df_res.BB_std.tolist(), 
                CC_std=df_res.CC_std.tolist(), 
                RU_std=df_res.RU_std.tolist(),
                B_amin=df_res.B_amin.tolist(), 
                C_amin=df_res.C_amin.tolist(), 
                BB_amin=df_res.BB_amin.tolist(), 
                CC_amin=df_res.CC_amin.tolist(), 
                RU_amin=df_res.RU_amin.tolist(), 
                B_amax=df_res.B_amax.tolist(), 
                C_amax=df_res.C_amax.tolist(), 
                BB_amax=df_res.BB_amax.tolist(), 
                CC_amax=df_res.CC_amax.tolist(), 
                RU_amax=df_res.RU_amax.tolist()
                )

    print("data keys={}".format(data.keys()))
    source = ColumnDataSource(data = data)
    
    width= 0.2 #0.5
    # px.vbar(x='x', top=new_cols[4], width=0.9, source=source, color="#c9d9d3")
            
    # px.vbar(x='x', top=new_cols[0], width=0.9, source=source, color="#718dbf")
    
    px.vbar(x=dodge('x', -0.3+0*width, range=px.x_range), top=new_cols[0], 
                    width=width, source=source, legend_label=new_cols[0], 
                    color="#c9d9d3")
    px.vbar(x=dodge('x', -0.3+1*width, range=px.x_range), top=new_cols[1], 
                    width=width, source=source, legend_label=new_cols[1], 
                    color="#718dbf")
    px.vbar(x=dodge('x', -0.3+2*width, range=px.x_range), top=new_cols[2], 
                    width=width, source=source, legend_label=new_cols[2], 
                    color="#e84d60")
    px.vbar(x=dodge('x', -0.3+3*width, range=px.x_range), top=new_cols[3], 
                    width=width, source=source, legend_label=new_cols[3], 
                    color="#ddb7b1")
    px.vbar(x=dodge('x', -0.3+4*width, range=px.x_range), top=new_cols[4], 
                    width=width, source=source, legend_label=new_cols[4], 
                    color="#FFD700")
    
    title = "comparison Gamma_version (rate:{}, price={})".format(rate, price)
    px.title.text = title
    px.y_range.start = df_res.RU_mean.min() - 1
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "algo"
    px.yaxis.axis_label = "values"
    
    return px
    

def plot_comparaison_gamma_version_all_scenarios(df_B_C_BB_CC_RU_M):
    rates = df_B_C_BB_CC_RU_M.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df_B_C_BB_CC_RU_M.prices.unique().tolist()
    
    dico_pxs = dict()
    for rate, price in it.product(rates, prices):
        mask_ra_pr = ((df_B_C_BB_CC_RU_M.rate == rate) \
                      | (df_B_C_BB_CC_RU_M.rate == 0)) \
                        & (df_B_C_BB_CC_RU_M.prices == price)
        df_ra_pr = df_B_C_BB_CC_RU_M[mask_ra_pr].copy()
        
        pxs_pr_ra = plot_gamma_version_all_scenarios(df_ra_pr, rate, price)
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
#                   plot RU 4 various gamma_version 4 all scenarios 
#                            --> fin
# _____________________________________________________________________________ 

# _____________________________________________________________________________
#
#                   distribution by states for periods ---> debut
# _____________________________________________________________________________
def plot_distribution(df_al_pr_ra_sc, algo, rate, price, scenario, gamma_version):
    """
    plot the bar plot with key is (t, stateX) (X={1,2,3})
    """
    cols = ["t", "state_i"]
    df_state = df_al_pr_ra_sc.groupby(cols)[["state_i"]].count()
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
               title="number of players, ({}, {}, {}, rate={}, price={})".format(
                  algo, scenario, gamma_version, rate, price),
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
    
def OLD_plot_distribution_by_states_4_periods(df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T):
    """
    plot the distribution by state for each period
    plot is the bar plot with key is (t, stateX) (X={1,2,3})
    
    """
    df_M_T = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T.copy()
    
    rates = df_M_T.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df_M_T.prices.unique().tolist()
    algos = df_M_T.algo.unique().tolist()
    scenarios = df_M_T.scenario.unique().tolist()
    gamma_versions = df_M_T.gamma_version.unique().tolist()
    
    dico_pxs = dict()
    for algo, price, rate, scenario, gamma_version in it.product(algos, prices, rates, 
                                                  scenarios, gamma_versions):
        mask_al_pr_ra_sc = ((df_M_T.rate == str(rate)) 
                                 | (df_M_T.rate == 0)) \
                            & (df_M_T.prices == price) \
                            & (df_M_T.algo == algo) \
                            & (df_M_T.scenario == scenario) \
                            & (df_M_T.gamma_version == gamma_version)
        df_al_pr_ra_sc = df_M_T[mask_al_pr_ra_sc].copy()
        
        pxs_al_pr_ra_scen = plot_distribution(df_al_pr_ra_sc, algo, rate, 
                                              price, scenario, gamma_version)
        
        if (algo, price, rate, scenario, gamma_version) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate, scenario, gamma_version)] \
                = [pxs_al_pr_ra_scen]
        else:
            dico_pxs[(algo, price, rate, scenario, gamma_version)].append(pxs_al_pr_ra_scen)
        
    rows_dists_ts = list()
    for key, pxs_al_pr_ra_scen in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra_scen)
        rows_dists_ts.append(col_px_sts)
    rows_dists_ts = column(children=rows_dists_ts, 
                           sizing_mode='stretch_both')
    return rows_dists_ts

def plot_distribution_by_states_4_periods(
        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
        dico_SelectGammaVersion
        ):
    """
    plot the distribution by state for each period
    plot is the bar plot with key is (t, stateX) (X={1,2,3})
    
    """
    df_M_T = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T.copy()
    
    rates = df_M_T.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df_M_T.prices.unique().tolist()
    algos = df_M_T.algo.unique().tolist()
    scenarios = df_M_T.scenario.unique().tolist()
    gamma_version_root = "".join(list(df_M_T.gamma_version.unique()[0])[:-1])
    
    dico_pxs = dict()
    for algo, price, rate, scenario in it.product(algos, prices, rates, 
                                                  scenarios):
        gamma_versions = dico_SelectGammaVersion[algo]
        for gamma_version_number in gamma_versions:
            gamma_version = gamma_version_root+str(gamma_version_number)
            mask_al_pr_ra_sc = ((df_M_T.rate == str(rate)) 
                                     | (df_M_T.rate == 0)) \
                                & (df_M_T.prices == price) \
                                & (df_M_T.algo == algo) \
                                & (df_M_T.scenario == scenario) \
                                & (df_M_T.gamma_version == gamma_version)
            df_al_pr_ra_sc = df_M_T[mask_al_pr_ra_sc].copy()
            
            pxs_al_pr_ra_scen = plot_distribution(df_al_pr_ra_sc, algo, rate, 
                                                  price, scenario, gamma_version)
            
            if (algo, price, rate, scenario, gamma_version) not in dico_pxs.keys():
                dico_pxs[(algo, price, rate, scenario, gamma_version)] \
                    = [pxs_al_pr_ra_scen]
            else:
                dico_pxs[(algo, price, rate, scenario, gamma_version)].append(pxs_al_pr_ra_scen)
        
    rows_dists_ts = list()
    for key, pxs_al_pr_ra_scen in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra_scen)
        rows_dists_ts.append(col_px_sts)
    rows_dists_ts = column(children=rows_dists_ts, 
                           sizing_mode='stretch_both')
    return rows_dists_ts

# _____________________________________________________________________________
#
#                   distribution by states for periods ---> fin
# _____________________________________________________________________________


# _____________________________________________________________________________
#
#              evolution prices B, C, BB, CC, RU for periods ---> debut
# _____________________________________________________________________________
def OLD_plot_evolution_prices_for_time(df_al_pr_ra_sc_gam, algo, rate, 
                                   price, scenario, gamma_version):
    
    cols = ["t", "B", "C", "BB", "CC", "RU"]
    
    df_res_t = df_al_pr_ra_sc_gam.groupby(cols[0])[cols[1:]]\
                .agg({cols[1]:[np.mean, np.std, np.min, np.max], 
                      cols[2]:[np.mean, np.std, np.min, np.max], 
                      cols[3]:[np.mean, np.std, np.min, np.max], 
                      cols[4]:[np.mean, np.std, np.min, np.max], 
                      cols[5]:[np.mean, np.std, np.min, np.max]})
    
    df_res_t.columns = ["_".join(x) for x in df_res_t.columns.ravel()]
    df_res_t = df_res_t.reset_index()
    
    df_res_t.t = df_res_t.t.astype("str")
    
    aggs = ["amin", "amax", "std", "mean"]
    tooltips = [("{}_{}".format(col, agg), "@{}_{}".format(col, agg)) 
                for (col, agg) in it.product(cols[1:], aggs)]
    TOOLS[7] = HoverTool(tooltips = tooltips)
    
    new_cols = [col[1].split("@")[1] 
                for col in tooltips if col[1].split("_")[1] == "mean"]
    print('new_cols={}, df_res_t.cols={}'.format(new_cols, df_res_t.columns))
    
    x = list(map(tuple,list(df_res_t.t.values)))
    px = figure(x_range=df_res_t.t.values.tolist(), 
                y_range=(0, df_res_t[new_cols].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
           
    data = dict(x = x, 
                B_mean=df_res_t.B_mean.tolist(), 
                C_mean=df_res_t.C_mean.tolist(), 
                BB_mean=df_res_t.BB_mean.tolist(), 
                CC_mean=df_res_t.CC_mean.tolist(), 
                RU_mean=df_res_t.RU_mean.tolist(), 
                B_std=df_res_t.B_std.tolist(), 
                C_std=df_res_t.C_std.tolist(), 
                BB_std=df_res_t.BB_std.tolist(), 
                CC_std=df_res_t.CC_std.tolist(), 
                RU_std=df_res_t.RU_std.tolist(),
                B_amin=df_res_t.B_amin.tolist(), 
                C_amin=df_res_t.C_amin.tolist(), 
                BB_amin=df_res_t.BB_amin.tolist(), 
                CC_amin=df_res_t.CC_amin.tolist(), 
                RU_amin=df_res_t.RU_amin.tolist(), 
                B_amax=df_res_t.B_amax.tolist(), 
                C_amax=df_res_t.C_amax.tolist(), 
                BB_amax=df_res_t.BB_amax.tolist(), 
                CC_amax=df_res_t.CC_amax.tolist(), 
                RU_amax=df_res_t.RU_amax.tolist()
                )

    print("data keys={}".format(data.keys()))
    source = ColumnDataSource(data = data)
    
    width= 0.1 #0.5
    # px.vbar(x='x', top=new_cols[4], width=0.9, source=source, color="#c9d9d3")
            
    # px.vbar(x='x', top=new_cols[0], width=0.9, source=source, color="#718dbf")
    
    px.vbar(x=dodge('x', -0.3+0*width, range=px.x_range), top=new_cols[0], 
                    width=width, source=source, legend_label=new_cols[0], 
                    color="#c9d9d3")
    px.vbar(x=dodge('x', -0.3+1*width, range=px.x_range), top=new_cols[1], 
                    width=width, source=source, legend_label=new_cols[1], 
                    color="#718dbf")
    px.vbar(x=dodge('x', -0.3+2*width, range=px.x_range), top=new_cols[2], 
                    width=width, source=source, legend_label=new_cols[2], 
                    color="#e84d60")
    px.vbar(x=dodge('x', -0.3+3*width, range=px.x_range), top=new_cols[3], 
                    width=width, source=source, legend_label=new_cols[3], 
                    color="#ddb7b1")
    px.vbar(x=dodge('x', -0.3+4*width, range=px.x_range), top=new_cols[4], 
                    width=width, source=source, legend_label=new_cols[4], 
                    color="#FFD700")
    
    title = "gain evolution over time ({}, {}, rate:{}, price={}, gamma_version={})".format(
                algo, scenario, rate, price, gamma_version)
    px.title.text = title
    px.y_range.start = df_res_t.RU_mean.min() - 1
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "t_periods"
    px.yaxis.axis_label = "values"
    
    return px


def plot_evolution_prices_for_time(df_al_pr_ra_sc_gam, algo, rate, 
                                   price, scenario, gamma_version):
    
    cols = ["t", "B", "C", "BB", "CC", "RU", "Cicum", "Picum"]
    
    df_res_t = df_al_pr_ra_sc_gam.groupby(cols[0])[cols[1:]]\
                .agg({cols[1]:[np.mean, np.std, np.min, np.max], 
                      cols[2]:[np.mean, np.std, np.min, np.max], 
                      cols[3]:[np.mean, np.std, np.min, np.max], 
                      cols[4]:[np.mean, np.std, np.min, np.max], 
                      cols[5]:[np.mean, np.std, np.min, np.max], 
                      cols[6]:[np.mean, np.std, np.min, np.max], 
                      cols[7]:[np.mean, np.std, np.min, np.max]
                      })
    
    df_res_t.columns = ["_".join(x) for x in df_res_t.columns.ravel()]
    df_res_t = df_res_t.reset_index()
    
    df_res_t.t = df_res_t.t.astype("str")
    
    aggs = ["amin", "amax", "std", "mean"]
    tooltips = [("{}_{}".format(col, agg), "@{}_{}".format(col, agg)) 
                for (col, agg) in it.product(cols[1:], aggs)]
    TOOLS[7] = HoverTool(tooltips = tooltips)
    
    new_cols = [col[1].split("@")[1] 
                for col in tooltips if col[1].split("_")[1] == "mean"]
    print('new_cols={}, df_res_t.cols={}'.format(new_cols, df_res_t.columns))
    
    x = list(map(tuple,list(df_res_t.t.values)))
    px = figure(x_range=df_res_t.t.values.tolist(), 
                y_range=(0, df_res_t[new_cols].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
           
    data = dict(x = df_res_t.t.values.tolist(), 
                B_mean=df_res_t.B_mean.tolist(), 
                C_mean=df_res_t.C_mean.tolist(), 
                BB_mean=df_res_t.BB_mean.tolist(), 
                CC_mean=df_res_t.CC_mean.tolist(), 
                RU_mean=df_res_t.RU_mean.tolist(), 
                
                Picum_mean=df_res_t.Picum_mean.tolist(), 
                Cicum_mean=df_res_t.Cicum_mean.tolist(), 
                
                B_std=df_res_t.B_std.tolist(), 
                C_std=df_res_t.C_std.tolist(), 
                BB_std=df_res_t.BB_std.tolist(), 
                CC_std=df_res_t.CC_std.tolist(), 
                RU_std=df_res_t.RU_std.tolist(),
                
                Picum_std=df_res_t.Picum_std.tolist(), 
                Cicum_std=df_res_t.Cicum_std.tolist(), 
                
                B_amin=df_res_t.B_amin.tolist(), 
                C_amin=df_res_t.C_amin.tolist(), 
                BB_amin=df_res_t.BB_amin.tolist(), 
                CC_amin=df_res_t.CC_amin.tolist(), 
                RU_amin=df_res_t.RU_amin.tolist(), 
                
                Picum_amin=df_res_t.Picum_amin.tolist(), 
                Cicum_amin=df_res_t.Cicum_amin.tolist(), 
                
                B_amax=df_res_t.B_amax.tolist(), 
                C_amax=df_res_t.C_amax.tolist(), 
                BB_amax=df_res_t.BB_amax.tolist(), 
                CC_amax=df_res_t.CC_amax.tolist(), 
                RU_amax=df_res_t.RU_amax.tolist(), 
                
                Picum_amax=df_res_t.Picum_amax.tolist(), 
                Cicum_amax=df_res_t.Cicum_amax.tolist() 
                )

    print("data keys={}".format(data.keys()))
    source = ColumnDataSource(data = data)
    
    width= 0.1 #0.5
    # px.vbar(x='x', top=new_cols[4], width=0.9, source=source, color="#c9d9d3")
            
    # px.vbar(x='x', top=new_cols[0], width=0.9, source=source, color="#718dbf")
    
    px.vbar(x=dodge('x', -0.3+0*width, range=px.x_range), top=new_cols[0], 
                    width=width, source=source, legend_label=new_cols[0], 
                    color="#c9d9d3")
    px.vbar(x=dodge('x', -0.3+1*width, range=px.x_range), top=new_cols[1], 
                    width=width, source=source, legend_label=new_cols[1], 
                    color="#718dbf")
    px.vbar(x=dodge('x', -0.3+2*width, range=px.x_range), top=new_cols[2], 
                    width=width, source=source, legend_label=new_cols[2], 
                    color="#e84d60")
    px.vbar(x=dodge('x', -0.3+3*width, range=px.x_range), top=new_cols[3], 
                    width=width, source=source, legend_label=new_cols[3], 
                    color="#ddb7b1")
    px.vbar(x=dodge('x', -0.3+4*width, range=px.x_range), top=new_cols[4], 
                    width=width, source=source, legend_label=new_cols[4], 
                    color="#FFD700")
    
    px.vbar(x=dodge('x', -0.3+5*width, range=px.x_range), top=new_cols[5], 
                    width=width, source=source, legend_label=new_cols[5], 
                    color="#bcbd22")
    px.vbar(x=dodge('x', -0.3+6*width, range=px.x_range), top=new_cols[6], 
                    width=width, source=source, legend_label=new_cols[6], 
                    color="#17becf")
    
    
    # px.vbar(x=dodge('x', -0.3+0*width, range=px.x_range), top=new_cols[0], 
    #                 width=width, source=source, legend_label=new_cols[0], 
    #                 color="#c9d9d3")
    # px.vbar(x=dodge('x', -0.3+1*width, range=px.x_range), top=new_cols[1], 
    #                 width=width, source=source, legend_label=new_cols[1], 
    #                 color="#718dbf")
    # px.vbar(x=dodge('x', -0.3+2*width, range=px.x_range), top=new_cols[2], 
    #                 width=width, source=source, legend_label=new_cols[2], 
    #                 color="#e84d60")
    # px.vbar(x=dodge('x', -0.3+3*width, range=px.x_range), top=new_cols[3], 
    #                 width=width, source=source, legend_label=new_cols[3], 
    #                 color="#ddb7b1")
    # px.vbar(x=dodge('x', -0.3+4*width, range=px.x_range), top=new_cols[4], 
    #                 width=width, source=source, legend_label=new_cols[4], 
    #                 color="#FFD700")
    
    title = "gain evolution over time ({}, {}, rate:{}, price={}, gamma_version={})".format(
                algo, scenario, rate, price, gamma_version)
    px.title.text = title
    px.y_range.start = df_res_t.RU_mean.min() - 1
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "t_periods"
    px.yaxis.axis_label = "values"
    
    return px
    
def OLD_plot_evolution_RU_C_B_CC_BB_over_time(
                    df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
                    algos=["LRI1"], 
                    gamma_versions=["gammaV1"], 
                    scenarios=["scenario0"]
                    ):
    
    df = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T.copy()
    
    rates = df.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df.prices.unique().tolist()
    scenarios = df.scenario.unique().tolist() if len(scenarios) == 0 \
                                                else scenarios
    gamma_versions = df.gamma_version.unique().tolist() \
                        if len(gamma_versions) == 0 \
                        else gamma_versions
    algos = df.algo.unique().tolist() if len(algos) == 0 else algos 
    
    dico_pxs = dict()
    for algo, price, rate, scenario, gamma_version \
        in it.product(algos, prices, rates, scenarios, gamma_versions):
        mask_al_pr_ra_sc_gam = ((df.rate == str(rate)) | (df.rate == 0)) \
                                & (df.prices == price) \
                                & (df.algo == algo) \
                                & (df.scenario == scenario) \
                                & (df.gamma_version == gamma_version)
        df_al_pr_ra_sc_gam = df[mask_al_pr_ra_sc_gam].copy()
        
        print("{}, {}, {}, df_al_pr_ra_sc_gam={}".format(algo, 
                scenario, gamma_version, df_al_pr_ra_sc_gam.shape ))
        pxs_al_pr_ra_sc_gam = plot_evolution_prices_for_time(
                                df_al_pr_ra_sc_gam, algo, rate, 
                                price, scenario, gamma_version)
        pxs_al_pr_ra_sc_gam.legend.click_policy="hide"
        
        if (algo, price, rate, scenario, gamma_version) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate, scenario, gamma_version)] \
                = [pxs_al_pr_ra_sc_gam]
        else:
            dico_pxs[(algo, price, rate, scenario, gamma_version)]\
                .append(pxs_al_pr_ra_sc_gam)
        
    rows_evol_RU_C_B_CC_BB = list()
    for key, pxs_al_pr_ra_sc_gam in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra_sc_gam)
        rows_evol_RU_C_B_CC_BB.append(col_px_sts)
    rows_evol_RU_C_B_CC_BB = column(children=rows_evol_RU_C_B_CC_BB, 
                                    sizing_mode='stretch_both')
    return rows_evol_RU_C_B_CC_BB
        
        
def plot_evolution_RU_C_B_CC_BB_over_time(
            df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
            algos=["LRI1"], 
            dico_SelectGammaVersion={"DETERMINIST": [1],"LRI1": [1],"LRI2": [0]}):
    
    df = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T.copy()
    
    rates = df.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df.prices.unique().tolist()
    scenarios = df.scenario.unique().tolist()
    gamma_version_root = "".join(list(df.gamma_version.unique()[0])[:-1])
    
    dico_pxs = dict()
    for algo, price, rate, scenario in it.product(algos, prices, 
                                                  rates, scenarios):
        gamma_versions = dico_SelectGammaVersion[algo]
        for gamma_version_number in gamma_versions:
            gamma_version = gamma_version_root+str(gamma_version_number)
            mask_al_pr_ra_sc_gam = ((df.rate == str(rate)) | (df.rate == 0)) \
                                    & (df.prices == price) \
                                    & (df.algo == algo) \
                                    & (df.scenario == scenario) \
                                    & (df.gamma_version == gamma_version)
            df_al_pr_ra_sc_gam = df[mask_al_pr_ra_sc_gam].copy()
            
            print("{}, {}, {}, df_al_pr_ra_sc_gam={}".format(algo, 
                    scenario, gamma_version, df_al_pr_ra_sc_gam.shape ))
            pxs_al_pr_ra_sc_gam = plot_evolution_prices_for_time(
                                    df_al_pr_ra_sc_gam, algo, rate, 
                                    price, scenario, gamma_version)
            pxs_al_pr_ra_sc_gam.legend.click_policy="hide"
            
            if (algo, price, rate, scenario, gamma_version) not in dico_pxs.keys():
                dico_pxs[(algo, price, rate, scenario, gamma_version)] \
                    = [pxs_al_pr_ra_sc_gam]
            else:
                dico_pxs[(algo, price, rate, scenario, gamma_version)]\
                    .append(pxs_al_pr_ra_sc_gam)
        
    rows_evol_RU_C_B_CC_BB = list()
    for key, pxs_al_pr_ra_sc_gam in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra_sc_gam)
        rows_evol_RU_C_B_CC_BB.append(col_px_sts)
    rows_evol_RU_C_B_CC_BB = column(children=rows_evol_RU_C_B_CC_BB, 
                                    sizing_mode='stretch_both')
    return rows_evol_RU_C_B_CC_BB
# _____________________________________________________________________________
#
#              evolution prices B, C, BB, CC, RU for periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#              evolution stocks by player over periods ---> debut
# _____________________________________________________________________________
def plot_evolution_stocks_by_players_for_time(df_al_pr_ra_sc_gam, algo, rate, 
                                              price, scenario, gamma_version):
    
    
    cols = ["pl_i", "t", "state_i", "mode_i", "Si"]
    df_pl_t = df_al_pr_ra_sc_gam[cols]
    
    x = list( map(tuple, df_pl_t[cols[:2]].values) )
    
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("Si", "@Si"),
                            ("mode_i", "@mode_i"),
                            ("state_i", "@state_i"),
                            ]
                        )
    
    px= figure(x_range=FactorRange(*x), 
               plot_height=350, plot_width = int(WIDTH*MULT_WIDTH),
               toolbar_location="above", tools=TOOLS)
    

    data = dict(x=x, Si=df_pl_t.Si.tolist(), 
                state_i=df_pl_t.state_i.tolist(), 
                mode_i=df_pl_t.mode_i.tolist() )
    
    source = ColumnDataSource(data=data)
    
    m_players = len(df_pl_t.pl_i.unique())
    colors = [color for i, color in enumerate(Viridis256) 
                      if i% int(256/m_players) == 0 ]
    px.vbar(x='x', top='Si', width=0.9, source=source, 
            fill_color=factor_cmap('x', 
                                   #palette=Category20[20], 
                                   palette=colors, 
                                   factors=list(df_pl_t.pl_i.unique()), 
                                   start=0, end=1))
    
    title = "evolution of stock by players over time ({}, {}, {}, rate={}, price={})".format(
            algo, scenario, gamma_version, rate, price)
    px.title.text = title
    px.y_range.start = df_pl_t.Si.min() - 1
    px.x_range.range_padding = 0.1#WIDTH
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.major_label_orientation = 1
    px.xaxis.axis_label = "players"
    px.yaxis.axis_label = "values"
    
    return px
    

def plot_evolution_Si_by_players_over_time(
            df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
            algos=["LRI1"], 
            dico_SelectGammaVersion={"DETERMINIST": [1],"LRI1": [1],"LRI2": [0]}):
    
    df = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T.copy()
    df["pl_i"] = df.pl_i.astype(str)
    df["t"] = df.t.astype(str)
    
    rates = df.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df.prices.unique().tolist()
    scenarios = df.scenario.unique().tolist()
    gamma_version_root = "".join(list(df.gamma_version.unique()[0])[:-1])
    
    dico_pxs = dict()
    for algo, price, rate, scenario in it.product(algos, prices, 
                                                  rates, scenarios):
        gamma_versions = dico_SelectGammaVersion[algo]
        for gamma_version_number in gamma_versions:
            gamma_version = gamma_version_root+str(gamma_version_number)
            mask_al_pr_ra_sc_gam = ((df.rate == str(rate)) | (df.rate == 0)) \
                                    & (df.prices == price) \
                                    & (df.algo == algo) \
                                    & (df.scenario == scenario) \
                                    & (df.gamma_version == gamma_version)
            df_al_pr_ra_sc_gam = df[mask_al_pr_ra_sc_gam].copy()
            
            print("{}, {}, {}, df_al_pr_ra_sc_gam={}".format(algo, 
                    scenario, gamma_version, df_al_pr_ra_sc_gam.shape ))
            pxs_al_pr_ra_sc_gam = plot_evolution_stocks_by_players_for_time(
                                    df_al_pr_ra_sc_gam, algo, rate, 
                                    price, scenario, gamma_version)
            #pxs_al_pr_ra_sc_gam.legend.click_policy="hide"
            
            if (algo, price, rate, scenario, gamma_version) not in dico_pxs.keys():
                dico_pxs[(algo, price, rate, scenario, gamma_version)] \
                    = [pxs_al_pr_ra_sc_gam]
            else:
                dico_pxs[(algo, price, rate, scenario, gamma_version)]\
                    .append(pxs_al_pr_ra_sc_gam)
        
    rows_evol_Si = list()
    for key, pxs_al_pr_ra_sc_gam in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra_sc_gam)
        rows_evol_Si.append(col_px_sts)
    rows_evol_Si = column(children=rows_evol_Si, sizing_mode='stretch_both')
    
    return rows_evol_Si
    
# _____________________________________________________________________________
#
#              evolution stocks by player over periods ---> fin
# _____________________________________________________________________________


# _____________________________________________________________________________
#
#      evolution of the number of player by situations over periods ---> debut
# _____________________________________________________________________________
def OLD_plot_evolution_DIFFERENCE_players_by_situation_for_time(df_al_pr_ra_sc_gam, 
                                                 algo, rate, 
                                                 price, scenario, 
                                                 gamma_version):
    """
    plot the bar plot with key is (t, setX) (X={A,B1,B2,C})
    identify the players that are changed the situation over the time
    """
    df_al_pr_ra_sc_gam["t"] = df_al_pr_ra_sc_gam["t"].astype(str)
    setX = df_al_pr_ra_sc_gam.set.unique().tolist()
    t_periods = df_al_pr_ra_sc_gam.t.unique().tolist()
    
    df = df_al_pr_ra_sc_gam.copy()
    
    data = {"t":t_periods}
    for setx in setX:
        players_t_minus_1 = list()
        for t in t_periods:
            mask = (df.t == t) & (df.set == setx); 
            df_t_setx = df[mask].copy()
            players_t = df_t_setx.pl_i.unique().tolist()
            diff_players_t_t_minus_1 = list()
            if t == "0":
                diff_players_t_t_minus_1 = []
                diff_players_t_t_minus_1\
                    .insert(0,"nb="+str(len(diff_players_t_t_minus_1)))
                # players_t_minus_1 = players_t
            else:
                diff_players_t_t_minus_1 \
                    = set(players_t_minus_1).union(set(players_t)) - set(players_t)
                diff_players_t_t_minus_1 = list(diff_players_t_t_minus_1)
                diff_players_t_t_minus_1\
                    .insert(0,"nb="+str(len(diff_players_t_t_minus_1)))
                # players_t_minus_1 = players_t
            if setx not in data:
                data[setx] = [diff_players_t_t_minus_1]
            else:
                data[setx].append(diff_players_t_t_minus_1)
                
            players_t_minus_1 = players_t
    
    t_setX = list(it.product(t_periods, setX)) 
    
    tooltips = [ (setx,"@"+setx) for setx in setX]
    tooltips.append(("t", "@t"))
    TOOLS[7] = HoverTool(tooltips= tooltips)
    px= figure(x_range=FactorRange(*t_setX), 
               plot_height=350, plot_width = int(WIDTH*MULT_WIDTH),
               title="number of players, ({}, {}, {}, rate={}, price={})".format(
                  algo, scenario, gamma_version, rate, price),
                toolbar_location="above", tools=TOOLS)
    
    source = ColumnDataSource(data=data)
    
    width= 0.1 #0.5
    # px.vbar(x='x', top=new_cols[4], width=0.9, source=source, color="#c9d9d3")
            
    # px.vbar(x='x', top=new_cols[0], width=0.9, source=source, color="#718dbf")
    
    px.vbar(x='t', top=setX[0], 
            width=width, source=source, legend_label=setX[0], 
            color="#c9d9d3")
    px.vbar(x='t', top=setX[1], 
            width=width, source=source, legend_label=setX[1], 
            color="#718dbf")
    px.vbar(x='t', top=setX[2], 
            width=width, source=source, legend_label=setX[2], 
            color="#e84d60")
    px.vbar(x='t', top=setX[3], 
            width=width, source=source, legend_label=setX[3], 
            color="#ddb7b1")
    
    px.y_range.start = 0
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    
    return px         
                
def plot_evolution_DIFFERENCE_players_by_situation_for_time(df_al_pr_ra_sc_gam, 
                                                 algo, rate, 
                                                 price, scenario, 
                                                 gamma_version):
    """
    plot the bar plot with key is (t, setX) (X={A,B1,B2,C})
    identify the players that are changed the situation over the time
    """
    df_al_pr_ra_sc_gam["t"] = df_al_pr_ra_sc_gam["t"].astype(str)
    setX = df_al_pr_ra_sc_gam.set.unique().tolist()
    t_periods = df_al_pr_ra_sc_gam.t.unique().tolist()
    
    df = df_al_pr_ra_sc_gam.copy()
    
    data = {"t":t_periods}
    for setx in setX:
        players_t_minus_1 = list()
        for t in t_periods:
            mask = (df.t == t) & (df.set == setx); 
            df_t_setx = df[mask].copy()
            players_t = df_t_setx.pl_i.unique().tolist()
            diff_players_t_t_minus_1 = list()
            if t == "0":
                diff_players_t_t_minus_1 = []
            else:
                diff_players_t_t_minus_1 \
                    = set(players_t_minus_1).union(set(players_t)) - set(players_t)
                diff_players_t_t_minus_1 = list(diff_players_t_t_minus_1)

            if setx not in data:
                data[setx] = [len(diff_players_t_t_minus_1)]
                data[setx+"Players"] = [diff_players_t_t_minus_1]
            else:
                data[setx].append(len(diff_players_t_t_minus_1))
                data[setx+"Players"].append(diff_players_t_t_minus_1)
                
            players_t_minus_1 = players_t
    
    t_setX = list(it.product(t_periods, setX)) 
    
    tooltips = list()
    for setx in setX:
        tooltips.append((setx,"@"+setx))
        tooltips.append((setx+"Players","@"+setx+"Players"))
    tooltips.append(("t", "@t"))
    TOOLS[7] = HoverTool(tooltips= tooltips)
    px= figure(x_range=FactorRange(*t_setX), 
               plot_height=350, plot_width = int(WIDTH*MULT_WIDTH),
               title="number of players, ({}, {}, {}, rate={}, price={})".format(
                  algo, scenario, gamma_version, rate, price),
                toolbar_location="above", tools=TOOLS)
    
    source = ColumnDataSource(data=data)
    
    width= 1.0 #0.1 #0.5
    # px.vbar(x='x', top=new_cols[4], width=0.9, source=source, color="#c9d9d3")
            
    colors = ["#c9d9d3", "#718dbf", "#e84d60", "#ddb7b1"]
    for i, setx in enumerate(setX):
        # px.vbar(x='t', top=setX[i], 
        #     width=width, source=source, legend_label=setX[i], 
        #     color=colors[i])
        px.vbar(x=dodge('t', i*width, range=px.x_range), top=setX[i], 
                width=width, source=source, legend_label=setX[i], 
                color=colors[i])
    
    px.y_range.start = 0
    #px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    
    return px         
                

def plot_evolution_players_by_situation_for_time(df_al_pr_ra_sc_gam, 
                                                 algo, rate, 
                                                 price, scenario, 
                                                 gamma_version):
    """
    plot the bar plot with key is (t, setX) (X={A,B1,B2,C})
    """
    cols = ["t", "set"]
    df_set = df_al_pr_ra_sc_gam.groupby(cols)[["set"]].count()
    df_set.rename(columns={"set":"nb_players"}, inplace=True)
    df_set = df_set.reset_index()
    df_set["t"] = df_set["t"].astype(str)
    
    x = list(map(tuple,list(df_set[cols].values)))
    nb_players = list(df_set["nb_players"])
    
    TOOLS[7] = HoverTool(tooltips=[
                            ("nb_players", "@nb_players")
                            ]
                        )
    px= figure(x_range=FactorRange(*x), 
               plot_height=350, plot_width = int(WIDTH*MULT_WIDTH),
               title="number of players, ({}, {}, {}, rate={}, price={})".format(
                  algo, scenario, gamma_version, rate, price),
                toolbar_location=None, tools=TOOLS)
    

    data = dict(x=x, nb_players=nb_players)
    
    source = ColumnDataSource(data=data)
    px.vbar(x='x', top='nb_players', width=0.9, source=source, 
            fill_color=factor_cmap('x', 
                                   palette=Category20[20], 
                                   factors=list(df_set["t"].unique()), 
                                   start=0, end=1))
    
    px.y_range.start = 0
    px.x_range.range_padding = 0.1
    px.xaxis.major_label_orientation = 1
    px.xgrid.grid_line_color = None
    
    return px
    

def plot_evolution_players_by_situation_over_time(
            df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
            algos=["LRI1"], 
            dico_SelectGammaVersion={"DETERMINIST": [1],"LRI1": [1],"LRI2": [0]}):
    """
    evolution of the number of player by situations over periods
    """
    df = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T.copy()
    df["pl_i"] = df.pl_i.astype(str)
    df["t"] = df.t.astype(str)
    
    rates = df.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df.prices.unique().tolist()
    scenarios = df.scenario.unique().tolist()
    gamma_version_root = "".join(list(df.gamma_version.unique()[0])[:-1])
    
    dico_pxs = dict()
    for algo, price, rate, scenario in it.product(algos, prices, 
                                                  rates, scenarios):
        gamma_versions = dico_SelectGammaVersion[algo]
        for gamma_version_number in gamma_versions:
            gamma_version = gamma_version_root+str(gamma_version_number)
            mask_al_pr_ra_sc_gam = ((df.rate == str(rate)) | (df.rate == 0)) \
                                    & (df.prices == price) \
                                    & (df.algo == algo) \
                                    & (df.scenario == scenario) \
                                    & (df.gamma_version == gamma_version)
            df_al_pr_ra_sc_gam = df[mask_al_pr_ra_sc_gam].copy()
            
            print("{}, {}, {}, df_al_pr_ra_sc_gam={}".format(algo, 
                    scenario, gamma_version, df_al_pr_ra_sc_gam.shape ))
            # pxs_al_pr_ra_sc_gam = plot_evolution_players_by_situation_for_time(
            #                         df_al_pr_ra_sc_gam, algo, rate, 
            #                         price, scenario, gamma_version)
            pxs_al_pr_ra_sc_gam = plot_evolution_DIFFERENCE_players_by_situation_for_time(
                                    df_al_pr_ra_sc_gam, algo, rate, 
                                    price, scenario, gamma_version)
            pxs_al_pr_ra_sc_gam.legend.click_policy="hide"
            
            if (algo, price, rate, scenario, gamma_version) not in dico_pxs.keys():
                dico_pxs[(algo, price, rate, scenario, gamma_version)] \
                    = [pxs_al_pr_ra_sc_gam]
            else:
                dico_pxs[(algo, price, rate, scenario, gamma_version)]\
                    .append(pxs_al_pr_ra_sc_gam)
        
    rows_evol_situations = list()
    for key, pxs_al_pr_ra_sc_gam in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra_sc_gam)
        rows_evol_situations.append(col_px_sts)
    rows_evol_situations = column(children=rows_evol_situations, 
                                  sizing_mode='stretch_both')
    
    return rows_evol_situations
    
    pass
# _____________________________________________________________________________
#
#      evolution of the number of player by situations over periods ---> fin
# _____________________________________________________________________________

# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> debut
# _____________________________________________________________________________
def group_plot_on_panel(df_B_C_BB_CC_RU_M, 
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        algos_to_show, 
                        dico_SelectGammaVersion):
    
    cols = ["B", "C", "BB", "CC", "RU"]
    for col in cols:
        df_B_C_BB_CC_RU_M[col] = df_B_C_BB_CC_RU_M[col].astype(float)
    
    cols = ["PROD", "CONS", "b0", "c0", "pi_sg_plus","pi_sg_minus", 
            "B", "C", "BB", "CC", "RU", "Cicum", "Picum"]
    for col in cols:
        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T[col] \
            = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T[col].astype(float)
            
    
    
    rows_RU_C_B_CC_BB = plot_comparaison_gamma_version_all_scenarios(
                            df_B_C_BB_CC_RU_M)
    tab_compGammaVersionAllScenario = Panel(child=rows_RU_C_B_CC_BB, 
                                  title="comparison Gamma_version all scenarios")
    print("comparison Gamma_version all scenarios: Terminee")
    
    rows_RU_CC_BB = plot_comparaison_gamma_version_RU(df_B_C_BB_CC_RU_M)
    tab_compGammaVersionRU = Panel(child=rows_RU_CC_BB, 
                                    title="comparison Gamma_version RU,BB,CC")
    print("comparison Gamma_version RU,BB,CC : Terminee")
    
    rows_B_C = plot_comparaison_gamma_version_BC(df_B_C_BB_CC_RU_M)
    tab_compGammaVersionBC = Panel(child=rows_B_C, 
                                    title="comparison Gamma_version B,C")
    print("comparison Gamma_version B,C : Terminee")
    
    rows_dists_ts = plot_distribution_by_states_4_periods(
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        dico_SelectGammaVersion=dico_SelectGammaVersion)
    tab_dists_ts = Panel(child=rows_dists_ts, title="distribution by state")
    print("Distribution of players: TERMINEE")
    
    rows_evol_RU_C_B_CC_BB = plot_evolution_RU_C_B_CC_BB_over_time(
                                df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                                algos=algos_to_show, 
                                dico_SelectGammaVersion=dico_SelectGammaVersion
                                )
    tabs_evol_over_time = Panel(child=rows_evol_RU_C_B_CC_BB, 
                                title="evolution C B CC BB RU over time")
    print("evolution of gains : TERMINEE")
    
    rows_evol_situations = plot_evolution_players_by_situation_over_time(
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        algos=algos_to_show, 
                        dico_SelectGammaVersion=dico_SelectGammaVersion
                    )
    tabs_evol_situation_over_time = Panel(child=rows_evol_situations, 
                                title="evolution of situation over time")
    print("evolution of Situation  : TERMINEE")
    
    rows_evol_Si = plot_evolution_Si_by_players_over_time(
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        algos=algos_to_show, 
                        dico_SelectGammaVersion=dico_SelectGammaVersion
                    )
    tabs_evol_Si_over_time = Panel(child=rows_evol_Si, 
                                title="evolution of stocks over time")
    print("evolution of Stocks Si : TERMINEE")
    
    tabs = Tabs(tabs= [ 
                        tab_compGammaVersionRU,
                        tab_compGammaVersionBC, 
                        tab_compGammaVersionAllScenario, 
                        tab_dists_ts,
                        tabs_evol_over_time, 
                        tabs_evol_situation_over_time,
                        tabs_evol_Si_over_time
                        ])
    NAME_RESULT_SHOW_VARS 
    name_result_show_vars = "comparaison_RU_BCBBCC_gammaVersionV1.html"
    output_file( os.path.join(name_dir, name_result_show_vars)  )
    save(tabs)
    show(tabs)

def OLD_group_plot_on_panel(df_B_C_BB_CC_RU_M, 
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        algos_to_show, 
                        gamma_versions_to_show, 
                        scenarios_to_show):
    
    cols = ["B", "C", "BB", "CC", "RU"]
    for col in cols:
        df_B_C_BB_CC_RU_M[col] = df_B_C_BB_CC_RU_M[col].astype(float)
    
    cols = ["PROD", "CONS", "b0", "c0", "pi_sg_plus","pi_sg_minus", 
            "B", "C", "BB", "CC", "RU"]
    for col in cols:
        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T[col] \
            = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T[col].astype(float)
            
    
    
    # rows_RU_C_B_CC_BB = plot_comparaison_gamma_version_all_scenarios(
    #                         df_B_C_BB_CC_RU_M)
    # tab_compGammaVersionAllScenario = Panel(child=rows_RU_C_B_CC_BB, 
    #                               title="comparison Gamma_version all scenarios")
    # print("comparison Gamma_version all scenarios: Terminee")
    
    # rows_RU_CC_BB = plot_comparaison_gamma_version_RU(df_B_C_BB_CC_RU_M)
    # tab_compGammaVersionRU = Panel(child=rows_RU_CC_BB, 
    #                                 title="comparison Gamma_version RU,BB,CC")
    # print("comparison Gamma_version RU,BB,CC : Terminee")
    
    # rows_B_C = plot_comparaison_gamma_version_BC(df_B_C_BB_CC_RU_M)
    # tab_compGammaVersionBC = Panel(child=rows_B_C, 
    #                                 title="comparison Gamma_version B,C")
    # print("comparison Gamma_version B,C : Terminee")
    
    # rows_dists_ts = plot_distribution_by_states_4_periods(
    #                     df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T)
    # tab_dists_ts = Panel(child=rows_dists_ts, title="distribution by state")
    # print("Distribution of players: TERMINEE")
    
    rows_evol_RU_C_B_CC_BB = plot_evolution_RU_C_B_CC_BB_over_time(
                                df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                                algos=algos_to_show, 
                                gamma_versions=gamma_versions_to_show, 
                                scenarios=scenarios_to_show
                                )
    tabs_evol_over_time = Panel(child=rows_evol_RU_C_B_CC_BB, 
                                title="evolution C B CC BB RU over time")
    print("evolution of gains : TERMINEE")
    
    rows_evol_Si = plot_evolution_Si_by_players_over_time(
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        algos=algos_to_show, 
                        dico_SelectGammaVersion=dico_SelectGammaVersion
                    )
    tabs_evol_Si_over_time = Panel(child=rows_evol_Si, 
                                title="evolution of stocks over time")
    print("evolution of Stocks Si : TERMINEE")
    
    tabs = Tabs(tabs= [ 
                        # tab_compGammaVersionRU,
                        # tab_compGammaVersionBC, 
                        # tab_compGammaVersionAllScenario, 
                        # tab_dists_ts,
                        tabs_evol_over_time,
                        tabs_evol_Si_over_time
                        ])
    NAME_RESULT_SHOW_VARS 
    name_result_show_vars = "comparaison_RU_BCBBCC_gammaVersionV1.html"
    output_file( os.path.join(name_dir, name_result_show_vars)  )
    save(tabs)
    show(tabs)
    
def DBG_group_plot_on_panel(df_B_C_BB_CC_RU_M, 
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        algos_to_show, 
                        dico_SelectGammaVersion):
    
    cols = ["B", "C", "BB", "CC", "RU"]
    for col in cols:
        df_B_C_BB_CC_RU_M[col] = df_B_C_BB_CC_RU_M[col].astype(float)
    
    cols = ["PROD", "CONS", "b0", "c0", "pi_sg_plus", "pi_sg_minus", 
            "B", "C", "BB", "CC", "RU"]
    for col in cols:
        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T[col] \
            = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T[col].astype(float)
            
    # rows_evol_RU_C_B_CC_BB = plot_evolution_RU_C_B_CC_BB_over_time(
    #                             df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
    #                             algos=algos_to_show, 
    #                             dico_SelectGammaVersion=dico_SelectGammaVersion
    #                             )
    # tabs_evol_over_time = Panel(child=rows_evol_RU_C_B_CC_BB, 
    #                             title="evolution C B CC BB RU over time")
    # print("evolution of gains : TERMINEE")
    
    # rows_evol_Si = plot_evolution_Si_by_players_over_time(
    #                     df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
    #                     algos=algos_to_show, 
    #                     dico_SelectGammaVersion=dico_SelectGammaVersion
    #                 )
    # tabs_evol_Si_over_time = Panel(child=rows_evol_Si, 
    #                             title="evolution of stocks over time")
    # print("evolution of Stocks Si : TERMINEE")
    
    rows_evol_situations = plot_evolution_players_by_situation_over_time(
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        algos=algos_to_show, 
                        dico_SelectGammaVersion=dico_SelectGammaVersion
                    )
    tabs_evol_situation_over_time = Panel(child=rows_evol_situations, 
                                title="evolution of situation over time")
    print("evolution of Situation  : TERMINEE")
    
    
    tabs = Tabs(tabs= [ 
                        #tabs_evol_over_time,
                        #tabs_evol_Si_over_time,
                        tabs_evol_situation_over_time
                        ])
    #NAME_RESULT_SHOW_VARS 
    name_result_show_vars = "comparaison_RU_BCBBCC_gammaVersionV1.html"
    output_file( os.path.join(name_dir, name_result_show_vars)  )
    #save(tabs)
    show(tabs)
        

# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> fin
# _____________________________________________________________________________

###############################################################################
#
#               representation des variables : fin
#
###############################################################################


if __name__ == "__main__":
    ti = time.time()
    
    k_steps = 250
    phi_name = "A1.2B0.8" #"A1B1" # A1B1, A1.2B0.8
    automate = "Automate2" # Automate1=Ancien automate, Automate2= nouveau automate du 04/05/2021
        
    # name_dir = os.path.join("tests", 
    #                         "gamma_V0_V1_V2_V3_V4_T20_kstep250_setACsetAB1B2C")
    # A1B1Automate2gamma_V0_V1_V2_V3_V4_T50_ksteps250_setACsetAB1B2C
    t_periods = 50 #20#50
    k_steps = 250 #5
    name_dir = os.path.join(
                "tests", 
                phi_name + automate \
                    +"gamma_V0_V1_V2_V3_V4_T"+str(t_periods)+"_ksteps"+str(k_steps)+"_setACsetAB1B2C")
    
    nb_sub_dir = len(name_dir.split(os.sep))
    
    
    selected_gamma_version = True;
    tuple_paths, path_2_best_learning_steps = list(), list()
    if selected_gamma_version:
        dico_SelectGammaVersion={"DETERMINIST": [0,1,2,3,4], 
                                  "LRI1": [0,1,2,3,4],
                                  "LRI2": [0,1,2,3,4]}
        dico_SelectGammaVersion={"DETERMINIST": [1], 
                                  "LRI1": [1], 
                                  "LRI2": [0]}
        tuple_paths, path_2_best_learning_steps \
            = get_tuple_paths_of_arrays_SelectGammaVersion(
                name_dirs=[name_dir], nb_sub_dir=nb_sub_dir,
                dico_SelectGammaVersion=dico_SelectGammaVersion)
    else:
        tuple_paths, path_2_best_learning_steps \
            = get_tuple_paths_of_arrays(name_dirs=[name_dir], 
                                        nb_sub_dir=nb_sub_dir)
    
    
    path_2_best_learning_steps = list(set(path_2_best_learning_steps))
    df_LRI_12_stop, dico_k_stop = dict(), dict()
    df_LRI_12_stop, df_k_stop = get_k_stop_4_periods(
                                    path_2_best_learning_steps,
                                    nb_sub_dir)
    print("get_k_stop_4_periods: TERMINE") 
    
    tuple_paths = list(set(tuple_paths))
    df_arr_M_T_Ks, df_ben_cst_M_T_K, \
    df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M, \
    df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T \
        = get_array_turn_df_for_t(tuple_paths, df_LRI_12_stop, 
                                  t=None, k_steps_args=k_steps, 
                                  nb_sub_dir=nb_sub_dir)    
    print("size")
    print("df_arr_M_T_Ks={} Mo".format(
            round(df_arr_M_T_Ks.memory_usage().sum()/(1024*1024), 2)))
    print("df_ben_cst_M_T_K={} Mo".format(
            round(df_ben_cst_M_T_K.memory_usage().sum()/(1024*1024), 2)))
    print("df_b0_c0_pisg_pi0_T_K={} Mo".format(
            round(df_b0_c0_pisg_pi0_T_K.memory_usage().sum()/(1024*1024), 2)))
    print("df_B_C_BB_CC_RU_M={} Mo".format(
            round(df_B_C_BB_CC_RU_M.memory_usage().sum()/(1024*1024), 2)))
    print("df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T={} Mo".format(
            round(df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T.memory_usage().sum()/(1024*1024), 2)))
    
   
    algos_to_show= ["LRI1", "DETERMINIST", "LRI2"];
    gamma_versions_to_show=[];
    scenarios_to_show=[];
   
    group_plot_on_panel(
        df_B_C_BB_CC_RU_M=df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T=df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
        algos_to_show=algos_to_show,
        dico_SelectGammaVersion=dico_SelectGammaVersion)
    
    # DBG_group_plot_on_panel(
    #     df_B_C_BB_CC_RU_M=df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
    #     df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T=df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
    #     algos_to_show=algos_to_show, 
    #     dico_SelectGammaVersion=dico_SelectGammaVersion)
    
   
    
    print("runtime={}".format(time.time() - ti))