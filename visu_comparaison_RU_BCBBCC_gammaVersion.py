# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:23:43 2021

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
    pi_hp_plus_s = np.load(os.path.join(path_to_variable, "pi_hp_plus_s.npy"),
                          allow_pickle=True)
    pi_hp_minus_s = np.load(os.path.join(path_to_variable, "pi_hp_minus_s.npy"),
                          allow_pickle=True)
    
    return arr_pl_M_T_K_vars, \
            b0_s_T_K, c0_s_T_K, \
            B_is_M, C_is_M, \
            BENs_M_T_K, CSTs_M_T_K, \
            BB_is_M, CC_is_M, RU_is_M, \
            pi_sg_plus_T_K, pi_sg_minus_T_K, \
            pi_0_plus_T_K, pi_0_minus_T_K, \
            pi_hp_plus_s, pi_hp_minus_s
            
            
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

def get_array_turn_df_for_t_WITHOUT_SCENARIO(
                            tuple_paths, t=1, k_steps_args=250, nb_sub_dir=1,
                            algos_4_no_learning=["DETERMINIST","RD-DETERMINIST",
                                                 "BEST-BRUTE-FORCE",
                                                 "BAD-BRUTE-FORCE", 
                                                 "MIDDLE-BRUTE-FORCE"], 
                            algos_4_learning=["LRI1", "LRI2"]):
    """
    la colonne SCenario n'est pas ajoutee dans le dataframe
    """
    df_arr_M_T_Ks = []
    df_b0_c0_pisg_pi0_T_K = []
    df_B_C_BB_CC_RU_M = []
    df_ben_cst_M_T_K = []
    for tuple_path in tuple_paths:
        path_to_variable = os.path.join(*tuple_path)
        
        arr_pl_M_T_K_vars, \
        b0_s_T_K, c0_s_T_K, \
        B_is_M, C_is_M, \
        BENs_M_T_K, CSTs_M_T_K, \
        BB_is_M, CC_is_M, RU_is_M, \
        pi_sg_plus_T, pi_sg_minus_T, \
        pi_0_plus_T, pi_0_minus_T, \
        pi_hp_plus_s, pi_hp_minus_s \
            = get_local_storage_variables(path_to_variable)
        
        # price = tuple_path[2].split("_")[3]+"_"+tuple_path[2].split("_")[-1]
        # algo = tuple_path[3];
        # rate = tuple_path[4] if algo in algos_4_learning else 0
        # gamma_version = "".join(list(tuple_path[1].split("_")[4])[3:])
        #______________
        print(tuple_path )
        price = tuple_path[nb_sub_dir+1].split("_")[nb_sub_dir+2] \
                +"_"+tuple_path[nb_sub_dir+1].split("_")[-1]
        algo = tuple_path[nb_sub_dir+2];
        rate = tuple_path[nb_sub_dir+3] if algo in algos_4_learning else 0
        gamma_version = "".join(list(tuple_path[nb_sub_dir].split("_")[4])[3:])
        #______________
        
        #print("{}, {} debut:".format( tuple_paths[0][1].split("_")[3], gamma_version))
        print("{}, {} debut:".format( tuple_path[nb_sub_dir].split("_")[3], gamma_version))
        
        
        m_players = arr_pl_M_T_K_vars.shape[0]
        k_steps = arr_pl_M_T_K_vars.shape[2] if arr_pl_M_T_K_vars.shape == 4 \
                                             else k_steps_args                                    
        #for t in range(0, t_periods):                                     
        t_periods = None; tu_mtk = None; tu_tk = None; tu_m = None
        if t is None:
            t_periods = arr_pl_M_T_K_vars.shape[1]
            tu_mtk = list(it.product([algo], [rate], [price], [gamma_version],
                                     range(0, m_players), 
                                     range(0, t_periods), 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [gamma_version],
                                    range(0, t_periods), 
                                    range(0, k_steps)))
            t_periods = list(range(0, t_periods))
        elif type(t) is list:
            t_periods = t
            tu_mtk = list(it.product([algo], [rate], [price], [gamma_version],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [gamma_version],
                                    t_periods, 
                                    range(0, k_steps)))
        elif type(t) is int:
            t_periods = [t]
            tu_mtk = list(it.product([algo], [rate], [price], [gamma_version],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [gamma_version],
                                    t_periods, 
                                    range(0, k_steps)))
                      
        print('t_periods = {}'.format(t_periods))
        tu_m = list(it.product([algo], [rate], [price], [gamma_version], 
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
    columns_ind = ["algo","rate","prices","gamma_version","pl_i","t","k"]
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
    columns_ind = ["algo","rate","prices","gamma_version","pl_i","t","k"]
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
    columns_ind = ["algo","rate","prices","gamma_version","t","k"]
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
    columns_ind = ["algo","rate","prices","gamma_version","pl_i"]
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
            
def get_array_turn_df_for_t(tuple_paths, t=1, k_steps_args=250, nb_sub_dir=1,
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
        
        arr_pl_M_T_K_vars, \
        b0_s_T_K, c0_s_T_K, \
        B_is_M, C_is_M, \
        BENs_M_T_K, CSTs_M_T_K, \
        BB_is_M, CC_is_M, RU_is_M, \
        pi_sg_plus_T, pi_sg_minus_T, \
        pi_0_plus_T, pi_0_minus_T, \
        pi_hp_plus_s, pi_hp_minus_s \
            = get_local_storage_variables(path_to_variable)
        
        price = tuple_path[nb_sub_dir+1].split("_")[3] \
                +"_"+tuple_path[nb_sub_dir+1].split("_")[-1]
        algo = tuple_path[nb_sub_dir+2];
        rate = tuple_path[nb_sub_dir+3] if algo in algos_4_learning else 0
        gamma_version = "".join(list(tuple_path[nb_sub_dir].split("_")[4])[3:])
        
        scenario = tuple_path[nb_sub_dir].split("_")[3]
        print("nb_sub_dir = " + str(nb_sub_dir))
        print(tuple_path)
        print("{}, {}, price={} debut:".format( scenario, gamma_version, price))
        
        
        m_players = arr_pl_M_T_K_vars.shape[0]
        k_steps = arr_pl_M_T_K_vars.shape[2] if arr_pl_M_T_K_vars.shape == 4 \
                                             else k_steps_args                                    
        #for t in range(0, t_periods):                                     
        t_periods = None; tu_mtk = None; tu_tk = None; tu_m = None
        if t is None:
            t_periods = arr_pl_M_T_K_vars.shape[1]
            tu_mtk = list(it.product([algo], [rate], [price], [gamma_version],
                                     [scenario], 
                                     range(0, m_players), 
                                     range(0, t_periods), 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [gamma_version],
                                    [scenario],
                                    range(0, t_periods), 
                                    range(0, k_steps)))
            t_periods = list(range(0, t_periods))
        elif type(t) is list:
            t_periods = t
            tu_mtk = list(it.product([algo], [rate], [price], [gamma_version],
                                     [scenario],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [gamma_version],
                                    [scenario],
                                    t_periods, 
                                    range(0, k_steps)))
        elif type(t) is int:
            t_periods = [t]
            tu_mtk = list(it.product([algo], [rate], [price], [gamma_version],
                                     [scenario],
                                     range(0, m_players), 
                                     t_periods, 
                                     range(0, k_steps)))
            tu_tk = list(it.product([algo], [rate], [price], [gamma_version],
                                    [scenario],
                                    t_periods, 
                                    range(0, k_steps)))
                      
        print('t_periods = {}'.format(t_periods))
        tu_m = list(it.product([algo], [rate], [price], [gamma_version],
                               [scenario],
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
    columns_ind = ["algo","rate","prices","gamma_version","scenario","pl_i","t","k"]
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
    columns_ind = ["algo","rate","prices","gamma_version","scenario","pl_i","t","k"]
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
    columns_ind = ["algo","rate","prices","gamma_version","scenario","t","k"]
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
    columns_ind = ["algo","rate","prices","gamma_version","scenario","pl_i"]
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
    
    df_B_C_BB_CC_RU_M["B"] = df_B_C_BB_CC_RU_M["B"].astype(float)
    df_B_C_BB_CC_RU_M["C"] = df_B_C_BB_CC_RU_M["C"].astype(float)
    df_B_C_BB_CC_RU_M["BB"] = df_B_C_BB_CC_RU_M["BB"].astype(float)
    df_B_C_BB_CC_RU_M["CC"] = df_B_C_BB_CC_RU_M["CC"].astype(float)
    df_B_C_BB_CC_RU_M["RU"] = df_B_C_BB_CC_RU_M["RU"].astype(float)
    
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
#                   plot RU 4 various gamma_version 4 each scenario 
#                            --> debut
# _____________________________________________________________________________ 
def plot_gamma_version_RU_OLD(df_ra_pr, rate, price, scenario):
    
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
    

def plot_comparaison_gamma_version_RU_OLD(df_B_C_BB_CC_RU_M):
    rates = df_B_C_BB_CC_RU_M.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df_B_C_BB_CC_RU_M.prices.unique().tolist()
    scenarios = df_B_C_BB_CC_RU_M.scenario.unique().tolist()
    
    df_B_C_BB_CC_RU_M["B"] = df_B_C_BB_CC_RU_M["B"].astype(float)
    df_B_C_BB_CC_RU_M["C"] = df_B_C_BB_CC_RU_M["C"].astype(float)
    df_B_C_BB_CC_RU_M["BB"] = df_B_C_BB_CC_RU_M["BB"].astype(float)
    df_B_C_BB_CC_RU_M["CC"] = df_B_C_BB_CC_RU_M["CC"].astype(float)
    df_B_C_BB_CC_RU_M["RU"] = df_B_C_BB_CC_RU_M["RU"].astype(float)
    
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
    
    df_B_C_BB_CC_RU_M["BB"] = df_B_C_BB_CC_RU_M["BB"].astype(float)
    df_B_C_BB_CC_RU_M["CC"] = df_B_C_BB_CC_RU_M["CC"].astype(float)
    df_B_C_BB_CC_RU_M["RU"] = df_B_C_BB_CC_RU_M["RU"].astype(float)
    
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
    
    df_B_C_BB_CC_RU_M["B"] = df_B_C_BB_CC_RU_M["B"].astype(float)
    df_B_C_BB_CC_RU_M["C"] = df_B_C_BB_CC_RU_M["C"].astype(float)
    
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
#                   affichage  dans tab  ---> debut
# _____________________________________________________________________________
def group_plot_on_panel(df_B_C_BB_CC_RU_M):
    
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
    tabs = Tabs(tabs= [ 
                        tab_compGammaVersionRU,
                        tab_compGammaVersionBC, 
                        tab_compGammaVersionAllScenario
                       ])
    NAME_RESULT_SHOW_VARS 
    name_result_show_vars = "comparaison_RU_BCBBCC_gammaVersionV1.html"
    output_file( os.path.join(name_dir, name_result_show_vars)  )
    save(tabs)
    show(tabs)
    
# _____________________________________________________________________________
#
#                   affichage  dans tab  ---> fin
# _____________________________________________________________________________


if __name__ == "__main__":
    ti = time.time()
        
    k_steps = 250
    
    name_dir = os.path.join("tests", 
                            "gamma_V0_V1_V2_V3_V4_T10_kstep250_setAB1B2C")
    name_dir = os.path.join("tests", 
                            "gamma_V0_V1_V2_V3_V4_T50_kstep250_setACsetAB1B2C")
    name_dir = os.path.join("tests", 
                            "gamma_V1_V3_T30_kstep250_setACsetAB1B2C")
    #gamma_V0_V1_V2_V3_T20_kstep250_setACsetAB1B2C
    name_dir = os.path.join("tests", 
                            "gamma_V0_V1_V2_V3_V4_T20_kstep250_setACsetAB1B2C")
    
    #_______ debug understand moy_Vi increase with t periods _________________
    # k_steps = 50
    # name_dir = os.path.join("tests", 
    #                         "DBGgammaMulti_V0_V1_V2_V3_V4_T3_kstep50_setACsetAB1B2C")
    # name_dir = os.path.join("tests", 
    #                         "DBGgammaProced_V0_V1_V2_V3_V4_T3_kstep50_setACsetAB1B2C")
    # name_dir = os.path.join("tests", 
    #                         "DBGgamma_V0_V1_V2_V3_V4_T3_kstep50_setAC")
    # name_dir = os.path.join("tests", 
    #                         "DBGgamma_V0_V1_V2_V3_V4_T3_kstep50_setAB1B2C")
    #_______ debug understand moy_Vi increase with t periods _________________
    
    #name_dir = os.path.join("tests_cp")
    nb_sub_dir = len(name_dir.split(os.sep))
    
        
    selected_gamma_version = True;
    tuple_paths, path_2_best_learning_steps = list(), list()
    if selected_gamma_version:
        dico_SelectGammaVersion={"DETERMINIST": [0,1,2,3,4], 
                                  "LRI1": [0,1,2,3,4],
                                  "LRI2": [0,1,2,3,4]}
        # dico_SelectGammaVersion={"DETERMINIST": [1,3], 
        #                          "LRI1": [1],
        #                          "LRI2": [3]}
        # dico_SelectGammaVersion={"DETERMINIST": [1,3], 
        #                          "LRI1": [1,3],
        #                           "LRI2": [1,3]}
        tuple_paths, path_2_best_learning_steps \
            = get_tuple_paths_of_arrays_SelectGammaVersion(
                name_dirs=[name_dir], nb_sub_dir=nb_sub_dir,
                dico_SelectGammaVersion=dico_SelectGammaVersion)
    else:
        tuple_paths, path_2_best_learning_steps \
            = get_tuple_paths_of_arrays(name_dirs=[name_dir], 
                                        nb_sub_dir=nb_sub_dir)
            
    tuple_paths = list(set(tuple_paths))
        
    df_arr_M_T_Ks, df_ben_cst_M_T_K, \
    df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M \
        = get_array_turn_df_for_t(tuple_paths, t=1, k_steps_args=k_steps, 
                                  nb_sub_dir=nb_sub_dir)
        
    print("size: df_arr_M_T_Ks={} Mo, df_ben_cst_M_T_K={} Mo, df_b0_c0_pisg_pi0_T_K={} Mo, df_B_C_BB_CC_RU_M={} Mo".format(
                  round(df_arr_M_T_Ks.memory_usage().sum()/(1024*1024), 2),  
                  round(df_ben_cst_M_T_K.memory_usage().sum()/(1024*1024), 2),
                  round(df_b0_c0_pisg_pi0_T_K.memory_usage().sum()/(1024*1024), 2),
                  round(df_B_C_BB_CC_RU_M.memory_usage().sum()/(1024*1024), 4)
                  ))
    
    group_plot_on_panel(df_B_C_BB_CC_RU_M)
    
    print("runtime={}".format(time.time() - ti))
    