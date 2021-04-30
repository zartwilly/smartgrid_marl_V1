# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:28:27 2021

@author: jwehounou
"""

import os
import time
import numpy as np
import pandas as pd
import itertools as it
import fonctions_auxiliaires as fct_aux
import visu_bkh_automate_v1 as autoVizGameV1
import visu_comparaison_RU_BCBBCC_gammaVersion as compVizRUBCBBCC

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
#               add new variables to array of players  --> debut
#                   pi_sg_{+,-}, b0, c0, B, C, BB, CC, RU 
# _____________________________________________________________________________ 
def add_new_vars_2_arr(algo_name, instance_name, gamma_version,
                       k_stop,
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
    Version compute B, BB, C, CC
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
    print("algo_name={}, t_periods={}".format(algo_name, t_periods))
    for t in range(0, t_periods):
        if algo_name in algos_4_no_learning:
            arr_pl_M_T_KSTOP_vars[:,t,list(fct_aux.AUTOMATE_INDEX_ATTRS.values()) ] \
                = arr_pl_M_T_K_vars[:,t,:] 
        else:
            arr_pl_M_T_KSTOP_vars[:,t,list(fct_aux.AUTOMATE_INDEX_ATTRS.values()) ] \
                = arr_pl_M_T_K_vars[:,t,k_stop,:]
            
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
            algo_name, instance_name, gamma_version))
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


# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> debut
# _____________________________________________________________________________
def get_list_Ninstances_specified_algo(rootdir_gamma, 
                                       algos_4_showing):
    
    paths_Ninstance_algos2show = list()
    cpt = 0
    for root, subdirs, files in os.walk(rootdir_gamma):
        if len(subdirs) == 0 \
            and len(files) > 1 \
            and (root.split(os.sep)[-2] in algos_4_showing 
                 or root.split(os.sep)[-1] in algos_4_showing):
            cpt+=1
            #print("root={}, subdirs={}, files={}".format( root, len(subdirs), len(files) ))
            paths_Ninstance_algos2show.append(
                root.split(os.sep))
    return paths_Ninstance_algos2show, cpt
            

def get_tuple_paths_of_arrays_4_many_simu(name_dir, sub1_name_dir,
                                          algos_4_showing, 
                        gamma_version_2_selecting_by_algo):
    
    # sub1_name_dirs = [ sub1_name_dir+""+gamma_version \
    #                    for gamma_version in gamma_version_2_selecting_by_algo]
    # algo_gammas = zip(algos_4_showing, gamma_version_2_selecting_by_algo)
    
    tuple_paths_Ninstance_algos = list()
    for gamma_version in set(gamma_version_2_selecting_by_algo):
        sub1_name_dir_ = sub1_name_dir+""+gamma_version
        
        # list recursively all N instances with specified algo 
        print("sub1_name_dir_ = {}".format(sub1_name_dir_))
        rootdir_gamma = os.path.join(name_dir, sub1_name_dir, sub1_name_dir_)
        paths_Ninstance_algos2show, cpt \
            = get_list_Ninstances_specified_algo(rootdir_gamma, 
                                                 algos_4_showing)
        print("{}, cpt={}".format(gamma_version, cpt))
        
        tuple_paths_Ninstance_algos.extend(paths_Ninstance_algos2show)
        
    return tuple_paths_Ninstance_algos
        
def get_k_stop_LRIx(path_to_variable):
    df_al = pd.read_csv(
                    os.path.join(path_to_variable, "best_learning_steps.csv"),
                    index_col=0)
    return df_al.loc["k_stop",'0']
   

        
def get_array_turn_df_for_t(tuple_paths_Ninstance_algos, 
                            t=0, k_steps_args=250, nb_sub_dir=-2,
                            dico_algo_gammas={'DETERMINIST':'GammaV1', 
                                              'LRI1':'GammaV1','LRI2':'GammaV0'},
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
    for tuple_path in tuple_paths_Ninstance_algos:
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
        
        price, algo_name, instance_name, rate, gamma_version, k_stop = None, None, None, None, None, None; 
        if len(tuple_path) == 6:
            algo_name = tuple_path[nb_sub_dir+1]                               # DETERMINIST
            price = tuple_path[nb_sub_dir].split("_")[3] \
                    +"_"+tuple_path[nb_sub_dir].split("_")[-1]
            rate = 0
        elif len(tuple_path) == 7:
            algo_name = tuple_path[nb_sub_dir]                                 # LRI1, LRI2
            price = tuple_path[nb_sub_dir-1].split("_")[3] \
                    +"_"+tuple_path[nb_sub_dir-1].split("_")[-1]
            rate = tuple_path[nb_sub_dir+1]
            k_stop = get_k_stop_LRIx(path_to_variable=path_to_variable)
        gamma_version = "".join(list(tuple_path[-1*nb_sub_dir])[-7:])
        instance_name = tuple_path[-1*nb_sub_dir+1]
        
        if dico_algo_gammas[algo_name] != gamma_version:
            continue
        
        arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW \
            = add_new_vars_2_arr(
                algo_name=algo_name, 
                instance_name=instance_name,
                gamma_version=gamma_version,
                k_stop=k_stop,
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
            = compVizRUBCBBCC.get_tuple_from_vars_4_columns(
                nb_t_periods, t, 
                algo_name, rate, price, 
                gamma_version, m_players, t_periods, k_steps, 
                scenario_name=instance_name)

            
        if algo_name in algos_4_learning:
            df_lri_x, df_b0_c0_pisg_pi0_T_K_lri = None, None
            df_ben_cst_M_T_K_lri, df_B_C_BB_CC_RU_M_lri = None, None
            df_M_T_lri_x = None
            
            df_lri_x, df_b0_c0_pisg_pi0_T_K_lri, \
            df_ben_cst_M_T_K_lri, df_B_C_BB_CC_RU_M_lri, \
            df_M_T_lri_x \
                = compVizRUBCBBCC.turn_arrays_2_2D_learning_algos(
                    arr_pl_M_T_K_vars, 
                    arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW, 
                    b0_s_T_K, c0_s_T_K,
                    B_is_M, C_is_M,
                    BENs_M_T_K, CSTs_M_T_K,
                    BB_is_M, CC_is_M, RU_is_M,
                    pi_sg_plus_T, pi_sg_minus_T,
                    pi_0_plus_T, pi_0_minus_T,
                    t_periods, k_steps=k_steps, 
                    tu_mtk=tu_mtk, tu_tk=tu_tk, tu_m=tu_m, tu_mt=tu_mt, 
                    variables=variables)
                
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
                = compVizRUBCBBCC.turn_arrays_2_2D_4_not_learning_algos(
                    arr_pl_M_T_K_vars, 
                    arr_pl_M_T_KSTOP_vars, AUTOMATE_INDEX_ATTRS_NEW, 
                    b0_s_T_K, c0_s_T_K,
                    B_is_M, C_is_M,
                    BENs_M_T_K, CSTs_M_T_K,
                    BB_is_M, CC_is_M, RU_is_M,
                    pi_sg_plus_T, pi_sg_minus_T,
                    pi_0_plus_T, pi_0_minus_T,
                    t_periods, k_steps=k_steps,
                    tu_mtk=tu_mtk, tu_tk=tu_tk, tu_m=tu_m, tu_mt=tu_mt, 
                    variables=variables
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
    columns_ind = ["algo","rate","prices","gamma_version","instance","pl_i","t","k"]
    df_arr_M_T_Ks = compVizRUBCBBCC.insert_index_as_df_columns(df_arr_M_T_Ks, columns_ind)
    ###  df_ben_cst_M_T_K
    columns_ind = ["algo","rate","prices","gamma_version","instance","pl_i","t","k"]
    df_ben_cst_M_T_K = compVizRUBCBBCC.insert_index_as_df_columns(df_ben_cst_M_T_K, columns_ind)
    df_ben_cst_M_T_K["state_i"] = df_arr_M_T_Ks["state_i"]
    ###  df_b0_c0_pisg_pi0_T_K
    columns_ind = ["algo","rate","prices","gamma_version","instance","t","k"]
    df_b0_c0_pisg_pi0_T_K = compVizRUBCBBCC.insert_index_as_df_columns(df_b0_c0_pisg_pi0_T_K, columns_ind)
    ###  df_B_C_BB_CC_RU_M
    columns_ind = ["algo","rate","prices","gamma_version","instance","pl_i"]
    df_B_C_BB_CC_RU_M = compVizRUBCBBCC.insert_index_as_df_columns(df_B_C_BB_CC_RU_M, columns_ind)
    ### df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T
    columns_ind = ["algo","rate","prices","gamma_version","instance","pl_i","t"]
    df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T \
        = compVizRUBCBBCC.insert_index_as_df_columns(df_B_C_BB_CC_RU_M_T, 
                                     columns_ind)
    
        
    return df_arr_M_T_Ks, df_ben_cst_M_T_K, \
            df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M, \
            df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T
        
# _____________________________________________________________________________ 
#               
#        get local variables and turn them into dataframe --> fin
# _____________________________________________________________________________

#------------------------------------------------------------------------------
#                       Plots variables
#------------------------------------------------------------------------------

# _____________________________________________________________________________ 
#               
#                   plot RU 4 various gamma_version 4 all scenarios 
#                            --> debut
# _____________________________________________________________________________ 
def plot_gamma_version_all_instances(df_ra_pr, rate, price):
    
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
    

def plot_comparaison_gamma_version_all_instances(df_B_C_BB_CC_RU_M):
    rates = df_B_C_BB_CC_RU_M.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df_B_C_BB_CC_RU_M.prices.unique().tolist()
    
    dico_pxs = dict()
    for rate, price in it.product(rates, prices):
        mask_ra_pr = ((df_B_C_BB_CC_RU_M.rate == rate) \
                      | (df_B_C_BB_CC_RU_M.rate == 0)) \
                        & (df_B_C_BB_CC_RU_M.prices == price)
        df_ra_pr = df_B_C_BB_CC_RU_M[mask_ra_pr].copy()
        
        pxs_pr_ra = plot_gamma_version_all_instances(df_ra_pr, rate, price)
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
#              evolution prices B, C, BB, CC, RU for periods ---> debut
# _____________________________________________________________________________

def plot_evolution_prices_for_instances(df_al_pr_ra_sc_gam, algo, rate, 
                                        price, gamma_version):
    
    cols = ["numInstance", "B", "C", "BB", "CC", "RU"]
    
    df_res_t = df_al_pr_ra_sc_gam.groupby(cols[0])[cols[1:]]\
                .agg({cols[1]:[np.mean, np.std, np.min, np.max], 
                      cols[2]:[np.mean, np.std, np.min, np.max], 
                      cols[3]:[np.mean, np.std, np.min, np.max], 
                      cols[4]:[np.mean, np.std, np.min, np.max], 
                      cols[5]:[np.mean, np.std, np.min, np.max]})
    
    df_res_t.columns = ["_".join(x) for x in df_res_t.columns.ravel()]
    df_res_t = df_res_t.reset_index()
    
    #df_res_t.t = df_res_t.t.astype("str")
    
    aggs = ["amin", "amax", "std", "mean"]
    tooltips = [("{}_{}".format(col, agg), "@{}_{}".format(col, agg)) 
                for (col, agg) in it.product(cols[1:], aggs)]
    TOOLS[7] = HoverTool(tooltips = tooltips)
    
    new_cols = [col[1].split("@")[1] 
                for col in tooltips if col[1].split("_")[1] == "mean"]
    print('new_cols={}, df_res_t.cols={}'.format(new_cols, df_res_t.columns))
    
    #x = list(map(tuple,list(df_res_t.t.values)))
    px = figure(x_range=df_res_t.numInstance.values.tolist(), 
                y_range=(0, df_res_t[new_cols].values.max() + 5), 
                plot_height = int(350), 
                plot_width = int(WIDTH*MULT_WIDTH), tools = TOOLS, 
                toolbar_location="above")
           
    data = dict(x = df_res_t.numInstance.values.tolist(), 
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
    
    title = "gain evolution over instances ({}, gamma_version={}, rate:{}, price={})".format(
                algo, gamma_version, rate, price, )
    px.title.text = title
    px.y_range.start = df_res_t.RU_mean.min() - 1
    px.x_range.range_padding = width
    px.xgrid.grid_line_color = None
    px.legend.location = "top_right" #"top_left"
    px.legend.orientation = "horizontal"
    px.xaxis.axis_label = "instances numbers"
    px.yaxis.axis_label = "values"
    
    return px
    


def plot_evolution_RU_C_B_CC_BB_over_instance(
                    df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
                    dico_algo_gammas
                    ):
    
    df = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T.copy()
    df["numInstance"] = df.instance.apply(lambda x: x.split("_")[3])
    
    rates = df.rate.unique(); rates = rates[rates!=0].tolist()
    prices = df.prices.unique().tolist()
    
    algos = df.algo.unique().tolist()
    
    dico_pxs = dict()
    for algo, price, rate in it.product(algos, prices, rates):
        mask_al_pr_ra_gam = ((df.rate == str(rate)) | (df.rate == 0)) \
                                & (df.prices == price) \
                                & (df.algo == algo) \
                                
        df_al_pr_ra_gam = df[mask_al_pr_ra_gam].copy()
        gamma_version = dico_algo_gammas[algo]
        
        print("{}, {}, df_al_pr_ra_gam={}".format(algo, 
                gamma_version, df_al_pr_ra_gam.shape ))
        pxs_al_pr_ra_gam = plot_evolution_prices_for_instances(
                                df_al_pr_ra_gam, algo, rate, 
                                price, gamma_version)
        pxs_al_pr_ra_gam.legend.click_policy="hide"
        
        if (algo, price, rate, gamma_version) not in dico_pxs.keys():
            dico_pxs[(algo, price, rate, gamma_version)] \
                = [pxs_al_pr_ra_gam]
        else:
            dico_pxs[(algo, price, rate, gamma_version)]\
                .append(pxs_al_pr_ra_gam)
        
    rows_evol_RU_C_B_CC_BB = list()
    for key, pxs_al_pr_ra_gam in dico_pxs.items():
        col_px_sts = column(pxs_al_pr_ra_gam)
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
#                   affichage  dans tab  ---> debut
# _____________________________________________________________________________
def group_plot_on_panel(df_B_C_BB_CC_RU_M, 
                        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                        dico_algo_gammas, 
                        name_dir_oneperiod):
    
    cols = ["B", "C", "BB", "CC", "RU"]
    for col in cols:
        df_B_C_BB_CC_RU_M[col] = df_B_C_BB_CC_RU_M[col].astype(float)
    
    cols = ["PROD", "CONS", "b0", "c0", "pi_sg_plus","pi_sg_minus", 
            "B", "C", "BB", "CC", "RU"]
    for col in cols:
        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T[col] \
            = df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T[col].astype(float)
            
    
    
    rows_RU_C_B_CC_BB = plot_comparaison_gamma_version_all_instances(
                            df_B_C_BB_CC_RU_M)
    tab_compGammaVersionAllScenario = Panel(child=rows_RU_C_B_CC_BB, 
                                  title="comparison Gamma_version all scenarios")
    print("comparison Gamma_version all scenarios: Terminee")
    
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
    
    rows_evol_RU_C_B_CC_BB = plot_evolution_RU_C_B_CC_BB_over_instance(
                                df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T,
                                dico_algo_gammas
                                )
    tabs_evol_over_instances = Panel(child=rows_evol_RU_C_B_CC_BB, 
                                title="evolution C B CC BB RU over time")
    print("evolution of gains : TERMINEE")
    
    tabs = Tabs(tabs= [ 
                        #tab_compGammaVersionRU,
                        #tab_compGammaVersionBC, 
                        tab_compGammaVersionAllScenario, 
                        #tab_dists_ts,
                        tabs_evol_over_instances
                        ])
    NAME_RESULT_SHOW_VARS 
    name_result_show_vars = "comparaison_RU_BCBBCC_50instances.html"
    output_file( os.path.join(name_dir_oneperiod, name_result_show_vars)  )
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
    
    algos_4_showing=["DETERMINIST", "LRI1", "LRI2"]
    gamma_version_2_selecting_by_algo = ["GammaV1", "GammaV1", "GammaV0"]
    dico_algo_gammas = dict(zip(algos_4_showing, gamma_version_2_selecting_by_algo))
    
    name_dir = "tests"; sub1_name_dir = "OnePeriod_50instances"
    name_dir_oneperiod = os.path.join(name_dir,"OnePeriod_50instances")
    
    N_INSTANCES = 50
    tuple_paths_Ninstance_algos = get_tuple_paths_of_arrays_4_many_simu(name_dir, sub1_name_dir, 
                    algos_4_showing=algos_4_showing, 
                    gamma_version_2_selecting_by_algo=gamma_version_2_selecting_by_algo)
    
    nb_sub_dir = -2
    k_steps_args = 250
    t = 0
    df_arr_M_T_Ks, df_ben_cst_M_T_K, \
    df_b0_c0_pisg_pi0_T_K, df_B_C_BB_CC_RU_M, \
    df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T \
        = get_array_turn_df_for_t(tuple_paths_Ninstance_algos, 
                            t=t, k_steps_args=k_steps_args, nb_sub_dir=nb_sub_dir,
                            dico_algo_gammas=dico_algo_gammas)
        
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
    instances_to_show=[];
   
    group_plot_on_panel(
        df_B_C_BB_CC_RU_M=df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
        df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T=df_B_C_BB_CC_RU_CONS_PROD_b0_c0_pisg_M_T, 
        dico_algo_gammas=dico_algo_gammas, 
        name_dir_oneperiod=name_dir_oneperiod)
    
    
    print("runtime={}".format(time.time() - ti))
    