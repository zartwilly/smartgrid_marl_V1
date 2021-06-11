# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 09:28:54 2021

@author: jwehounou
"""

import os
import sys
import time
import math
import json
import numpy as np
import pandas as pd
import itertools as it
import multiprocessing as mp

import fonctions_auxiliaires as fct_aux
import deterministic_game_model_automate_4_all_t as autoDetGameModel
import lri_game_model_automate_4_all_t as autoLriGameModel
import force_brute_game_model_automate_4_all_t as autoBfGameModel
import detection_nash_game_model_automate_4_all_t as autoNashGameModel

from datetime import datetime
from pathlib import Path

ALGOS_LRI = ["LRI1", "LRI2"]
ALGOS_DET = ["DETERMINIST", "RD-DETERMINIST"]
#------------------------------------------------------------------------------
#                   definitions of functions
#------------------------------------------------------------------------------
#_________              create dico of parameters: debut         ______________ 
def define_parameters(dico_params):
    """
    create a list of parameters applied to the running of each algo 
    in the multiprocessing execution 

    """
    
    params = list()
    
    for algo_name, pi_hp_plus_minus, learning_rate \
        in it.product(dico_params["algos"], 
                      dico_params["tuple_pi_hp_plus_minus"],
                      dico_params['learning_rates']):
        params.append( (dico_params["arr_pl_M_T_vars_init"], 
                        algo_name, 
                        pi_hp_plus_minus[0], 
                        pi_hp_plus_minus[1],
                        dico_params["a"],
                        dico_params["b"],
                        learning_rate, 
                        dico_params["k_steps"],
                        dico_params["name_dir"], 
                        dico_params["date_hhmm"],
                        dico_params["gamma_version"], 
                        dico_params["used_instances"], 
                        dico_params["used_storage_det"], 
                        dico_params["manual_debug"], 
                        dico_params["criteria_bf"],
                        dico_params["debug"]
                        ) )
    return params

def define_parameters_multi_gammaV(dico_params):
    """
    create a list of parameters applied to the running of each algo 
    in the multiprocessing execution 

    """
    
    params = list()
    
    for algo_name, pi_hp_plus_minus, learning_rate, gamma_version \
        in it.product(dico_params["algos"], 
                      dico_params["tuple_pi_hp_plus_minus"],
                      dico_params['learning_rates'], 
                      dico_params["gamma_versions"]):
        
        date_hhmm_new = "_".join([dico_params["date_hhmm"], dico_params["scenario"], 
                              "".join(["T", str(dico_params["t_periods"]),
                                "".join(["gammaV", str(gamma_version)])])])
            
        params.append( (dico_params["arr_pl_M_T_vars_init"], 
                        algo_name, 
                        pi_hp_plus_minus[0], 
                        pi_hp_plus_minus[1],
                        dico_params["a"],
                        dico_params["b"],
                        learning_rate, 
                        dico_params["k_steps"],
                        dico_params["name_dir"], 
                        date_hhmm_new,
                        gamma_version, 
                        dico_params["used_instances"], 
                        dico_params["used_storage_det"], 
                        dico_params["manual_debug"], 
                        dico_params["criteria_bf"],
                        dico_params["debug"]
                        ) )
    return params


def define_parameters_multi_gammaV_arrplMTVars(dico_params):
    """
    create a list of parameters applied to the running of each algo 
    in the multiprocessing execution 
    two lists might differ by the array arr_pl_M_T_vars_init
    """
    params = list()
    
    for (scenario_name_012, arrplMT_012), algo_name, pi_hp_plus_minus, \
        learning_rate, gamma_version \
        in it.product(dico_params["zip_scen_arr"],
                      dico_params["algos"], 
                      dico_params["tuple_pi_hp_plus_minus"],
                      dico_params['learning_rates'], 
                      dico_params["gamma_versions"]):
        
        #print("arrplMT_scens={},len={}".format( type(arrplMT_scens), len(arrplMT_scens) ))
        date_hhmm_new = "_".join([dico_params["date_hhmm"], scenario_name_012, 
                          "".join(["T", str(dico_params["t_periods"]),
                            "".join(["gammaV", str(gamma_version)])])])
        #print("arr_pl_M_T_vars_init: type={}, len={}, ".format(type(arr_pl_M_T_vars_init), len(arr_pl_M_T_vars_init) ))
        params.append( (arrplMT_012, 
                        algo_name, 
                        pi_hp_plus_minus[0], 
                        pi_hp_plus_minus[1],
                        dico_params["a"],
                        dico_params["b"],
                        learning_rate, 
                        dico_params["k_steps"],
                        dico_params["name_dir"], 
                        date_hhmm_new,
                        gamma_version, 
                        dico_params["used_instances"], 
                        dico_params["used_storage_det"], 
                        dico_params["manual_debug"], 
                        dico_params["criteria_bf"],
                        dico_params["debug"]
                        ) )
        
    return params
            
            
#_________              create dico of parameters: fin           ______________ 

#_________                  all ALGOs: debut                     ______________ 
def execute_algos_used_Generated_instances(arr_pl_M_T_vars_init,
                                            name_dir=None,
                                            date_hhmm=None,
                                            k_steps=None,
                                            NB_REPEAT_K_MAX=None,
                                            algos=None,
                                            learning_rates=None,
                                            pi_hp_plus=None,
                                            pi_hp_minus=None,
                                            a=1, b=1,
                                            gamma_version=1,
                                            used_instances=True,
                                            used_storage_det=True,
                                            manual_debug=False, 
                                            criteria_bf="Perf_t", 
                                            debug=False):
    """
    execute algos by using generated instances if there exists or 
        by generating new instances
    
    date_hhmm="1041"
    algos=["LRI1"]
    
    """
    # directory to save  execution algos
    name_dir = "tests" if name_dir is None else name_dir
    date_hhmm = datetime.now().strftime("%d%m_%H%M") \
            if date_hhmm is None \
            else date_hhmm
    
    # steps of learning
    k_steps = 5 if k_steps is None else k_steps
    fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    # list of algos
    ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]\
            + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    algos = ALGOS if algos is None \
                  else algos
    # list of pi_hp_plus, pi_hp_minus
    pi_hp_plus = [0.2*pow(10,-3)] if pi_hp_plus is None else pi_hp_plus
    pi_hp_minus = [0.33] if pi_hp_minus is None else pi_hp_minus
    # learning rate 
    learning_rates = [0.01] \
            if learning_rates is None \
            else learning_rates # list(np.arange(0.05, 0.15, step=0.05))
    
    
    
    zip_pi_hp = list(zip(pi_hp_plus, pi_hp_minus))
    
    cpt = 0
    algo_piHpPlusMinus_learningRate \
            = it.product(algos, zip_pi_hp, learning_rates)
    
    for (algo_name, (pi_hp_plus_elt, pi_hp_minus_elt), 
             learning_rate) in algo_piHpPlusMinus_learningRate:
        
        print("______ execution {}: {}, rate={}______".format(cpt, 
                    algo_name, learning_rate))
        cpt += 1
        msg = "pi_hp_plus_"+str(pi_hp_plus_elt)\
                       +"_pi_hp_minus_"+str(pi_hp_minus_elt)
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
        if algo_name == ALGOS[0]:
            # 0: LRI1
            print("*** ALGO: {} *** ".format(algo_name))
            utility_function_version = 1
            path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name, str(learning_rate)
                                    )
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_K_vars = autoLriGameModel\
                                .lri_balanced_player_game_all_pijk_upper_08(
                                    arr_pl_M_T_vars_init.copy(),
                                    pi_hp_plus=pi_hp_plus_elt, 
                                    pi_hp_minus=pi_hp_minus_elt,
                                    a=a, b=b,
                                    gamma_version=gamma_version,
                                    k_steps=k_steps, 
                                    learning_rate=learning_rate,
                                    p_i_j_ks=p_i_j_ks,
                                    utility_function_version=utility_function_version,
                                    path_to_save=path_to_save, 
                                    manual_debug=manual_debug, dbg=debug)
        elif algo_name == ALGOS[1]:
            # 1: LRI2
            print("*** ALGO: {} *** ".format(algo_name))
            utility_function_version = 2
            path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name, str(learning_rate)
                                    )
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_K_vars = autoLriGameModel\
                                .lri_balanced_player_game_all_pijk_upper_08(
                                    arr_pl_M_T_vars_init.copy(),
                                    pi_hp_plus=pi_hp_plus_elt, 
                                    pi_hp_minus=pi_hp_minus_elt,
                                    a=a, b=b,
                                    gamma_version=gamma_version,
                                    k_steps=k_steps, 
                                    learning_rate=learning_rate,
                                    p_i_j_ks=p_i_j_ks,
                                    utility_function_version=utility_function_version,
                                    path_to_save=path_to_save, 
                                    manual_debug=manual_debug, dbg=debug)
                                
        elif algo_name == ALGOS[2] or algo_name == ALGOS[3]:
            # 2: DETERMINIST, 3: RANDOM DETERMINIST
            print("*** ALGO: {} *** ".format(algo_name))
            random_determinist = False if algo_name == ALGOS[2] else True
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoDetGameModel.determinist_balanced_player_game(
                             arr_pl_M_T_vars_init.copy(),
                             pi_hp_plus=pi_hp_plus_elt, 
                             pi_hp_minus=pi_hp_minus_elt,
                             a=a, b=b,
                             gamma_version=gamma_version,
                             random_determinist=random_determinist,
                             used_storage=used_storage_det,
                             path_to_save=path_to_save, 
                             manual_debug=manual_debug, dbg=debug)
            
        elif algo_name == fct_aux.ALGO_NAMES_BF[0] :
            # 0: BEST_BRUTE_FORCE (BF) , 1:BAD_BF, 2: MIDDLE_BF
            # execute tous les BF
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoBfGameModel\
                            .bf_balanced_player_game(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                a=a, b=b,
                                gamma_version=gamma_version,
                                path_to_save=path_to_save, 
                                name_dir=name_dir, 
                                date_hhmm=date_hhmm,
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
                            
                           
        elif algo_name == fct_aux.ALGO_NAMES_NASH[0] :
            # 0: "BEST-NASH", 1: "BAD-NASH", 2: "MIDDLE-NASH"
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoNashGameModel\
                            .nash_balanced_player_game(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                a=a, b=b,
                                gamma_version=gamma_version,
                                path_to_save=path_to_save, 
                                name_dir=name_dir, 
                                date_hhmm=date_hhmm,
                                manual_debug=manual_debug, 
                                dbg=debug)    
    
        
    print("NB_EXECUTION cpt={}".format(cpt))
#_________                   all ALGOs: fin                     ______________ 

#_________            all ALGOs N instances: debut                     __________ 
def execute_algos_used_Generated_instances_N_INSTANCES(arr_pl_M_T_vars_init,
                                            name_dir=None,
                                            date_hhmm=None,
                                            k_steps=None,
                                            NB_REPEAT_K_MAX=None,
                                            algos=None,
                                            learning_rates=None,
                                            pi_hp_plus=None,
                                            pi_hp_minus=None,
                                            a=1, b=1,
                                            gamma_version=1,
                                            used_instances=True,
                                            used_storage_det=True,
                                            manual_debug=False, 
                                            criteria_bf="Perf_t", 
                                            debug=False):
    """
    execute algos by using generated instances if there exists or 
        by generating new instances
    
    date_hhmm="1041"
    algos=["LRI1"]
    
    """
    # directory to save  execution algos
    name_dir = "tests" if name_dir is None else name_dir
    date_hhmm = datetime.now().strftime("%d%m_%H%M") \
            if date_hhmm is None \
            else date_hhmm
    
    # steps of learning
    k_steps = 5 if k_steps is None else k_steps
    fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    # list of algos
    ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]\
            + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    algos = ALGOS if algos is None \
                  else algos
    # list of pi_hp_plus, pi_hp_minus
    pi_hp_plus = [0.2*pow(10,-3)] if pi_hp_plus is None else pi_hp_plus
    pi_hp_minus = [0.33] if pi_hp_minus is None else pi_hp_minus
    # learning rate 
    learning_rates = [0.01] \
            if learning_rates is None \
            else learning_rates # list(np.arange(0.05, 0.15, step=0.05))
    
    
    
    zip_pi_hp = list(zip(pi_hp_plus, pi_hp_minus))
    
    cpt = 0
    algo_piHpPlusMinus_learningRate \
            = it.product(algos, zip_pi_hp, learning_rates)
    
    for (algo_name, (pi_hp_plus_elt, pi_hp_minus_elt), 
             learning_rate) in algo_piHpPlusMinus_learningRate:
        
        print("______ execution {}: {}, rate={}______".format(cpt, 
                    algo_name, learning_rate))
        cpt += 1
        msg = "pi_hp_plus_"+str(pi_hp_plus_elt)\
                       +"_pi_hp_minus_"+str(pi_hp_minus_elt)
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name
                                    )
        if algo_name == ALGOS[0]:
            # 0: LRI1
            print("*** ALGO: {} *** ".format(algo_name))
            utility_function_version = 1
            path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name, str(learning_rate)
                                    )
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_K_vars = autoLriGameModel\
                                .lri_balanced_player_game_all_pijk_upper_08(
                                    arr_pl_M_T_vars_init.copy(),
                                    pi_hp_plus=pi_hp_plus_elt, 
                                    pi_hp_minus=pi_hp_minus_elt,
                                    a=a, b=b,
                                    gamma_version=gamma_version,
                                    k_steps=k_steps, 
                                    learning_rate=learning_rate,
                                    p_i_j_ks=p_i_j_ks,
                                    utility_function_version=utility_function_version,
                                    path_to_save=path_to_save, 
                                    manual_debug=manual_debug, dbg=debug)
        elif algo_name == ALGOS[1]:
            # 1: LRI2
            print("*** ALGO: {} *** ".format(algo_name))
            utility_function_version = 2
            path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                    msg, algo_name, str(learning_rate)
                                    )
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_K_vars = autoLriGameModel\
                                .lri_balanced_player_game_all_pijk_upper_08(
                                    arr_pl_M_T_vars_init.copy(),
                                    pi_hp_plus=pi_hp_plus_elt, 
                                    pi_hp_minus=pi_hp_minus_elt,
                                    a=a, b=b,
                                    gamma_version=gamma_version,
                                    k_steps=k_steps, 
                                    learning_rate=learning_rate,
                                    p_i_j_ks=p_i_j_ks,
                                    utility_function_version=utility_function_version,
                                    path_to_save=path_to_save, 
                                    manual_debug=manual_debug, dbg=debug)
                                
        elif algo_name == ALGOS[2] or algo_name == ALGOS[3]:
            # 2: DETERMINIST, 3: RANDOM DETERMINIST
            print("*** ALGO: {} *** ".format(algo_name))
            random_determinist = False if algo_name == ALGOS[2] else True
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoDetGameModel.determinist_balanced_player_game(
                             arr_pl_M_T_vars_init.copy(),
                             pi_hp_plus=pi_hp_plus_elt, 
                             pi_hp_minus=pi_hp_minus_elt,
                             a=a, b=b,
                             gamma_version=gamma_version,
                             random_determinist=random_determinist,
                             used_storage=used_storage_det,
                             path_to_save=path_to_save, 
                             manual_debug=manual_debug, dbg=debug)
            
        elif algo_name in fct_aux.ALGO_NAMES_BF :
            # 0: BEST_BRUTE_FORCE (BF) , 1:BAD_BF, 2: MIDDLE_BF
            # execute tous les BF
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoBfGameModel\
                            .bf_balanced_player_game_ONE_ALGO(
                                arr_pl_M_T_vars_init.copy(),
                                algo_name=algo_name,
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                a=a, b=b,
                                gamma_version=gamma_version,
                                path_to_save=path_to_save, 
                                name_dir=name_dir, 
                                date_hhmm=date_hhmm,
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
                            
                           
        elif algo_name in fct_aux.ALGO_NAMES_NASH :
            # 0: "BEST-NASH", 1: "BAD-NASH", 2: "MIDDLE-NASH"
            print("*** ALGO: {} *** ".format(algo_name))
            Path(path_to_save).mkdir(parents=True, exist_ok=True)
            arr_M_T_vars = autoNashGameModel\
                            .nash_balanced_player_game_ONE_ALGO(
                                arr_pl_M_T_vars_init.copy(),
                                algo_name=algo_name,
                                pi_hp_plus=pi_hp_plus_elt, 
                                pi_hp_minus=pi_hp_minus_elt,
                                a=a, b=b,
                                gamma_version=gamma_version,
                                path_to_save=path_to_save, 
                                name_dir=name_dir, 
                                date_hhmm=date_hhmm,
                                manual_debug=manual_debug, 
                                dbg=debug)    
    
        
    print("NB_EXECUTION cpt={}".format(cpt))
#_________                   all ALGOs: fin                     ______________ 

#_________                      One algo: debut                 _______________
def execute_algos_used_Generated_instances_ONE_ALGO(arr_pl_M_T_vars_init,
                                                    algo_name,
                                                    pi_hp_plus=None,
                                                    pi_hp_minus=None,
                                                    a=1, b=1,
                                                    learning_rate=None,
                                                    k_steps=None,
                                                    name_dir="",
                                                    date_hhmm="",
                                                    gamma_version=1,
                                                    used_instances=True,
                                                    used_storage_det=True,
                                                    manual_debug=False, 
                                                    criteria_bf="Perf_t", 
                                                    debug=False):
    """
    execute algos by using generated instances if there exists or 
        by generating new instances
    
    date_hhmm="1041"
    algos=["LRI1"]
    
    """
    # directory to save  execution algos
    print("______ execution: {}, rate={}______".format( 
                algo_name, learning_rate))
    
    msg = "pi_hp_plus_"+str(pi_hp_plus)+"_pi_hp_minus_"+str(pi_hp_minus)
    path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name
                                    )
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    if algo_name == ALGOS_LRI[0]:
        # 0: LRI1
        print("*** ALGO: {} *** ".format(algo_name))
        utility_function_version = 1
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name, str(learning_rate)
                                )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_K_vars = autoLriGameModel\
                            .lri_balanced_player_game_all_pijk_upper_08(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                a=a, b=b,
                                gamma_version=gamma_version,
                                k_steps=k_steps, 
                                learning_rate=learning_rate,
                                p_i_j_ks=p_i_j_ks,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, 
                                manual_debug=manual_debug, dbg=debug)
    elif algo_name == ALGOS_LRI[1]:
        # LRI2
        print("*** ALGO: {} *** ".format(algo_name))
        utility_function_version = 2
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name, str(learning_rate)
                                )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_K_vars = autoLriGameModel\
                            .lri_balanced_player_game_all_pijk_upper_08(
                                arr_pl_M_T_vars_init.copy(),
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                a=a, b=b,
                                gamma_version=gamma_version,
                                k_steps=k_steps, 
                                learning_rate=learning_rate,
                                p_i_j_ks=p_i_j_ks,
                                utility_function_version=utility_function_version,
                                path_to_save=path_to_save, 
                                manual_debug=manual_debug, dbg=debug)
        
    elif algo_name in ALGOS_DET:
        # 2: DETERMINIST, 3: RANDOM DETERMINIST
        print("*** ALGO: {} *** ".format(algo_name))
        random_determinist = False if algo_name == ALGOS_DET[0] else True
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name
                                )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_vars = autoDetGameModel.determinist_balanced_player_game(
                         arr_pl_M_T_vars_init.copy(),
                         pi_hp_plus=pi_hp_plus, 
                         pi_hp_minus=pi_hp_minus,
                         a=a, b=b,
                         gamma_version=gamma_version,
                         random_determinist=random_determinist,
                         used_storage=used_storage_det,
                         path_to_save=path_to_save, 
                         manual_debug=manual_debug, dbg=debug)
        
    elif algo_name in fct_aux.ALGO_NAMES_BF:
        # 0: BEST_BRUTE_FORCE (BF) , 1:BAD_BF, 2: MIDDLE_BF
        print("*** ALGO: {} *** ".format(algo_name))
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name
                                    )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_vars = autoBfGameModel\
                            .bf_balanced_player_game_ONE_ALGO(
                                arr_pl_M_T_vars_init=arr_pl_M_T_vars_init.copy(),
                                algo_name=algo_name,
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                a=a, b=b,
                                gamma_version=gamma_version,
                                path_to_save=path_to_save, 
                                name_dir=name_dir, 
                                date_hhmm=date_hhmm,
                                manual_debug=manual_debug, 
                                criteria_bf=criteria_bf, dbg=debug)
                       
    elif algo_name == fct_aux.ALGO_NAMES_NASH[0] :
        # 0: "BEST-NASH", 1: "BAD-NASH", 2: "MIDDLE-NASH"
        print("*** ALGO: {} *** ".format(algo_name))
        path_to_save = os.path.join(name_dir, "simu_"+date_hhmm,
                                msg, algo_name
                                    )
        Path(path_to_save).mkdir(parents=True, exist_ok=True)
        arr_M_T_vars = autoNashGameModel\
                        .nash_balanced_player_game(
                            arr_pl_M_T_vars_init.copy(),
                            pi_hp_plus=pi_hp_plus, 
                            pi_hp_minus=pi_hp_minus,
                            gamma_version=gamma_version,
                            path_to_save=path_to_save, 
                            name_dir=name_dir, 
                            date_hhmm=date_hhmm,
                            manual_debug=manual_debug, 
                            dbg=debug)          
    
#_________                      One algo: fin                   _______________

#------------------------------------------------------------------------------
#                   definitions of unittests
#------------------------------------------------------------------------------

def test_execute_algos_used_Generated_instances():
    t_periods = 2
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    t_periods = 3
    setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    setA_m_players, setB_m_players, setC_m_players = 4, 2, 1
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    gamma_version = 1
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                            setA_m_players, setB_m_players, setC_m_players, 
                            t_periods, 
                            scenario1,
                            path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    
    algos = None
    if arr_pl_M_T_vars_init.shape[0] <= 16:
        algos = ["LRI1", "LRI2", "DETERMINIST"] \
                + fct_aux.ALGO_NAMES_NASH \
                + fct_aux.ALGO_NAMES_BF 
    else:
        algos = ["LRI1", "LRI2", "DETERMINIST"]
    k_steps = 25 #250
    learning_rates = [0.1]
    a=1; b=1
    pi_hp_plus=[10]; pi_hp_minus=[20]
    execute_algos_used_Generated_instances(
        arr_pl_M_T_vars_init, algos=algos, 
        k_steps=k_steps, 
        pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
        a=a, b=b,
        learning_rates=learning_rates, 
        gamma_version=gamma_version)
 
    
def test_debug_procedurale():
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario_name = 'scenario1'
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    t_periods = 3
    #setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3               # 12 players
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                            setA_m_players, setB_m_players, setC_m_players, 
                            t_periods, 
                            scenario1,
                            path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    
    """ _____ list of parameters with their type
    arr_pl_M_T_vars_init : array of shape (m_players, t_periods, k_steps, len(vars)) 
    scenario: string , scenario name of data
    name_dir: string,
    date_hhmm: string,
    t_periods: int,
    k_steps: integer,
    NB_REPEAT_K_MAX: integer,
    algos: list of string,
    learning_rates: list of integer,
    "tuple_pi_hp_plus_minus": tuple of a couple of pi_hp_plus, pi_hp_minus
    a: integer, enable to compute phi_hp_plus
    b: integer, enable to compute phi_hp_minus
    gamma_version: integer,
    used_instances: boolean,
    used_storage_det: boolean,
    manual_debug: boolean, 
    criteria_bf: string, 
    debug: boolean
    ____ """
    name_dir = "tests"
    date_hhmm = datetime.now().strftime("%d%m_%H%M") 
    
    # steps of learning
    k_steps = 25 #250
    k_steps = 5 if k_steps is None else k_steps
    NB_REPEAT_K_MAX = None
    fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
    
    # list of algos
    algos = None
    ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]\
            + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    algos = ALGOS if algos is None \
                  else algos
    # list of pi_hp_plus, pi_hp_minus
    a = 1; b = 1;
    pi_hp_plus = [10]
    pi_hp_minus = [20]
    tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
    
    # learning rate 
    learning_rates = [0.1]
    learning_rates = [0.01] \
            if learning_rates is None \
            else learning_rates
            
    gamma_versions = [1]
    utility_function_version = 1
    used_instances = True
    used_storage_det = True
    manual_debug = False 
    criteria_bf = "Perf_t" 
    debug = False
    
    dico_params = {
        "arr_pl_M_T_vars_init" : arr_pl_M_T_vars_init, 
        "scenario": scenario_name,
        "name_dir": name_dir,
        "date_hhmm": date_hhmm,
        "t_periods": t_periods,
        "k_steps": k_steps,
        "NB_REPEAT_K_MAX": NB_REPEAT_K_MAX,
        "algos": algos,
        "learning_rates": learning_rates,
        "tuple_pi_hp_plus_minus": tuple_pi_hp_plus_minus,
        "a": a, "b": b,
        "utility_function_version": utility_function_version,
        "gamma_versions": gamma_versions,
        "used_instances": used_instances,
        "used_storage_det": used_storage_det,
        "manual_debug": manual_debug, 
        "criteria_bf": criteria_bf, 
        "debug": debug
        }
    # params = define_parameters(dico_params)
    params = define_parameters_multi_gammaV(dico_params)
    print("define parameters finished")
    print(params[0])
    
    for param in params:
        print("param {} debut".format(param[1]))
        execute_algos_used_Generated_instances_ONE_ALGO(
            arr_pl_M_T_vars_init=param[0],
            algo_name=param[1],
            pi_hp_plus=param[2],
            pi_hp_minus=param[3],
            a=param[4],
            b=param[5],
            learning_rate=param[6],
            k_steps=param[7],
            name_dir=param[8],
            date_hhmm=param[9],
            gamma_version=param[10],
            used_instances=param[11],
            used_storage_det=param[12],
            manual_debug=param[13], 
            criteria_bf=param[14], 
            debug=param[15]
            )
            
        print("param {} FIN".format(param[1]))
    
def test_execute_algos_used_Generated_instances_MULTIPROCESS():
    prob_A_A = 0.8; prob_A_B = 0.2; prob_A_C = 0.0;
    prob_B_A = 0.3; prob_B_B = 0.4; prob_B_C = 0.3;
    prob_C_A = 0.1; prob_C_B = 0.2; prob_C_C = 0.7;
    scenario_name = 'scenario1'
    scenario1 = [(prob_A_A, prob_A_B, prob_A_C), 
                 (prob_B_A, prob_B_B, prob_B_C),
                 (prob_C_A, prob_C_B, prob_C_C)]
    
    t_periods = 4
    #setA_m_players, setB_m_players, setC_m_players = 10, 6, 5
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3               # 12 players
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = True #False #True
    gamma_version = 1
    
    arr_pl_M_T_vars_init = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE(
                            setA_m_players, setB_m_players, setC_m_players, 
                            t_periods, 
                            scenario1,
                            path_to_arr_pl_M_T, used_instances)
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    
    """ _____ list of parameters with their type
    arr_pl_M_T_vars_init : array of shape (m_players, t_periods, k_steps, len(vars)) 
    scenrio: string, scenario name of data
    name_dir: string,
    date_hhmm: string,
    t_periods: integer,
    k_steps: integer,
    NB_REPEAT_K_MAX: integer,
    algos: list of string,
    learning_rates: list of integer,
    "tuple_pi_hp_plus_minus": tuple of a couple of pi_hp_plus, pi_hp_minus
    gamma_version: integer,
    used_instances: boolean,
    used_storage_det: boolean,
    manual_debug: boolean, 
    criteria_bf: string, 
    debug: boolean
    ____ """
    name_dir = "tests"
    date_hhmm = datetime.now().strftime("%d%m_%H%M") 
    
    # steps of learning
    k_steps = 25 #250
    k_steps = 5 if k_steps is None else k_steps
    NB_REPEAT_K_MAX = None
    fct_aux.NB_REPEAT_K_MAX = 3 if NB_REPEAT_K_MAX is None else NB_REPEAT_K_MAX
    
    # list of algos
    algos = None
    ALGOS = ["LRI1", "LRI2", "DETERMINIST", "RD-DETERMINIST"]\
            + fct_aux.ALGO_NAMES_BF + fct_aux.ALGO_NAMES_NASH
    algos = ALGOS if algos is None \
                  else algos
    # list of pi_hp_plus, pi_hp_minus
    a = 1; b = 1;
    pi_hp_plus = [10]
    pi_hp_minus = [20]
    tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
    
    # learning rate 
    learning_rates = [0.1]
    learning_rates = [0.01] \
            if learning_rates is None \
            else learning_rates
            
    gamma_versions = [1]
    utility_function_version = 1
    used_instances = True
    used_storage_det = True
    manual_debug = False 
    criteria_bf = "Perf_t" 
    debug = False
    
    dico_params = {
        "arr_pl_M_T_vars_init" : arr_pl_M_T_vars_init, 
        "scenario": scenario_name,
        "name_dir": name_dir,
        "date_hhmm": date_hhmm,
        "t_periods": t_periods,
        "k_steps": k_steps,
        "NB_REPEAT_K_MAX": NB_REPEAT_K_MAX,
        "algos": algos,
        "learning_rates": learning_rates,
        "tuple_pi_hp_plus_minus": tuple_pi_hp_plus_minus,
        "a": a, "b": b,
        "gamma_versions": gamma_versions,
        "used_instances": used_instances,
        "used_storage_det": used_storage_det,
        "manual_debug": manual_debug, 
        "criteria_bf": criteria_bf, 
        "debug": debug
        }
    #params = define_parameters(dico_params)
    params = define_parameters_multi_gammaV(dico_params)
    print("define parameters finished")
    
    # multi processing execution
    p = mp.Pool(mp.cpu_count()-1)
    p.starmap(execute_algos_used_Generated_instances_ONE_ALGO, 
              params)
    # multi processing execution
    
    return params
#------------------------------------------------------------------------------
#                   definitions of unittests
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#           execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
    ti = time.time()
    
    boolean_execute = True #False
    
    if boolean_execute:
        #test_execute_algos_used_Generated_instances()
        #test_debug_procedurale()
        test_execute_algos_used_Generated_instances_MULTIPROCESS()
    
    print("runtime = {}".format(time.time() - ti))