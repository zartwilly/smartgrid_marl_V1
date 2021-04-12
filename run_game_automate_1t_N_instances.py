# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:10:48 2021

@author: jwehounou
"""

import os
import time
import numpy as np
import execution_game_automate_4_all_t as autoExeGame4T
import fonctions_auxiliaires as fct_aux
import visu_bkh_automate_1t_N_instances as autoVizGame1TnInstances

from pathlib import Path

if __name__ == "__main__":
    ti = time.time()
    
    date_hhmm="DDMM_HHMM"
    t_periods = 1 #50 #30 #35 #55 #117 #15 #3
    k_steps = 250 #5000 #2000 #50 #250
    NB_REPEAT_K_MAX= 10 #3 #15 #30
    learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
    fct_aux.N_DECIMALS = 8
    a = 1; b = 1;
    pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
    pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
    fct_aux.PI_0_PLUS_INIT = 4 #20 #4
    fct_aux.PI_0_MINUS_INIT = 3 #10 #3
    NB_INSTANCES = 50 #50
    
    algos = ["LRI1", "LRI2", "DETERMINIST"] \
            + fct_aux.ALGO_NAMES_NASH \
            + fct_aux.ALGO_NAMES_BF
            
    # ---- initialization of variables for generating instances ----
    setA_m_players, setB_m_players, setC_m_players = 15, 10, 10                # 35 players 
    setA_m_players, setB_m_players, setC_m_players = 10, 6, 5                  # 21 players 
    setA_m_players, setB_m_players, setC_m_players = 8, 4, 4                   # 16 players
    setA_m_players, setB_m_players, setC_m_players = 6, 3, 3                   # 12 players
    scenario_name = "scenario0"
    
    name_dir = "tests"
    gamma_versions = [3] #[0,1,2,3,4]
    for gamma_version in gamma_versions:
        
        # ----   execution of 50 instances    ----
        criteria_bf = "Perf_t"
        used_storage_det = True #False #True
        manual_debug = False #True #False #True
        name_dir_oneperiod = os.path.join(
                                name_dir,
                                "OnePeriod_"+str(NB_INSTANCES)+"instances"+"GammaV"+str(gamma_version))
        
        for i in range(0, NB_INSTANCES):
            used_instances = False #True
            arr_pl_M_T_vars_init = None
            path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
            arr_pl_M_T_vars_init \
                = fct_aux.get_or_create_instance_Pi_Ci_one_period(
                    setA_m_players, setB_m_players, setC_m_players, 
                    t_periods, 
                    scenario_name,
                    path_to_arr_pl_M_T, used_instances)
            
            # ---- execution of various data  ----
            date_hhmm_new = "_".join([date_hhmm, str(i), "t", str(t_periods)])
            
            autoExeGame4T\
                .execute_algos_used_Generated_instances(
                    arr_pl_M_T_vars_init, 
                    name_dir = name_dir_oneperiod,
                    date_hhmm = date_hhmm_new,
                    k_steps = k_steps,
                    NB_REPEAT_K_MAX = NB_REPEAT_K_MAX,
                    algos = algos,
                    learning_rates = learning_rates,
                    pi_hp_plus = pi_hp_plus,
                    pi_hp_minus = pi_hp_minus,
                    a = a, b = b,
                    gamma_version=gamma_version,
                    used_instances = used_instances,
                    used_storage_det = used_storage_det,
                    manual_debug = manual_debug, 
                    criteria_bf = criteria_bf, 
                    debug = False
                    )
                
    print("runtime = {}".format(time.time() - ti))
    
    # gamma_versions = [0,1,2,3,4] #1 #2
    
    # name_dir = "tests"
    # name_dir_oneperiod = os.path.join(
    #                         name_dir,
    #                         "OnePeriod_50instances"+"GammaV"+str(gamma_version))
    
    
    # pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
    # pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
    # fct_aux.PI_0_PLUS_INIT = 4 #20 #4
    # fct_aux.PI_0_MINUS_INIT = 3 #10 #3
    
    # algos = ["LRI1", "LRI2", "DETERMINIST"] \
    #         + fct_aux.ALGO_NAMES_NASH \
    #         + fct_aux.ALGO_NAMES_BF
            
    
    # Visualisation = True #False, True
    
    # scenario = None
    
    # # ---- initialization of variables for generating instances ----
    # setA_m_players, setB_m_players, setC_m_players = 15, 10, 10                # 35 players 
    # setA_m_players, setB_m_players, setC_m_players = 10, 6, 5                  # 21 players 
    # setA_m_players, setB_m_players, setC_m_players = 8, 4, 4                   # 16 players
    # setA_m_players, setB_m_players, setC_m_players = 6, 3, 3                   # 12 players
    
        
    # # ----   execution of 50 instances    ----
    # criteria_bf = "Perf_t"
    # used_storage_det = True #False #True
    # manual_debug = False #True #False #True
    # NB_INSTANCES = 50
    # for i in range(0, NB_INSTANCES):
    #     fct_aux.MANUEL_DBG_GAMMA_I = np.random.randint(low=2, high=21, size=1)[0]
    #     # ---- generation of data into array ----
    #     used_instances = False #True
    #     arr_pl_M_T_vars_init = None
    #     path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
    #     arr_pl_M_T_vars_init \
    #         = fct_aux.get_or_create_instance_Pi_Ci_one_period(
    #             setA_m_players, setB_m_players, setC_m_players, 
    #             t_periods, 
    #             scenario,
    #             path_to_arr_pl_M_T, used_instances)
            
    #     # ---- execution of various data  ----
    #     date_hhmm_new = "_".join([date_hhmm, str(i), "t", str(t_periods)])
        
    #     autoExeGame4T\
    #         .execute_algos_used_Generated_instances(
    #             arr_pl_M_T_vars_init, 
    #             name_dir = name_dir_oneperiod,
    #             date_hhmm = date_hhmm_new,
    #             k_steps = k_steps,
    #             NB_REPEAT_K_MAX = NB_REPEAT_K_MAX,
    #             algos = algos,
    #             learning_rates = learning_rates,
    #             pi_hp_plus = pi_hp_plus,
    #             pi_hp_minus = pi_hp_minus,
    #             gamma_version=gamma_version,
    #             used_instances = used_instances,
    #             used_storage_det = used_storage_det,
    #             manual_debug = manual_debug, 
    #             criteria_bf = criteria_bf, 
    #             debug = False
    #             )
     
    # # ----    visualisation     ----
    # algos_4_showing=["DETERMINIST", "LRI1", "LRI2"] \
    #                 + fct_aux.ALGO_NAMES_BF \
    #                 + fct_aux.ALGO_NAMES_NASH
    # algos_4_no_learning=["DETERMINIST","RD-DETERMINIST"] \
    #                      + fct_aux.ALGO_NAMES_BF \
    #                      + fct_aux.ALGO_NAMES_NASH
    # tuple_paths, \
    # prices_new, \
    # algos_new, \
    # learning_rates_new, \
    # path_2_best_learning_steps = autoVizGame1TnInstances\
    #                                 .get_tuple_paths_of_arrays_4_many_simu(
    #                                     name_dir=name_dir_oneperiod,
    #                                     algos_4_showing=algos_4_showing,
    #                                     algos_4_no_learning=algos_4_no_learning)
    # print("tuple_paths = {}".format(len(tuple_paths)))

    # t = 0
    # k_steps_args = k_steps #250
    # algos_4_learning = ["LRI1", "LRI2"]
    # df_arr_M_T_Ks, \
    # df_ben_cst_M_T_K, \
    # df_b0_c0_pisg_pi0_T_K, \
    # df_B_C_BB_CC_RU_M \
    #     = autoVizGame1TnInstances\
    #         .get_array_turn_df_for_t(tuple_paths, t=t, k_steps_args=k_steps_args, 
    #                               algos_4_no_learning=algos_4_no_learning, 
    #                               algos_4_learning=algos_4_learning)
    # print("size t={}, df_arr_M_T_Ks={} Mo, df_ben_cst_M_T_K={} Mo, df_b0_c0_pisg_pi0_T_K={} Mo, df_B_C_BB_CC_RU_M={} Mo".format(
    #             t, 
    #           round(df_arr_M_T_Ks.memory_usage().sum()/(1024*1024), 2),  
    #           round(df_ben_cst_M_T_K.memory_usage().sum()/(1024*1024), 2),
    #           round(df_b0_c0_pisg_pi0_T_K.memory_usage().sum()/(1024*1024), 2),
    #           round(df_B_C_BB_CC_RU_M.memory_usage().sum()/(1024*1024), 4)
    #           ))
    # print("get_array_turn_df_for_t: TERMINE")
    
    # ti_ = time.time()
    # df_ben_cst_M_T_Kmax = autoVizGame1TnInstances\
    #                         .create_dataframe_kmax(df_ben_cst_M_T_K)
    # print("create_dataframe_k_max, df_ben_cst_M_T_kmax:{}, runtime={} TERMINE"\
    #       .format(df_ben_cst_M_T_Kmax.shape, time.time()-ti_ ))
    
    # ti_ = time.time()
    # df_algo_instance_state_moyVi = autoVizGame1TnInstances\
    #                                 .create_dataframe_mean_Vi_for(
    #                                     df_ben_cst_M_T_Kmax)
    # print("create_dataframe_mean_Vi: df_algo_instance_state_moyVi={}, runtime={},TERMINE"\
    #       .format(df_algo_instance_state_moyVi.shape, time.time()-ti_))
    
    # path_to_save = os.path.join(name_dir_oneperiod, "AVERAGE_RESULTS")
    # Path(path_to_save).mkdir(parents=True, exist_ok=True)
    # autoVizGame1TnInstances.group_plot_on_panel(
    #                     df_arr_M_T_Ks, 
    #                     df_ben_cst_M_T_K, 
    #                     df_B_C_BB_CC_RU_M,
    #                     df_b0_c0_pisg_pi0_T_K,
    #                     df_algo_instance_state_moyVi,
    #                     t, k_steps_args, path_to_save,
    #                     path_2_best_learning_steps, 
    #                     autoVizGame1TnInstances.NAME_RESULT_SHOW_VARS)

    # print("runtime = {}".format(time.time() - ti))