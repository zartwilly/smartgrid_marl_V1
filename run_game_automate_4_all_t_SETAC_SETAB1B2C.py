# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:08:29 2021

@author: jwehounou
"""

import os
import time
import execution_game_automate_4_all_t as autoExeGame4T
import fonctions_auxiliaires as fct_aux
import itertools as it

def generate_arr_M_T_vars(doc_VALUES, dico_scenario, 
                          setA_m_players_12, setB1_m_players_12, 
                          setB2_m_players_12, setC_m_players_12, 
                          t_periods, scenario_name,
                          path_to_arr_pl_M_T, used_instances):
    arr_pl_M_T_vars_init = None
    if doc_VALUES==15:
        if scenario_name in ["scenario1", "scenario2"]:
            arr_pl_M_T_vars_init \
                = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc15(
                            setA_m_players_12, setB1_m_players_12, 
                            setB2_m_players_12, setC_m_players_12, 
                            t_periods, 
                            dico_scenario[scenario_name],
                            scenario_name,
                            path_to_arr_pl_M_T, used_instances)
        else:
            arr_pl_M_T_vars_init \
                = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC(
                        setA_m_players_0, setC_m_players_0, 
                        t_periods, 
                        dico_scenario[scenario_name],
                        scenario_name,
                        path_to_arr_pl_M_T, used_instances)
    elif doc_VALUES==16:
        if scenario_name in ["scenario1", "scenario2"]:
            arr_pl_M_T_vars_init \
                = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc16(
                            setA_m_players_12, setB1_m_players_12, 
                            setB2_m_players_12, setC_m_players_12, 
                            t_periods, 
                            dico_scenario[scenario_name],
                            scenario_name,
                            path_to_arr_pl_M_T, used_instances)
        else:
            arr_pl_M_T_vars_init \
                = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC(
                        setA_m_players_0, setC_m_players_0, 
                        t_periods, 
                        dico_scenario[scenario_name],
                        scenario_name,
                        path_to_arr_pl_M_T, used_instances)
    elif doc_VALUES==17:
        print("scenario_name={}, scenario={}".format(scenario_name, dico_scenario[scenario_name]))
        if scenario_name in ["scenario1", "scenario2"]:
            arr_pl_M_T_vars_init \
                = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc17(
                            setA_m_players_12, setB1_m_players_12, 
                            setB2_m_players_12, setC_m_players_12, 
                            t_periods, 
                            dico_scenario[scenario_name],
                            scenario_name,
                            path_to_arr_pl_M_T, used_instances)
        else:
            arr_pl_M_T_vars_init \
                = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC(
                        setA_m_players_0, setC_m_players_0, 
                        t_periods, 
                        dico_scenario[scenario_name],
                        scenario_name,
                        path_to_arr_pl_M_T, used_instances)
      
    return arr_pl_M_T_vars_init

if __name__ == "__main__":
    ti = time.time()
    
    # _____                     scenarios --> debut                 __________
    doc_VALUES = 17; # 15: doc version 15, 16: doc version 16, 17: doc version 17
    dico_scenario = None
    root_doc_VALUES = None
    if doc_VALUES == 15:
        prob_A_A = 0.6; prob_A_C = 0.4;
        prob_C_A = 0.4; prob_C_C = 0.6;
        scenario0 = [(prob_A_A, prob_A_C), 
                    (prob_C_A, prob_C_C)] 
        
        prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
        prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
        prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
        prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
        scenario1 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                     (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                     (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                     (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        
        prob_A_A = 0.8; prob_A_B1 = 0.2; prob_A_B2 = 0.0; prob_A_C = 0.0;
        prob_B1_A = 0.8; prob_B1_B1 = 0.2; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
        prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.2; prob_B2_C = 0.8;
        prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.2; prob_C_C = 0.8
        scenario2 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                     (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                     (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                     (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        
        root_doc_VALUES = "Doc{}".format(doc_VALUES)
        scenario_names = {"scenario0", "scenario1", "scenario2"}
        dico_scenario = {"scenario0": scenario0,
                         "scenario1": scenario1, 
                         "scenario2": scenario2}
    elif doc_VALUES == 16:
        prob_A_A = 0.6; prob_A_C = 0.4;
        prob_C_A = 0.4; prob_C_C = 0.6;
        scenario0 = [(prob_A_A, prob_A_C), 
                    (prob_C_A, prob_C_C)] 
        
        prob_A_A = 0.4; prob_A_B1 = 0.6; prob_A_B2 = 0.0; prob_A_C = 0.0;
        prob_B1_A = 0.4; prob_B1_B1 = 0.6; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
        prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.5; prob_B2_C = 0.5;
        prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.5; prob_C_C = 0.5 
        scenario1 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                     (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                     (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                     (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        
        prob_A_A = 0.2; prob_A_B1 = 0.8; prob_A_B2 = 0.0; prob_A_C = 0.0;
        prob_B1_A = 0.8; prob_B1_B1 = 0.2; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
        prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.2; prob_B2_C = 0.8;
        prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.8; prob_C_C = 0.2
        scenario2 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                     (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                     (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                     (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        
        root_doc_VALUES = "Doc{}".format(doc_VALUES)
        scenario_names = {"scenario0", "scenario1", "scenario2"}
        dico_scenario = {"scenario0": scenario0,
                         "scenario1": scenario1, 
                         "scenario2": scenario2}
    elif doc_VALUES == 17:
        
        setA_m_players_12 = 8; setB1_m_players_12 = 5; 
        setB2_m_players_12 = 5; setC_m_players_12 = 8;                         # 26 players
        setA_m_players_0 = 10; setC_m_players_0 = 10;                          # 20 players
            
        prob_A_A = 0.6; prob_A_C = 0.4;
        prob_C_A = 0.4; prob_C_C = 0.6;
        scenario0 = [(prob_A_A, prob_A_C), 
                    (prob_C_A, prob_C_C)] 
        
        prob_A_A = 0.4; prob_A_B1 = 0.6; prob_A_B2 = 0.0; prob_A_C = 0.0;
        prob_B1_A = 0.4; prob_B1_B1 = 0.6; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
        prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.5; prob_B2_C = 0.5;
        prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.5; prob_C_C = 0.5 
        scenario1 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                     (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                     (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                     (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        
        prob_A_A = 0.2; prob_A_B1 = 0.8; prob_A_B2 = 0.0; prob_A_C = 0.0;
        prob_B1_A = 0.8; prob_B1_B1 = 0.2; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
        prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.2; prob_B2_C = 0.8;
        prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.8; prob_C_C = 0.2
        scenario2 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                     (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                     (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                     (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
        
        root_doc_VALUES = "Doc{}".format(doc_VALUES)
        scenario_names = {"scenario0", "scenario1", "scenario2"}
        dico_scenario = {"scenario0": scenario0,
                         "scenario1": scenario1, 
                         "scenario2": scenario2}
        
    # _____                     scenarios --> fin                   __________
    # _____                     gamma_version --> debut             __________
    #2 #1 #3: gamma_i_min #4: square_root
    gamma_versions = [0,1,2,3,4]
    gamma_versions = [1,3]
    gamma_versions = [0,2,3]
    gamma_versions = [4]
    gamma_versions = [0,1,2,3,4]
    gamma_versions = [0,1]
    # _____                     gamma_version --> fin               __________
    
    
    debug_all_periods = True #False #True #False #False #True
    debug_one_period = not debug_all_periods
    
    name_dir="tests"
    
    pi_hp_plus, pi_hp_minus, tuple_pi_hp_plus_minus = None, None, None
    setA_m_players, setC_m_players = None, None
    t_periods, k_steps, NB_REPEAT_K_MAX = None, None, None
    learning_rates = None, None, None
    date_hhmm, Visualisation = None, None
    used_storage_det=True
    used_instances=True
    criteria_bf="Perf_t" # "In_sg_Out_sg"
    dbg_234_players = None
    arr_pl_M_T_vars_init = None
    path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
    
    PHI = None # A1B1: lineaire, A1.2B0.8: convexe/concave
    
    if debug_all_periods:
        nb_periods = None
        # ---- new constances simu_DDMM_HHMM --- **** debug *****
        date_hhmm = "DDMM_HHMM"
        t_periods = 50 #20 #50 #10 #30 #50 #30 #35 #55 #117 #15 #3
        k_steps = 5 #250 #5 #100 #250 #5000 #2000 #50 #250
        NB_REPEAT_K_MAX= 1#10 #3 #15 #30
        learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
        fct_aux.N_DECIMALS = 8
        
        a = 1; b = 1; PHI = "A{}B{}".format(a,b)
        pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
        fct_aux.PI_0_PLUS_INIT = 4 #10 #4
        fct_aux.PI_0_MINUS_INIT = 3 #20 #3
        tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
        
        algos = ["LRI1", "LRI2", "DETERMINIST"]
        
        used_storage_det= True #False #True
        manual_debug = False #True #False #True
     
    elif debug_one_period:
        nb_periods = 0
        # ---- new constances simu_DDMM_HHMM  ONE PERIOD t = 0 --- **** debug *****
        date_hhmm="DDMM_HHMM"
        t_periods = 1 #50 #30 #35 #55 #117 #15 #3
        k_steps = 250 #5000 #2000 #50 #250
        NB_REPEAT_K_MAX= 10 #3 #15 #30
        learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
        fct_aux.N_DECIMALS = 8
        
        a = 1; b = 1; PHI = "A{}B{}".format(a,b)
        pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
        fct_aux.PI_0_PLUS_INIT = 4 #10 #4
        fct_aux.PI_0_MINUS_INIT = 3 #20 #3
        tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
        
        algos = ["LRI1", "LRI2", "DETERMINIST"] 
        
        used_storage_det= True #False #True
        manual_debug = False #True #False #True
        # gamma_versions = [1, 3] # 2
        Visualisation = True #False, True
        
        used_instances = True
    
    else:
        nb_periods = None
        # ---- new constances simu_2306_2206 --- **** debug ***** 
        date_hhmm="2306_2206"
        t_periods = 110
        k_steps = 1000
        NB_REPEAT_K_MAX = 15 #30
        learning_rates = [0.1] #[0.01] #[0.0001]
        
        a = 1; b = 1; PHI = "A{}B{}".format(a,b)
        pi_hp_plus = [0.2*pow(10,-3)] #[5, 15]
        pi_hp_minus = [0.33] #[15, 5]
        fct_aux.PI_0_PLUS_INIT = 4 #10 #4
        fct_aux.PI_0_MINUS_INIT = 3 #20 #3
        tuple_pi_hp_plus_minus = tuple(zip(pi_hp_plus, pi_hp_minus))
       
        used_storage_det=True #False #True
        manual_debug = False #True
    
    setX = ""
    if set(dico_scenario.keys()) == {"scenario0", "scenario1", "scenario2"} \
        or set(dico_scenario.keys()) == {"scenario0", "scenario1"} \
        or set(dico_scenario.keys()) == {"scenario0", "scenario2"} :
        setX = "setACsetAB1B2C"
    elif set(dico_scenario.keys()) == {"scenario0"}:
        setX = "setAC"
    elif set(dico_scenario.keys()) == {"scenario1", "scenario2"} \
        or set(dico_scenario.keys()) == {"scenario1"} \
        or set(dico_scenario.keys()) == {"scenario2"}:
        setX = "setAB1B2C"

    gamma_Vxs = set(gamma_versions)
    root_gamVersion = "gamma"
    for gamma_version in gamma_Vxs:
        root_gamVersion = root_gamVersion + "_V"+str(gamma_version)
    root_gamVersion = PHI + root_doc_VALUES + root_gamVersion
    # "AxBxDocYgamma_V0_V1_V2_V3_V4_T20_kstep250_setACsetAB1B2C"
    name_execution = root_gamVersion \
                        + "_" + "T"+str(t_periods) \
                        + "_" + "ksteps" + str(k_steps) \
                        + "_" + setX 
                        
    for scenario_name, gamma_version in it.product(list(dico_scenario.keys()), 
                                                   gamma_versions):
        date_hhmm_new = "_".join([date_hhmm, scenario_name, 
                              "".join(["T", str(t_periods),
                                "".join(["gammaV", str(gamma_version)])])])
        
        used_instances = True
        arr_pl_M_T_vars_init = None
        
        arr_pl_M_T_vars_init \
           = generate_arr_M_T_vars(doc_VALUES, dico_scenario, 
                                   setA_m_players_12, setB1_m_players_12, 
                                   setB2_m_players_12, setC_m_players_12, 
                                   t_periods, scenario_name,
                                   path_to_arr_pl_M_T, used_instances)
    
        autoExeGame4T.execute_algos_used_Generated_instances(
                    arr_pl_M_T_vars_init, 
                    name_dir = os.path.join(name_dir,name_execution),
                    date_hhmm = date_hhmm_new,
                    k_steps = k_steps,
                    NB_REPEAT_K_MAX = NB_REPEAT_K_MAX,
                    algos = algos,
                    learning_rates = learning_rates,
                    pi_hp_plus = pi_hp_plus,
                    pi_hp_minus = pi_hp_minus,
                    a = a,
                    b = b,
                    gamma_version = gamma_version,
                    used_instances = used_instances,
                    used_storage_det = used_storage_det,
                    manual_debug = manual_debug, 
                    criteria_bf = criteria_bf, 
                    debug = False
                    )                
    
    
    print("runtime = {}".format(time.time() - ti))
    
    # # _____                     scenarios --> debut                 __________
    
    # prob_A_A = 0.6; prob_A_C = 0.4;
    # prob_C_A = 0.4; prob_C_C = 0.6;
    # scenario0 = [(prob_A_A, prob_A_C), 
    #             (prob_C_A, prob_C_C)] 
    
    # prob_A_A = 0.6; prob_A_B1 = 0.4; prob_A_B2 = 0.0; prob_A_C = 0.0;
    # prob_B1_A = 0.6; prob_B1_B1 = 0.4; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
    # prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.4; prob_B2_C = 0.6;
    # prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.4; prob_C_C = 0.6 
    # scenario1 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
    #              (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
    #              (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
    #              (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
    
    # prob_A_A = 0.8; prob_A_B1 = 0.2; prob_A_B2 = 0.0; prob_A_C = 0.0;
    # prob_B1_A = 0.8; prob_B1_B1 = 0.2; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
    # prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.2; prob_B2_C = 0.8;
    # prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.2; prob_C_C = 0.8
    # scenario2 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
    #              (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
    #              (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
    #              (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
    
    
    # dico_scenario = {"scenario0": scenario0,
    #                  "scenario1": scenario1, 
    #                  "scenario2": scenario2}
    # # _____                     scenarios --> fin                   __________
    
    # # _____                     gamma_version --> debut             __________
    # #2 #1 #3: gamma_i_min #4: square_root
    # gamma_versions = [0,1,2,3,4]
    # #gamma_versions = [1,3]
    # # _____                     gamma_version --> fin               __________
    
    # # _____                    players by sets --> debut             __________
    # # ---- initialization of variables for generating instances ----
    # setA_m_players_12 = 10; setB1_m_players_12 = 3; 
    # setB2_m_players_12 = 5; setC_m_players_12 = 8;                             # 26 players
    # setA_m_players_0 = 10; setC_m_players_0 = 10;                              # 20 players
    
    # # _____                    players by sets --> fin               __________
    
    # debug_all_periods = True #False #True #False #False #True
    # debug_one_period = not debug_all_periods
    
    # name_dir="tests"
    
    # a, b = None, None
    # pi_hp_plus, pi_hp_minus = None, None
    # t_periods, k_steps, NB_REPEAT_K_MAX = None, None, None
    # learning_rates = None, None, None
    # date_hhmm, Visualisation = None, None
    # used_storage_det = True
    # criteria_bf="Perf_t" # "In_sg_Out_sg"
    # dbg_234_players = None
    # arr_pl_M_T_vars_init = None
    # path_to_arr_pl_M_T = os.path.join(*[name_dir, "AUTOMATE_INSTANCES_GAMES"])
    
    # if debug_all_periods:
    #     nb_periods = None
    #     # ---- new constances simu_DDMM_HHMM --- **** debug *****
    #     date_hhmm = "DDMM_HHMM"
    #     t_periods = 3#4 #10 #4 #10 #30 #50 #30 #35 #55 #117 #15 #3
    #     k_steps = 25 #250 #250 #100 #250 #5000 #2000 #50 #250
    #     NB_REPEAT_K_MAX= 10 #3 #15 #30
    #     learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
    #     fct_aux.N_DECIMALS = 8
        
    #     a, b = 1, 1
    #     pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
    #     pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
    #     fct_aux.PI_0_PLUS_INIT = 4 # 20
    #     fct_aux.PI_0_MINUS_INIT = 3 # 10
        
    #     algos = ["LRI1", "LRI2", "DETERMINIST"]
        
    #     dbg_234_players = False #True #False
    #     used_storage_det= True #False #True
    #     manual_debug = False #True #False #True
    #     Visualisation = True #False, True
    
    # elif debug_one_period:
    #     nb_periods = 0
    #     # ---- new constances simu_DDMM_HHMM  ONE PERIOD t = 0 --- **** debug *****
    #     date_hhmm="DDMM_HHMM"
    #     t_periods = 1 #50 #30 #35 #55 #117 #15 #3
    #     k_steps = 250 #5000 #2000 #50 #250
    #     NB_REPEAT_K_MAX= 10 #3 #15 #30
    #     learning_rates = [0.1]#[0.1] #[0.001]#[0.00001] #[0.01] #[0.0001]
    #     fct_aux.N_DECIMALS = 8
        
    #     a, b = 1, 1
    #     pi_hp_plus = [10] #[10] #[0.2*pow(10,-3)] #[5, 15]
    #     pi_hp_minus = [20] #[20] #[0.33] #[15, 5]
    #     fct_aux.PI_0_PLUS_INIT = 4 # 20
    #     fct_aux.PI_0_MINUS_INIT = 3 # 10
        
    #     algos = ["LRI1", "LRI2", "DETERMINIST"] 
        
    #     dbg_234_players = False #True #False
    #     used_storage_det= True #False #True
    #     manual_debug = True#False #True #False #True
    #     gamma_version = 1 # 2
    #     Visualisation = True #False, True
        
    # else:
    #     nb_periods = None
    #     # ---- new constances simu_2306_2206 --- **** debug ***** 
    #     date_hhmm="2306_2206"
    #     t_periods = 110
    #     k_steps = 1000
    #     NB_REPEAT_K_MAX = 15 #30
    #     learning_rates = [0.1] #[0.01] #[0.0001]
       
    #     a, b = 1, 1
    #     pi_hp_plus = [0.2*pow(10,-3)] #[5, 15]
    #     pi_hp_minus = [0.33] #[15, 5]
    #     fct_aux.PI_0_PLUS_INIT = 4 # 20
    #     fct_aux.PI_0_MINUS_INIT = 3 # 10
       
    #     dbg_234_players = False
    #     used_storage_det= True #False #True
    #     manual_debug = False #True
    #     Visualisation = True #False, True
       
        
    # setX = ""
    # if set(dico_scenario.keys()) == {"scenario0", "scenario1", "scenario2"} \
    #     or set(dico_scenario.keys()) == {"scenario0", "scenario1"} \
    #     or set(dico_scenario.keys()) == {"scenario0", "scenario2"} :
    #     setX = "setACsetAB1B2C"
    # elif set(dico_scenario.keys()) == {"scenario0"}:
    #     setX = "setAC"
    # elif set(dico_scenario.keys()) == {"scenario1", "scenario2"} \
    #     or set(dico_scenario.keys()) == {"scenario1"} \
    #     or set(dico_scenario.keys()) == {"scenario2"}:
    #     setX = "setAB1B2C"

    # gamma_Vxs = set(gamma_versions)
    # root_gamVersion = "gamma"
    # for gamma_version in gamma_Vxs:
    #     root_gamVersion =  root_gamVersion + "_V"+str(gamma_version)
    
    # "gamma_V0_V1_V2_V3_V4_T20_kstep250_setACsetAB1B2C"
    # name_execution = root_gamVersion \
    #                     + "_" + "T"+str(t_periods) \
    #                     + "_" + "ksteps" + str(k_steps) \
    #                     + "_" + setX 
       
    # for scenario_name, gamma_version in it.product( list(dico_scenario.keys()), 
    #                                           gamma_versions):
    #     date_hhmm_new = "_".join([date_hhmm, scenario_name, 
    #                           "".join(["T", str(t_periods),
    #                             "".join(["gammaV", str(gamma_version)])])])
        
    #     used_instances = True
    #     arr_pl_M_T_vars_init = None
    #     if scenario_name in ["scenario1", "scenario2"]:
    #         arr_pl_M_T_vars_init \
    #             = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C(
    #                         setA_m_players_12, setB1_m_players_12, 
    #                         setB2_m_players_12, setC_m_players_12, 
    #                         t_periods, 
    #                         dico_scenario[scenario_name],
    #                         scenario_name,
    #                         path_to_arr_pl_M_T, used_instances)
    #     else:
    #         arr_pl_M_T_vars_init \
    #             = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAC(
    #                     setA_m_players_0, setC_m_players_0, 
    #                     t_periods, 
    #                     dico_scenario[scenario_name],
    #                     scenario_name,
    #                     path_to_arr_pl_M_T, used_instances)
    
    #     autoExeGame4T.execute_algos_used_Generated_instances(
    #                 arr_pl_M_T_vars_init, 
    #                 name_dir = os.path.join(name_dir,name_execution),
    #                 date_hhmm = date_hhmm_new,
    #                 k_steps = k_steps,
    #                 NB_REPEAT_K_MAX = NB_REPEAT_K_MAX,
    #                 algos = algos,
    #                 learning_rates = learning_rates,
    #                 pi_hp_plus = pi_hp_plus,
    #                 pi_hp_minus = pi_hp_minus,
    #                 a = a,
    #                 b = b,
    #                 gamma_version = gamma_version,
    #                 used_instances = used_instances,
    #                 used_storage_det = used_storage_det,
    #                 manual_debug = manual_debug, 
    #                 criteria_bf = criteria_bf, 
    #                 debug = False
    #                 )
    # print("Procedural running time ={}".format(time.time()-ti))
    
        
    # print("runtime = {}".format(time.time() - ti))