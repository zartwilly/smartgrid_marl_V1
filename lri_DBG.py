# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 08:44:45 2021

@author: jwehounou
"""
import os
import time

import numpy as np
import pandas as pd
import smartgrids_players as players
import fonctions_auxiliaires as fct_aux

from pathlib import Path
from datetime import datetime

VARS = ["Ci", "Pi", "Si", "Si_max", "gamma_i", 
        "prod_i", "cons_i", "r_i", "state_i", "mode_i",
        "Profili", "Casei", "R_i_old", "Si_old", 
        "balanced_pl_i", "formule", "Si_minus",
        "Si_plus", "u_i", "bg_i",
        "S1_p_i_j_k", "S2_p_i_j_k", 
        "non_playing_players", "set", 
        "ben_i", "cst_i", "B_i", "C_i", "BB_i", "CC_i", "RU_i", 
        "pi_hp_plus_t", "pi_hp_minus_t", "pi_sg_plus_t", "pi_sg_minus_t", 
        "pi_0_plus_t", "pi_0_minus_t", "b0_t", "c0_t"]
INDEX_ATTRS = dict()
for i, var in enumerate(VARS):
    INDEX_ATTRS[var] = i
    

###############################################################################
#
#                   definition  des fonctions annexes
#
###############################################################################
# _____________________________________________________________________________
#           compute starting prices at t: debut
# _____________________________________________________________________________
# _________         compute q_t_minus, q_t_plus:  debut        ________________
def compute_upper_bound_quantity_energy(arr_M_K_vars, k):
    """
    compute bought upper bound quantity energy q_t_minus
        and sold upper bound quantity energy q_t_plus
    """
    q_t_minus, q_t_plus = 0, 0
    m_players = arr_pl_M_T_K_vars_modif.shape[0]
    for num_pl_i in range(0, m_players):
        Pi, Ci, Si, Si_max = None, None, None, None
        Pi = arr_M_K_vars[num_pl_i, k, INDEX_ATTRS["Pi"]]
        Ci = arr_M_K_vars[num_pl_i, k, INDEX_ATTRS["Ci"]]
        Si = arr_M_K_vars[num_pl_i, k, INDEX_ATTRS["Si"]]
        Si_max = arr_M_K_vars[num_pl_i, k, INDEX_ATTRS["Si_max"]]
    
        diff_Ci_Pi = fct_aux.fct_positive(sum_list1=Ci, sum_list2=Pi)
        diff_Pi_Ci_Si_max = fct_aux.fct_positive(sum_list1=Pi, sum_list2=Ci+Si_max-Si)
        diff_Pi_Ci = fct_aux.fct_positive(sum_list1=Pi, sum_list2=Ci)
        diff_Ci_Pi_Si = fct_aux.fct_positive(sum_list1=Ci, sum_list2=Pi+Si)
        diff_Ci_Pi_Simax_Si = diff_Ci_Pi - diff_Pi_Ci_Si_max
        diff_Pi_Ci_Si = diff_Pi_Ci - diff_Ci_Pi_Si
        
        q_t_minus += diff_Ci_Pi_Simax_Si
        q_t_plus += diff_Pi_Ci_Si
        
        # print("Pi={}, Ci={}, Si_max={}, Si={}".format(Pi, Ci, Si_max, Si))
        # print("player {}: diff_Ci_Pi_Simax_Si={} -> q_t_minus={}, diff_Pi_Ci_Si={} -> q_t_plus={} ".format(
        #     num_pl_i, diff_Ci_Pi_Simax_Si, q_t_minus, diff_Pi_Ci_Si, q_t_plus))
        
    # print("q_t_minus={}, q_t_plus={}".format(q_t_minus, q_t_plus))
    q_t_minus = q_t_minus if q_t_minus >= 0 else 0
    q_t_plus = q_t_plus if q_t_plus >= 0 else 0
    return q_t_minus, q_t_plus
# _________         compute q_t_minus, q_t_plus:  fin          ________________


def compute_starting_prices(arr_M_K_vars, pi_hp_plus, pi_hp_minus, a, b,
                            pi_sg_plus_t, pi_sg_minus_t,
                            t, manual_debug):
    """
    compute initial prices ie prices used since algo runs at a specified period
    """
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = None
    pi_0_plus_t, pi_0_minus_t = None
    if manual_debug:
        pi_sg_plus_t_minus_1 = fct_aux.MANUEL_DBG_PI_SG_PLUS_T_K #8
        pi_sg_minus_t_minus_1 = fct_aux.MANUEL_DBG_PI_SG_MINUS_T_K #10
        pi_0_plus_t = fct_aux.MANUEL_DBG_PI_0_PLUS_T_K #4 
        pi_0_minus_t = fct_aux.MANUEL_DBG_PI_0_MINUS_T_K #3
    else:
        q_t_minus, q_t_plus = compute_upper_bound_quantity_energy(
                                arr_M_K_vars, k=0)
        phi_hp_minus_t = fct_aux.compute_cost_energy_bought_by_SG_2_HP(
                            pi_hp_minus=pi_hp_minus, 
                            quantity=q_t_minus,
                            b=b)
        phi_hp_plus_t = fct_aux.compute_benefit_energy_sold_by_SG_2_HP(
                            pi_hp_plus=pi_hp_plus, 
                            quantity=q_t_plus,
                            a=a)
        pi_hp_minus_t = round(phi_hp_minus_t/q_t_minus, fct_aux.N_DECIMALS) \
                        if q_t_minus != 0 \
                        else 0
        pi_hp_plus_t = round(phi_hp_plus_t/q_t_plus, fct_aux.N_DECIMALS) \
                        if q_t_plus != 0 \
                        else 0
        pi_sg_plus_t0_minus_1, pi_sg_minus_t0_minus_1 = None, None            
        if t == 0:
            pi_sg_plus_t0_minus_1 = pi_hp_plus_t - 1
            pi_sg_minus_t0_minus_1 = pi_hp_minus_t - 1
        pi_sg_plus_t_minus_1 = pi_sg_plus_t0_minus_1 if t == 0 \
                                                     else pi_sg_plus_t
        pi_sg_minus_t_minus_1 = pi_sg_minus_t0_minus_1 if t == 0 \
                                                        else pi_sg_minus_t
        
        print("q_t-={}, phi_hp-={}, pi_hp-={}, pi_sg-_t-1={}, ".format(q_t_minus, phi_hp_minus_t, pi_hp_minus_t, pi_sg_minus_t_minus_1))
        print("q_t+={}, phi_hp+={}, pi_hp+={}, pi_sg+_t-1={}".format(q_t_plus, phi_hp_plus_t, pi_hp_plus_t, pi_sg_plus_t_minus_1))
        
        pi_0_plus_t = round(pi_sg_minus_t_minus_1*pi_hp_plus_t/pi_hp_minus_t, 
                            fct_aux.N_DECIMALS) \
                        if t > 0 \
                        else fct_aux.PI_0_PLUS_INIT #4
                            
        pi_0_minus_t = pi_sg_minus_t_minus_1 \
                        if t > 0 \
                        else fct_aux.PI_0_MINUS_INIT #3
        print("t={}, pi_0_plus_t={}, pi_0_minus_t={}".format(t, pi_0_plus_t, pi_0_minus_t))
        
    return pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1, \
            pi_hp_plus_t, pi_hp_minus_t, \
            pi_0_plus_t, pi_0_minus_t
                
# _____________________________________________________________________________
#           compute starting prices at t: fin
# _____________________________________________________________________________

#_____________________________________________________________________________
#           compute state and gamma of players : debut
#_____________________________________________________________________________
# ______                update variables to arr_M_K_vars: debut            ____
def MIX_LRI_update_variables(arr_M_K_vars, variables,
                             num_pl_i, k, gamma_i,
                             pi_0_minus_t, pi_0_plus_t, 
                             pi_hp_minus_t, pi_hp_plus_t, dbg):
    """
    update variables to arr_M_K_vars
    """
    # ____              update cell arrays: debut               _______
    for (var, val) in variables:
            arr_M_K_vars[num_pl_i, :, INDEX_ATTRS[var]] = val
            
    # ____              update cell arrays: fin                 _______
    
    bool_gamma_i = (gamma_i >= min(pi_0_minus_t, pi_0_plus_t)-1) \
                    & (gamma_i <= max(pi_hp_minus_t, pi_hp_plus_t)+1)
    print("GAMMA :  player={}, val={}, bool_gamma_i={}"\
          .format(num_pl_i, gamma_i, bool_gamma_i)) if dbg else None
        
    return arr_M_K_vars
# ______                update variables to arr: fin               ____

# ______                compute state and gamma : debut               ____
def MIX_LRI_compute_gamma_state_4_period_t(arr_M_K_vars, 
                                           arr_M_t_plus_1_vars,
                                           arr_M_t_minus_1_vars,
                                           gamma_version, 
                                           pi_0_plus_t,
                                           pi_0_minus_t,
                                           pi_hp_plus_t,
                                           pi_hp_minus_t, 
                                           manual_debug=False, 
                                           dbg=False):
    """
    compute state and gamma of each player from algo and gamma_version
    
    arr_M_t_plus_1_vars: shape M_PLAYERS * len(INDEX_ATTRS)
    arr_M_t_minus_1_vars: shape M_PLAYERS * len(INDEX_ATTRS)
    arr_M_K_vars : shape M_PLAYERS * k_steps * len(INDEX_ATTRS)

    """
    k = 0
    m_players = arr_M_K_vars.shape[0]
    Cis_t_plus_1 = arr_M_t_plus_1_vars[:, k, INDEX_ATTRS["Ci"]] 
    Pis_t_plus_1 = arr_M_t_plus_1_vars[:, k, INDEX_ATTRS["Pi"]] 
    Cis_Pis_t_plus_1 = Cis_t_plus_1 - Pis_t_plus_1
    Cis_Pis_t_plus_1[Cis_Pis_t_plus_1 < 0] = 0
    GC_t = np.sum(Cis_Pis_t_plus_1)
    
    # initialisation of variables for gamma_version = 2
    state_is = np.empty(shape=(m_players,), dtype=object)
    Sis = np.zeros(shape=(m_players,))
    GSis_t_minus = np.zeros(shape=(m_players,)) 
    GSis_t_plus = np.zeros(shape=(m_players,))
    Xis = np.zeros(shape=(m_players,)) 
    Yis = np.zeros(shape=(m_players,))
    
    
    for num_pl_i in range(0, m_players):
        Pi = arr_M_K_vars[num_pl_i, k, INDEX_ATTRS["Pi"]]
        Ci = arr_M_K_vars[num_pl_i, k, INDEX_ATTRS["Ci"]]
        Si_max = arr_M_K_vars[num_pl_i, k, INDEX_ATTRS["Si_max"]]
        Pi_t_plus_1 = arr_M_t_plus_1_vars[num_pl_i, INDEX_ATTRS["Pi"]]
        Ci_t_plus_1 = arr_M_t_plus_1_vars[num_pl_i, INDEX_ATTRS["Ci"]]
        Si = arr_M_t_minus_1_vars[num_pl_i, INDEX_ATTRS["Si"]]
        
        prod_i, cons_i, r_i, gamma_i, state_i = 0, 0, 0, 0, ""
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        state_i = pl_i.find_out_state_i()
        state_is[num_pl_i] = state_i
        Si_t_plus_1, Si_t_minus, Si_t_plus = None, None, None
        Xi, Yi = None, None
        if state_i == fct_aux.STATES[0]:                                       # state1 or Deficit
            Si_t_minus = 0
            Si_t_plus = Si
            Xi = pi_0_minus_t
            Yi = pi_hp_minus_t
        elif state_i == fct_aux.STATES[1]:                                     # state2 or Self
            Si_t_minus = Si - (Ci - Pi)
            Si_t_plus = Si
            Xi = pi_0_minus_t
            Yi = pi_hp_minus_t
        elif state_i == fct_aux.STATES[2]:                                     # state3 or Surplus
            Si_t_minus = Si
            Si_t_plus = max(Si_max, Si+(Pi-Ci))
            Xi = pi_0_plus_t
            Yi = pi_hp_plus_t
        Sis[num_pl_i] = Si
        GSis_t_minus[num_pl_i] = Si_t_minus
        GSis_t_plus[num_pl_i] = Si_t_plus
        Xis[num_pl_i] = Xi; Yis[num_pl_i] = Yi
        
        gamma_i, gamma_i_min, gamma_i_max, gamma_i_mid  = None, None, None, None
        res_mid = None
        if manual_debug:
            gamma_i_min = fct_aux.MANUEL_DBG_GAMMA_I
            gamma_i_mid = fct_aux.MANUEL_DBG_GAMMA_I
            gamma_i_max = fct_aux.MANUEL_DBG_GAMMA_I
            gamma_i = fct_aux.MANUEL_DBG_GAMMA_I
        else:
            Si_t_plus_1 = fct_aux.fct_positive(Ci_t_plus_1, Pi_t_plus_1)
            gamma_i_min = Xi - 1
            gamma_i_max = Yi + 1
            # print("Pi={}, Ci={}, Si={}, Si_t_plus_1={}, Si_t_minus={}, Si_t_plus={}".format(Pi, 
            #         Ci, Si, Si_t_plus_1, Si_t_minus, Si_t_plus))
            if Si_t_plus_1 < Si_t_minus:
                # Xi - 1
                gamma_i = gamma_i_min
            elif Si_t_plus_1 >= Si_t_plus:
                # Yi + 1
                gamma_i = gamma_i_max
            elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                res_mid = ( Si_t_plus_1 - Si_t_minus) / \
                        (Si_t_plus - Si_t_minus)
                Z = Xi + (Yi-Xi)*res_mid
                gamma_i_mid = int(np.floor(Z))
                gamma_i = gamma_i_mid
                
        if gamma_version == 0:
            gamma_i = 0
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                         ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_M_K_vars = MIX_LRI_update_variables(
                            arr_M_K_vars=arr_M_K_vars, variables=variables,
                            num_pl_i=num_pl_i, k=k, gamma_i=gamma_i,
                            pi_0_minus_t=pi_0_minus_t, pi_0_plus_t=pi_0_plus_t, 
                            pi_hp_minus_t=pi_hp_minus_t, 
                            pi_hp_plus_t=pi_hp_plus_t, dbg=dbg)
            
        elif gamma_version == 1:
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                         ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_M_K_vars = MIX_LRI_update_variables(
                            arr_M_K_vars=arr_M_K_vars, variables=variables,
                            num_pl_i=num_pl_i, k=k, gamma_i=gamma_i,
                            pi_0_minus_t=pi_0_minus_t, pi_0_plus_t=pi_0_plus_t, 
                            pi_hp_minus_t=pi_hp_minus_t, 
                            pi_hp_plus_t=pi_hp_plus_t, dbg=dbg)
            
        elif gamma_version == 3:
            gamma_i = None
            if manual_debug:
                gamma_i = fct_aux.MANUEL_DBG_GAMMA_I
            elif Si_t_plus_1 < Si_t_minus:
                gamma_i = gamma_i_min
            else :
                gamma_i = gamma_i_max
                
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_M_K_vars = MIX_LRI_update_variables(
                            arr_M_K_vars=arr_M_K_vars, variables=variables,
                            num_pl_i=num_pl_i, k=k, gamma_i=gamma_i,
                            pi_0_minus_t=pi_0_minus_t, pi_0_plus_t=pi_0_plus_t, 
                            pi_hp_minus_t=pi_hp_minus_t, 
                            pi_hp_plus_t=pi_hp_plus_t, dbg=dbg)
            
        elif gamma_version == 4:
            gamma_i = None
            if manual_debug:
                gamma_i = fct_aux.MANUEL_DBG_GAMMA_I
            else:
                if Si_t_plus_1 < Si_t_minus:
                    # Xi - 1
                    gamma_i = gamma_i_min
                elif Si_t_plus_1 >= Si_t_plus:
                    # Yi + 1
                    gamma_i = gamma_i_max
                elif Si_t_plus_1 >= Si_t_minus and Si_t_plus_1 < Si_t_plus:
                    res_mid = ( Si_t_plus_1 - Si_t_minus) / \
                            (Si_t_plus - Si_t_minus)
                    Z = Xi + (Yi-Xi) * np.sqrt(res_mid)
                    gamma_i_mid = int(np.floor(Z))
                    gamma_i = gamma_i_mid
                
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                     ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_M_K_vars = MIX_LRI_update_variables(
                            arr_M_K_vars=arr_M_K_vars, variables=variables,
                            num_pl_i=num_pl_i, k=k, gamma_i=gamma_i,
                            pi_0_minus_t=pi_0_minus_t, pi_0_plus_t=pi_0_plus_t, 
                            pi_hp_minus_t=pi_hp_minus_t, 
                            pi_hp_plus_t=pi_hp_plus_t, dbg=dbg)
            
        elif gamma_version == -1:
            gamma_i = np.random.randint(low=2, high=21, size=1)[0]
            variables = [("Si", Si), ("state_i", state_i), ("gamma_i", gamma_i), 
                         ("Si_minus", Si_t_minus), ("Si_plus", Si_t_plus)]
            arr_M_K_vars = MIX_LRI_update_variables(
                            arr_M_K_vars=arr_M_K_vars, variables=variables,
                            num_pl_i=num_pl_i, k=k, gamma_i=gamma_i,
                            pi_0_minus_t=pi_0_minus_t, pi_0_plus_t=pi_0_plus_t, 
                            pi_hp_minus_t=pi_hp_minus_t, 
                            pi_hp_plus_t=pi_hp_plus_t, dbg=dbg)
        
        Si_t_minus_1 = arr_M_t_minus_1_vars[num_pl_i, INDEX_ATTRS["Si"]]
        print("Si_t_minus_1={}, Si={}".format(Si_t_minus_1, Si)) \
        if dbg else None
        
    if gamma_version == 2:
        GS_t_minus = np.sum(GSis_t_minus)
        GS_t_plus = np.sum(GSis_t_plus)
        gamma_is = None
        if GC_t <= GS_t_minus:
            gamma_is = Xis - 1
        elif GC_t > GS_t_plus:
            gamma_is = Yis + 1
        else:
            frac = (GC_t - GS_t_minus) / (GS_t_plus - GS_t_minus)
            res_is = Xis + (Yis-Xis)*frac
            gamma_is = np.floor(res_is)
            
        # ____              update cell arrays: debut               _______
        variables = [("Si", Sis), ("state_i", state_is), 
                     ("Si_minus", GSis_t_minus), ("Si_plus", GSis_t_plus)]
        for (var,vals) in variables:
            arr_M_K_vars[:, :, INDEX_ATTRS[var]] = vals
            if manual_debug:
                arr_M_K_vars[:, :, INDEX_ATTRS["gamma_i"]] \
                    = fct_aux.MANUEL_DBG_GAMMA_I
            else:
                arr_M_K_vars[:, :, INDEX_ATTRS["gamma_i"]] = gamma_is
                
    return arr_M_K_vars
# ______                compute state and gamma : fin              ____
#_____________________________________________________________________________
#           compute state and gamma of players : fin
#_____________________________________________________________________________

#_____________________________________________________________________________
#                   balance players of game : debut
#_____________________________________________________________________________
def MIX_LRI_compute_prod_cons_SG(arr_M_k_vars):
    """
    compute the production In_sg and the consumption Out_sg in the SG.
    arr_M_k_vars: shape M_PLAYERS * len(INDEX_ATTRS)
    """
    In_sg = sum( arr_M_k_vars[:, INDEX_ATTRS["prod_i"]].astype(np.float64) )
    Out_sg = sum( arr_M_k_vars[:, INDEX_ATTRS["cons_i"]].astype(np.float64) )
    return In_sg, Out_sg
    
def MIX_LRI_compute_utility_players(arr_M_k_vars, gamma_is, k, b0_k, c0_k):
    """
    compute the benefit and the cost of each player at time t
    """
    bens_k = b0_k * arr_M_k_vars[:, INDEX_ATTRS["prod_i"]] \
            + gamma_is * arr_M_k_vars[:, INDEX_ATTRS["r_i"]]
    csts_k = c0_k * arr_M_k_vars[:, INDEX_ATTRS["cons_i"]]
    bens_k = np.around(np.array(bens_k, dtype=float), fct_aux.N_DECIMALS)
    csts_k = np.around(np.array(csts_k, dtype=float), fct_aux.N_DECIMALS)
    return bens_k, csts_k

def balanced_player_game_t(arr_M_k_vars, arr_M_k_minus_1_vars, k, 
                           pi_hp_plus, pi_hp_minus, 
                           a, b,
                           pi_0_plus_t, pi_0_minus_t,
                           random_mode,
                           dbg=False):
    """
    balanced each player of game and compute b0, c0, ben_i, cst_i 
    arr_M_k_vars: shape M_PLAYERS * len(INDEX_ATTRS)
    """
    
    m_players = arr_M_k_vars.shape[0]
    for num_pl_i in range(0, m_players):
        Pi = arr_M_k_vars[num_pl_i, INDEX_ATTRS['Pi']]
        Ci = arr_M_k_vars[num_pl_i, INDEX_ATTRS['Ci']]
        Si = arr_M_k_vars[num_pl_i, INDEX_ATTRS['Si']]
        Si_max = arr_M_k_vars[num_pl_i, INDEX_ATTRS['Si_max']]
        gamma_i = arr_M_k_vars[num_pl_i, INDEX_ATTRS['gamma_i']]
        prod_i, cons_i, r_i = 0, 0, 0
        state_i = arr_M_k_vars[num_pl_i, INDEX_ATTRS['state_i']]
        
        pl_i = None
        pl_i = players.Player(Pi, Ci, Si, Si_max, gamma_i, 
                              prod_i, cons_i, r_i, state_i)
        pl_i.set_R_i_old(Si_max-Si)                                            # update R_i_old
        
        # select mode for player num_pl_i
        mode_i = None
        if random_mode:
            S1_p_i_t_k = arr_M_k_vars[num_pl_i, INDEX_ATTRS["S1_p_i_j_k"]] \
                if k == 0 \
                else arr_M_k_minus_1_vars[num_pl_i, INDEX_ATTRS["S1_p_i_j_k"]]
            pl_i.select_mode_i(p_i=S1_p_i_t_k)
            mode_i = pl_i.get_mode_i()
        else:
            mode_i = arr_M_k_vars[num_pl_i, INDEX_ATTRS['mode_i']]
            pl_i.set_mode_i(mode_i)
        
        # compute cons, prod, r_i
        pl_i.update_prod_cons_r_i()

        # is pl_i balanced?
        boolean, formule = fct_aux.balanced_player(pl_i, thres=0.1)
        
        # update variables in arr_pl_M_T_k
        tup_cols_values = [("prod_i", pl_i.get_prod_i()), 
                ("cons_i", pl_i.get_cons_i()), ("r_i", pl_i.get_r_i()),
                ("R_i_old", pl_i.get_R_i_old()), ("Si", pl_i.get_Si()),
                ("Si_old", pl_i.get_Si_old()), ("mode_i", pl_i.get_mode_i()), 
                ("balanced_pl_i", boolean), ("formule", formule)]
        for col, val in tup_cols_values:
            arr_M_k_vars[num_pl_i, INDEX_ATTRS[col]] = val
            
    ## compute prices inside smart grids
    # compute In_sg, Out_sg
    In_sg, Out_sg = MIX_LRI_compute_prod_cons_SG(
                        arr_M_k_vars=arr_M_k_vars)
    # compute prices of an energy unit price for cost and benefit players
    b0_k, c0_k = fct_aux.compute_energy_unit_price(
                        pi_0_plus=pi_0_plus_t, pi_0_minus=pi_0_minus_t, 
                        pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus,
                        a=a, b=b,
                        In_sg=In_sg, Out_sg=Out_sg)
    # compute ben, cst of shapes (M_PLAYERS,) 
    # compute cost (csts) and benefit (bens) players by energy exchanged.
    gamma_is = arr_M_k_vars[:, INDEX_ATTRS["gamma_i"]]
    bens_k, csts_k = MIX_LRI_compute_utility_players(
                            arr_M_k_vars= arr_M_k_vars, 
                            gamma_is=gamma_is, 
                            k=k, 
                            b0_k=b0_k, 
                            c0_k=c0_k)
    print('#### bens_t_k={}, csts_t_k={}'.format(
            bens_k.shape, csts_k.shape)) \
        if dbg else None

    ## update variables of arr_M_K 
    cols_vals = [("ben_i", bens_k) , ("cst_i", csts_k), ("b0_t",b0_k), ("c0_t",c0_k)]
    for col, vals in cols_vals:
        arr_M_k_vars[:, INDEX_ATTRS[col]] = vals
    
    return arr_M_k_vars
#_____________________________________________________________________________
#                   balance players of game : fin
#_____________________________________________________________________________

#_____________________________________________________________________________
#             update p_i_j_k players at t and k --> debut       
#_____________________________________________________________________________
def algo_utility_version1(arr_M_k_vars, arr_bg_i_nb_repeat_k,
                          learning_rate, 
                          bg_min_M_0_k_minus_2, bg_max_M_0_k_minus_2, 
                          nb_repeat_k, dbg):
    """
    compute the utility of players following the version 1 in the document
    
    arr_M_k_vars: shape M_PLAYERS * len(INDEX_ATTRS)
    arr_bg_i_nb_repeat_k: shape M_PLAYERS * fct_aux.NB_REPEAT_K_MAX
    
    """
    
    # compute stock maximal
    stock_max = np.max(arr_M_k_vars[:,INDEX_ATTRS["Si_plus"]] 
                       * arr_M_k_vars[:,INDEX_ATTRS["gamma_i"]],
                       axis=0)
    
    m_players = arr_M_k_vars.shape[0]
    # compute bg_i
    for num_pl_i in range(0, m_players):
        state_i = arr_M_k_vars[num_pl_i, INDEX_ATTRS["state_i"]]
        if state_i == fct_aux.STATES[2]:
            arr_M_k_vars[num_pl_i, INDEX_ATTRS["bg_i"]] \
                = arr_M_k_vars[num_pl_i, INDEX_ATTRS["ben_i"]]
        else:
            arr_M_k_vars[num_pl_i, INDEX_ATTRS["bg_i"]] \
                = arr_M_k_vars[num_pl_i, INDEX_ATTRS["cst_i"]] \
                    - arr_M_k_vars[num_pl_i, INDEX_ATTRS["ben_i"]] + stock_max
    
    # checkout players' bg_min, bg_max from k=0 to k
    bg_k = arr_M_k_vars[num_pl_i, INDEX_ATTRS["bg_i"]] 
    bg_max_M_0_k_minus_1 = np.maximum(bg_max_M_0_k_minus_2, bg_k)
    bg_min_M_0_k_minus_1 = np.minimum(bg_min_M_0_k_minus_2, bg_k)
    
    # True in comp_min_max_bg MEANS bg_min=bg_max and num_pli doesn't play
    comp_min_max_bg = np.isclose(bg_min_M_0_k_minus_1,
                                 bg_max_M_0_k_minus_1, 
                                 equal_nan=False,
                                 atol=pow(10,-fct_aux.N_DECIMALS))
    
    arr_M_k_vars[:, INDEX_ATTRS["non_playing_players"]] \
        = np.invert(comp_min_max_bg).astype(int)
        
    if comp_min_max_bg.any() == True \
        and nb_repeat_k != fct_aux.NB_REPEAT_K_MAX:
        bool_bg_i_min_eq_max = True
        
        return arr_M_k_vars, bool_bg_i_min_eq_max
    
    # TODO STOP HERE FOR IDENTIFY THE ROLE OF arr_bg_i_nb_repeat_k
                              
    
    
    
    

def update_p_i_j_k_by_defined_utility_funtion(arr_M_k_vars, 
                                              arr_bg_i_nb_repeat_k,
                                              nb_repeat_k,
                                              learning_rate, 
                                              algo_utility, 
                                              bg_min_M_0_k_minus_2, 
                                              bg_max_M_0_k_minus_2, dbg=False):
    if algo_utility == 1:
        # version 1 of utility fonction
        algo_utility_version1(arr_M_k_vars=arr_M_k_vars, 
                              arr_bg_i_nb_repeat_k=arr_bg_i_nb_repeat_k,
                              learning_rate=learning_rate, 
                              bg_min_M_0_k_minus_2=bg_min_M_0_k_minus_2, 
                              bg_max_M_0_k_minus_2=bg_max_M_0_k_minus_2,
                              nb_repeat_k=nb_repeat_k,
                              dbg=dbg)
    elif algo_utility == 2:
        # version 1 of utility fonction
        #algo_utility_version2()
        pass
    
#_____________________________________________________________________________
#             update p_i_j_k players at t and k --> fin       
#_____________________________________________________________________________

###############################################################################
#                   definition  de l algo LRI
#
###############################################################################

# _________        learning steps of LRI algo : debut ________________________
def lri_learning_steps(arr_M_K_vars, arr_M_t_plus_1_vars, arr_M_t_minus_1_vars,
                       algo_utility, gamma_version, k_steps,
                       pi_0_plus_t, pi_0_minus_t,
                       pi_hp_plus_t, pi_hp_minus_t, 
                       pi_hp_plus, pi_hp_minus, 
                       a, b,
                       learning_rate,
                       manual_debug, dbg):
    """
    learning steps of LRI algo 
    """
    
    # __ compute state and gamma of players : debut __
    arr_M_K_vars = MIX_LRI_compute_gamma_state_4_period_t(
                        arr_M_K_vars=arr_M_K_vars, 
                        arr_M_t_plus_1_vars=arr_M_t_plus_1_vars,
                        arr_M_t_minus_1_vars=arr_M_t_minus_1_vars,
                        gamma_version=gamma_version, 
                        pi_0_plus_t=pi_0_plus_t,
                        pi_0_minus_t=pi_0_minus_t,
                        pi_hp_plus_t=pi_hp_plus_t,
                        pi_hp_minus_t=pi_hp_minus_t, 
                        manual_debug=manual_debug, 
                        dbg=dbg)
    # __ compute state and gamma of players : fin __
    
    m_players = arr_M_K_vars.shape[0]
    arr_bg_i_nb_repeat_k = np.empty(shape=(m_players, fct_aux.NB_REPEAT_K_MAX)) # enable to get bg_min and bg_max
    arr_bg_i_nb_repeat_k.fill(np.nan)
    
    bg_min_M_0_k_minus_1 = np.empty(shape=(m_players,))                         # bg min for each players from 0 to k-1
    bg_max_M_0_k_minus_1 = np.empty(shape=(m_players,))                         # bg max for each players from 0 to k-1
    
    # ____   run balanced sg for one period and all k_steps : debut   _____
    dico_gamma_players_t = dict()
    bool_stop_learning = False
    k_stop_learning = 0
    nb_repeat_k = 0
    k = 0
    while k<k_steps and not bool_stop_learning:
        print(" -------  k = {}, nb_repeat_k = {}  ------- ".format(k, 
                nb_repeat_k)) if k%50 == 0 else None
        ### balanced_player_game_t
        arr_M_k_minus_1_vars = arr_M_K_vars[:,k,:] \
                                if k == 0 \
                                else arr_M_K_vars[:,k-1,:]
        arr_M_k_vars \
            = balanced_player_game_t(
                arr_M_k_vars=arr_M_K_vars[:,k,:].copy(), 
                arr_M_k_minus_1_vars=arr_M_k_minus_1_vars, 
                k=k, 
                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, 
                a=a, b=b,
                pi_0_plus_t=pi_0_plus_t, pi_0_minus_t=pi_0_minus_t,
                random_mode=True,
                dbg=dbg)

        ## looking for the max and min bg for each player
        bg_min_M_0_k_minus_2, bg_max_M_0_k_minus_2 \
            = looking_bg_max_min(arr_M_K_vars=arr_M_K_vars, k=k-1)              # 0 to k-2= k-1 items
        ### compute p_i_j_k of players and compute players' utility
        update_p_i_j_k_by_defined_utility_funtion(
                arr_M_k_vars=arr_M_k_vars, 
                arr_bg_i_nb_repeat_k=arr_bg_i_nb_repeat_k.copy(),
                nb_repeat_k=nb_repeat_k,
                learning_rate=learning_rate, 
                algo_utility=algo_utility, 
                bg_min_M_0_k_minus_2=bg_min_M_0_k_minus_2, 
                bg_max_M_0_k_minus_2=bg_max_M_0_k_minus_2, 
                dbg=dbg)
    
    # ____   run balanced sg for one period and all k_steps : fin     _____
        
    pass
# _________        learning steps of LRI algo : fin ________________________

# ______________       main function of LRI   ---> debut      _________________
def lri_balanced_player_game(arr_pl_M_T_vars_init,
                             pi_hp_plus, pi_hp_minus,
                             a, b,
                             gamma_versions, utility_function_versions,
                             k_steps, learning_rate, p_i_j_ks,
                             path_to_save, manual_debug, dbg=False):
    """
    algorithm LRI with 
    * stopping learning when all players p_i_j_ks are higher 
        than STOP_LEARNING_PROBA = 0.8
    * choosing utility function version having the best Perf or gain Ru at each time period
        Perf = sum_{0<=i<N} ben_i - cst_i
        
    """
    # algo_gamma=(utility_function, gamma_version)
    algos_gamma = [(tu[0], tu[1]) 
                   for tu in zip(utility_function_versions, gamma_versions)]
    
    m_players = arr_pl_M_T_vars_init.shape[0]
    t_periods = arr_pl_M_T_vars_init.shape[1]
    arr_pl_M_T_vars_res = arr_pl_M_T_vars_init.copy()
    
    pi_sg_plus_t0_minus_1 = None
    pi_sg_minus_t0_minus_1 = None
    pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1 = None, None
    pi_sg_plus_t, pi_sg_minus_t = None, None
    
    for t in range(0, t_periods):
        
        # __ create a arr of shape m_players, k_steps, len(INDEX_ATTRS): debut ___
        arr_M_K_vars = np.zeros((m_players, k_steps, len(INDEX_ATTRS)),
                                 dtype=object)
        s = set(fct_aux.AUTOMATE_INDEX_ATTRS.keys()).intersection(INDEX_ATTRS)
        inter_keys_arr = sorted(list(map(lambda x: fct_aux.AUTOMATE_INDEX_ATTRS[x],
                                         s)))
        arr_M_K_vars[:,:,inter_keys_arr] = arr_pl_M_T_vars_init[:,t,inter_keys_arr]
        
        arr_M_K_vars[:,:,INDEX_ATTRS["S1_p_i_j_k"]] = 0.5
        arr_M_K_vars[:,:,INDEX_ATTRS["S2_p_i_j_k"]] = 0.5
        arr_M_K_vars[:,:,INDEX_ATTRS["non_playing_players"]] \
            = fct_aux.NON_PLAYING_PLAYERS["PLAY"]
        
        ## check Pi of arr_M_K_vars and arr_pl_M_T_vars_init are the same
        name_vars = ["Ci", "Pi", "Si", "Si_max"]
        boolx = list()
        for i,name_var in enumerate(name_vars):
            k = np.random.randint(low=0, high=k_steps)
            boolx[i] \
                = all(arr_M_K_vars[:,k, INDEX_ATTRS[name_var]] \
                      == arr_pl_M_T_vars_init[
                          :, t, fct_aux.AUTOMATE_INDEX_ATTRS[name_var]])
        res = "OK" if all(boolx) else "NOK"
        print("random check Pi,Ci,Si on arr_M_K_vars = {}".format(res))
        # __ create a arr of shape m_players, k_steps, len(INDEX_ATTRS): fin ___
        
        # __ compute starting prices : debut ____
        pi_sg_plus_t_minus_1, pi_sg_minus_t_minus_1, \
        pi_hp_plus_t, pi_hp_minus_t, \
        pi_0_plus_t, pi_0_minus_t \
            = compute_starting_prices(arr_M_K_vars=arr_M_K_vars, 
                                      pi_hp_plus=pi_hp_plus, 
                                      pi_hp_minus=pi_hp_minus, 
                                      a=a, b=b,
                                      pi_sg_plus_t=pi_sg_plus_t, 
                                      pi_sg_minus_t=pi_sg_minus_t,
                                      t=t, manual_debug=manual_debug)
        # __ compute starting prices : fin ____
        
        # __ update some variables of arr_M_K_vars: debut __
        arr_M_K_vars[:, :, INDEX_ATTRS["pi_0_plus_t"]] = pi_0_plus_t
        arr_M_K_vars[:, :, INDEX_ATTRS["pi_0_minus_t"]] = pi_0_minus_t
        arr_M_K_vars[:, :, INDEX_ATTRS["pi_hp_plus_t"]] = pi_hp_plus_t
        arr_M_K_vars[:, :, INDEX_ATTRS["pi_hp_minus_t"]] = pi_hp_minus_t
        arr_M_K_vars[:, :, INDEX_ATTRS["pi_sg_plus_t"]] = pi_sg_plus_t_minus_1
        arr_M_K_vars[:, :, INDEX_ATTRS["pi_sg_minus_t"]] = pi_sg_minus_t_minus_1
        # __ update some variables of arr_M_K_vars: fin __
        
        
        # __ learning steps from algo utility and gamma version : debut __
        arr_M_t_plus_1_vars = arr_pl_M_T_vars_init[:, t+1, :] \
                                if t+1 > t_periods \
                                else arr_pl_M_T_vars_init[:, t, :]
        arr_M_t_minus_1_vars = arr_pl_M_T_vars_init[:, t-1, :] \
                                if t-1 >= 0 \
                                else arr_pl_M_T_vars_init[:, t, :]
        # lri1V1
        arr_M_K_vars_lri1V1 \
            = lri_learning_steps(
                arr_M_K_vars=arr_M_K_vars.copy(), 
                arr_M_t_plus_1_vars=arr_M_t_plus_1_vars,
                arr_M_t_minus_1_vars=arr_M_t_minus_1_vars,
                algo_utility=algos_gamma[0][0], 
                gamma_version=algos_gamma[0][1],
                k_steps=k_steps, 
                pi_0_plus_t=pi_0_plus_t,
                pi_0_minus_t=pi_0_minus_t,
                pi_hp_plus_t=pi_hp_plus_t,
                pi_hp_minus_t=pi_hp_minus_t, 
                pi_hp_plus=pi_hp_plus, pi_hp_minus=pi_hp_minus, 
                a=a, b=b,
                learning_rate=learning_rate,
                manual_debug=manual_debug, 
                dbg=dbg)
            
        # lri2V0
        arr_M_K_vars_lri1V1 \
            = lri_learning_steps(
                arr_M_K_vars=arr_M_K_vars.copy(), 
                algo_utility=algos_gamma[1][0], 
                gamma_version=algos_gamma[1][1])
            
        # __ learning steps from algo utility and gamma version : fin __
        
        
        pass
    
    
# ______________       main function of LRI   ---> fin        _________________

###############################################################################
#                   definition  des unittests
#
###############################################################################

def test_lri_balanced_player_game_all_pijk_upper_08_Pi_Ci_NEW_AUTOMATE():
    # steps of learning
    k_steps = 250 # 5,250
    p_i_j_ks = [0.5, 0.5, 0.5]
    
    a = 1; b = 1; #a = 3; b = 5
    pi_hp_plus = 10 #0.2*pow(10,-3)
    pi_hp_minus = 20 # 0.33
    learning_rate = 0.1
    utility_function_versions = [1,2] #2 #1,2
    
    manual_debug = False #True #False #True
    gamma_versions = [1,0] #2 #1 #3: gamma_i_min #4: square_root
    fct_aux.N_DECIMALS = 2
    
    prob_A_A = 0.4; prob_A_B1 = 0.6; prob_A_B2 = 0.0; prob_A_C = 0.0;
    prob_B1_A = 0.4; prob_B1_B1 = 0.6; prob_B1_B2 = 0.0; prob_B1_C = 0.0;
    prob_B2_A = 0.0; prob_B2_B1 = 0.0; prob_B2_B2 = 0.5; prob_B2_C = 0.5;
    prob_C_A = 0.0; prob_C_B1 = 0.0; prob_C_B2 = 0.5; prob_C_C = 0.5 
    scenario1 = [(prob_A_A, prob_A_B1, prob_A_B2, prob_A_C), 
                     (prob_B1_A, prob_B1_B1, prob_B1_B2, prob_B1_C),
                     (prob_B2_A, prob_B2_B1, prob_B2_B2, prob_B2_C),
                     (prob_C_A, prob_C_B1, prob_C_B2, prob_C_C)]
    scenario_name = "scenario1"
    
    t_periods = 3#4
    setA_m_players = 4; setB1_m_players = 2; 
    setB2_m_players = 2; setC_m_players = 4;                                   # 12 players
    path_to_arr_pl_M_T = os.path.join(*["tests", "AUTOMATE_INSTANCES_GAMES"])
    used_instances = False #False #True
    
    arr_pl_M_T_vars_init \
        = fct_aux.get_or_create_instance_Pi_Ci_etat_AUTOMATE_SETAB1B2C_doc17(
            setA_m_players, setB1_m_players, 
            setB2_m_players, setC_m_players, 
            t_periods, 
            scenario=scenario1, 
            scenario_name=scenario_name,
            path_to_arr_pl_M_T=path_to_arr_pl_M_T, 
            used_instances=used_instances)
    
    fct_aux.checkout_values_Pi_Ci_arr_pl(arr_pl_M_T_vars_init)
    # return arr_pl_M_T_vars_init
    name_simu = "LRIx_simu_"+datetime.now().strftime("%d%m_%H%M")
    path_to_save = os.path.join("tests", name_simu)
    
    arr_pl_M_T_K_vars_modif = lri_balanced_player_game(
                                arr_pl_M_T_vars_init,
                                pi_hp_plus=pi_hp_plus, 
                                pi_hp_minus=pi_hp_minus,
                                a=a, b=b,
                                gamma_versions=gamma_versions,
                                utility_function_versions=utility_function_versions,
                                k_steps=k_steps, 
                                learning_rate=learning_rate,
                                p_i_j_ks=p_i_j_ks,
                                path_to_save=path_to_save, 
                                manual_debug=manual_debug, 
                                dbg=False)
    return arr_pl_M_T_K_vars_modif  

###############################################################################
#                   Execution
#
###############################################################################
if __name__ == "__main__":
    ti = time.time()
    
    arr_pl_M_T_K_vars_modif \
        = test_lri_balanced_player_game_all_pijk_upper_08_Pi_Ci_NEW_AUTOMATE()
    
    print("runtime = {}".format(time.time() - ti))
