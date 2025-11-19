#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 09:54:03 2022

@author: ying
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import random
import sys
from ebal_new import ebal
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.svm import SVR
# import sklearn
# import utils
# from skgarden import RandomForestQuantileRegressor


def recursive_EB(inputs, if_test, X_soft, X_force, X_soft_backup, X_force_backup, verbose=-1):
    
    success = False
    for a, b in inputs:  
        try:
            print("-===-")
            print(f"Running ({a}, {b})") 
            ebal_obj = ebal(PCA_ratio_bound = b, 
                            PCA_var_bound = a,
                            print_level = verbose)
            ebal_out = ebal_obj.ebalance(if_test = if_test, 
                                         X_soft = X_soft,  
                                         X_force = X_force)  
            # x_soft = X_soft 
            # x_force = X_force
            success = ebal_out['converged']
            if success: 
                weights = ebal_out['w']
                print("-===-")
                print(f"Attempt with ({a}, {b}) succeeded!")
                print("-===-")
                break
            else:
                print("-===-")
                print(f"Attempt with ({a}, {b}) didn't converge")
                print("-===-") 
        except Exception as e:
                # If an error occurs, print out message
                print("-===-")
                print(f"Attempt with ({a}, {b}) failed: {e}")
                print("-===-")
            
     
        
    if not success:
        print("None-successful EB with hidden features, use default")
        ebal_obj = ebal(PCA_ratio_bound = 1, 
                        PCA_var_bound = 0.0001)
        ebal_out = ebal_obj.ebalance(if_test = if_test, 
                                     X_soft =  X_soft_backup, 
                                     X_force = X_force_backup)    
        weights = ebal_out['w']
        # x_soft = X_soft_backup 
        # x_force = X_force_backup

    return ebal_out, weights 

def create_qt(data, name):
    data['qt1'+name] = 1 * (data[name] <= np.quantile(data[name], 0.1))
    data['qt2'+name] = 1 * (data[name] > np.quantile(data[name], 0.1)) * (data[name] <= np.quantile(data[name], 0.2))
    data['qt3'+name] = 1 * (data[name] > np.quantile(data[name], 0.2)) * (data[name] <= np.quantile(data[name], 0.3))
    data['qt4'+name] = 1 * (data[name] > np.quantile(data[name], 0.3)) * (data[name] <= np.quantile(data[name], 0.4))
    data['qt5'+name] = 1 * (data[name] > np.quantile(data[name], 0.4)) * (data[name] <= np.quantile(data[name], 0.5))
    data['qt6'+name] = 1 * (data[name] > np.quantile(data[name], 0.5)) * (data[name] <= np.quantile(data[name], 0.6))
    data['qt7'+name] = 1 * (data[name] > np.quantile(data[name], 0.6)) * (data[name] <= np.quantile(data[name], 0.7)) 
    data['qt8'+name] = 1 * (data[name] > np.quantile(data[name], 0.7)) * (data[name] <= np.quantile(data[name], 0.8)) 
    data['qt9'+name] = 1 * (data[name] > np.quantile(data[name], 0.8)) * (data[name] <= np.quantile(data[name], 0.9)) 
    return data

def eval_sel(sel_idx, ys, cs):
    if len(sel_idx) == 0:
        fdp = 0
        power = 0
    else:
        fdp = np.sum(ys[sel_idx] <= cs[sel_idx]) / len(sel_idx)
        power = np.sum(ys[sel_idx] > cs[sel_idx]) / sum(ys > cs) 
    return fdp, power


def BH(pvals, q=0.1):
    mh = len(pvals)
    df_all = pd.DataFrame({"id": range(mh), "pval": pvals})
    df_sorted = df_all.sort_values(by = "pval")
    df_sorted['threshold'] = (1+np.arange(mh)) * q / mh
    id_in = [j for j in range(mh) if df_sorted.iloc[j,1] <= df_sorted.iloc[j,2]]
    if len(id_in)==0:
        return(np.array([]))
    else:
        return(np.array(df_sorted.index[range(np.max(id_in)+1)]))
    

def weighted_BH(calib_scores, calib_weights, test_scores, test_weights, q = 0.1):
    pvals = np.zeros(len(test_scores))
    df_all = pd.concat((pd.DataFrame({"score": calib_scores, "weight": calib_weights, "cal": 1}), pd.DataFrame({"score": test_scores, "weight": test_weights, "cal": 0})))
    df_sorted = df_all.sort_values(by = 'score')
    all_sorted = np.array(df_sorted)
    sum_calib_weight = np.sum(calib_weights)
    
    p_vals = []
    for j in range(all_sorted.shape[0]):
        if all_sorted[j,2] == 0:
            p_vals.append( (np.sum(all_sorted[range(j),1] * all_sorted[range(j),2]) + all_sorted[j,1] * np.random.uniform(size=1)[0]) / (sum_calib_weight + all_sorted[j,1]) )
        else:
            p_vals.append(-1)
            
    df_sorted['pvals'] = p_vals
        
    df_test = df_sorted[df_sorted['cal'] == 0]
    df_test_sorted = df_test.sort_values(by='pvals')
    
    # BH(q)
    ntest = len(test_scores)
    df_test_sorted['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test_sorted.iloc[j,3] <= df_test_sorted.iloc[j,4]]
    
    if len(idx_smaller) == 0:
        return(np.array([]))
    else:
        idx_sel = np.array(df_test_sorted.index[range(np.max(idx_smaller)+1)])
        return(idx_sel)
        

def weighted_CC(calib_scores, calib_weights, test_scores, test_weights, q = 0.1):
    sum_calib_weight = np.sum(calib_weights)
    
    ntest = len(test_scores)
    # sel_0 = np.zeros((ntest, ntest)) # row j indicates hat{R}_j
    Rj_sizes = np.zeros(ntest)
    w_pvals = np.zeros(ntest)
    xis = np.random.uniform(size=ntest)
    
    for j in range(ntest):
        w_pvals[j] = ( np.sum(calib_weights[calib_scores < test_scores[j]]) + (np.sum(calib_weights[calib_scores == test_scores[j]]) + test_weights[j])  * np.random.uniform(size=1)[0] ) / (sum_calib_weight + test_weights[j])
        
    # bh_thds = q * np.linspace(1, ntest, num=ntest) / ntest 
    # bh_hat = np.array([np.sum(w_pvals <= ts) for ts in bh_thds]) * q / ntest
    
    for j in range(ntest): 
        # if np.floor(j/500) == j/500:
            # print(str(j)+"complete")
        if w_pvals[j] <= q:
            # compute all other pvals 
            pval_j = np.zeros(ntest)
            for k in range(ntest):
                if k != j:
                    pval_j[k] =  np.sum(calib_weights[calib_scores < test_scores[k]]) + test_weights[k] * (test_scores[j] < test_scores[k]) 
            pval_j = pval_j / (sum_calib_weight + test_weights[j])
             
            # run BH
            df_j = pd.DataFrame({"id": range(ntest), "pval": pval_j}).sort_values(by = 'pval')
            df_j['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
            idx_small_j = [s for s in range(ntest) if df_j.iloc[s, 1] <= df_j.iloc[s, 2]]
            Rj = np.array(df_j['id'])[range(np.max(idx_small_j)+1)] 
            Rj_sizes[j] = len(Rj)
        
    Cj = q * Rj_sizes / ntest
    
    df_all = pd.DataFrame({"id": range(ntest), "pval": w_pvals, "c": Cj, 
                           "hete_Rj": Rj_sizes * xis, 
                           "homo_Rj": Rj_sizes * np.random.uniform(size=1)[0], 
                           "Rj": Rj_sizes})
    
    pj_sel0 = w_pvals[w_pvals <= Cj]
    
    
    if len(pj_sel0) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]) 
    else:
        # heterogeneous pruning
        df_sel0 = df_all[df_all['pval'] <= df_all['c']].sort_values(by='hete_Rj')
        df_sel0['threshold'] = np.linspace(1, df_sel0.shape[0], num = df_sel0.shape[0])
        smaller = [j for j in range(df_sel0.shape[0]) if df_sel0.iloc[j,3] <= df_sel0.iloc[j,6]]
        if len(smaller) == 0:
            idx_sel_hete = np.array([])
        else:
            idx_sel_hete = np.array(df_sel0['id'])[range(np.max(smaller)+1)]
        
        # homogeneous pruning
        df_sel0 = df_all[df_all['pval'] <= df_all['c']].sort_values(by='homo_Rj')
        df_sel0['threshold'] = np.linspace(1, df_sel0.shape[0], num = df_sel0.shape[0])
        smaller = [j for j in range(df_sel0.shape[0]) if df_sel0.iloc[j,4] <= df_sel0.iloc[j,6]]
        if len(smaller) == 0:
            idx_sel_homo = np.array([])
        else:
            idx_sel_homo = np.array(df_sel0['id'])[range(np.max(smaller)+1)]
        
        # deterministic pruning
        df_sel0 = df_all[df_all['pval'] <= df_all['c']].sort_values(by='homo_Rj')
        df_sel0['threshold'] = np.linspace(1, df_sel0.shape[0], num = df_sel0.shape[0])
        smaller = [j for j in range(df_sel0.shape[0]) if df_sel0.iloc[j,5] <= df_sel0.iloc[j,6]]
        if len(smaller) == 0:
            idx_sel_dete = np.array([])
        else:
            idx_sel_dete = np.array(df_sel0['id'])[range(np.max(smaller)+1)]
            
            
        return np.array(df_sel0['id']), idx_sel_hete, idx_sel_homo, idx_sel_dete #, Cj
    
 




def weighted_eBH(calib_scores, calib_weights, test_scores, test_weights, q = 0.1):
    sum_calib_weight = np.sum(calib_weights)
    df_all = pd.concat((pd.DataFrame({"score": calib_scores, "weight": calib_weights[:,0], "cal": 1}), pd.DataFrame({"score": test_scores, "weight": test_weights[:,0], "cal": 0})))
    df_sorted = df_all.sort_values(by = 'score')
    arr_sorted = np.array(df_sorted)
    
    ntest = len(test_scores) 
    hat_fdps = np.zeros(ntest) 
    
    # compute cumulative weights sum_i w(Xi)ind{V_i <= t} 
    df_sorted['cw_calib'] = 0
    df_sorted['cw_test'] = 0
    
    for k in range(df_sorted.shape[0]):
        cw_calib = np.sum(arr_sorted[range(k), 1] * arr_sorted[range(k),2])
        cw_test = max(1, np.sum(1-arr_sorted[range(k),2]))
        df_sorted.iloc[k,3] = cw_calib
        df_sorted.iloc[k,4] = cw_test
    
    w_evals = np.zeros(df_sorted.shape[0])
    for j in range(df_sorted.shape[0]):
        if df_sorted.iloc[j,2] == 0: # test point
            hat_fdps = np.array((df_sorted['cw_calib'] + df_sorted.iloc[j,1]) * ntest / (df_sorted['cw_test'] * (sum_calib_weight + df_sorted.iloc[j, 1])))
            smaller = [s for s in range(df_sorted.shape[0]) if hat_fdps[s]<=q]
            if len(smaller) > 0: 
                Tj = df_sorted.iloc[np.max(smaller), 0] 
                if df_sorted.iloc[j, 0] <= Tj:
                    w_evals[j] = (sum_calib_weight + arr_sorted[j,1]) / (df_sorted.iloc[np.max(smaller), 3] + arr_sorted[j,1])
        else: 
            w_evals[j] = -1
    
    df_sorted['eval'] = w_evals
    df_test = df_sorted[df_sorted['cal']==0]
    
    # eBH(q)
    df_test = df_test.sort_values(by='eval', ascending=False)
    df_test['threshold'] = ntest / (q * np.linspace(1, ntest, num=ntest))
    ebh_smaller = [j for j in range(ntest) if df_test.iloc[j, 5] >= df_test.iloc[j,6]]
    
    if len(ebh_smaller) == 0:
        return(np.array([]))
    else:
        idx_sel = np.array(df_test.index)[range(np.max(ebh_smaller)+1)]
        return(idx_sel)
    
    
    
    
def weighted_eBH_rand_hetero(calib_scores, calib_weights, test_scores, test_weights, q = 0.1, c = 0.1):
    sum_calib_weight = np.sum(calib_weights)
    df_all = pd.concat((pd.DataFrame({"score": calib_scores, "weight": calib_weights[:,0], "cal": 1}), pd.DataFrame({"score": test_scores, "weight": test_weights[:,0], "cal": 0})))
    df_sorted = df_all.sort_values(by = 'score')
    arr_sorted = np.array(df_sorted)
    
    ntest = len(test_scores) 
    hat_fdps = np.zeros(ntest) 
    
    # compute cumulative weights sum_i w(Xi)ind{V_i <= t} 
    df_sorted['cw_calib'] = 0
    df_sorted['cw_test'] = 0
    
    for k in range(df_sorted.shape[0]):
        cw_calib = np.sum(arr_sorted[range(k), 1] * arr_sorted[range(k),2])
        cw_test = max(1, np.sum(1-arr_sorted[range(k),2]))
        df_sorted.iloc[k,3] = cw_calib
        df_sorted.iloc[k,4] = cw_test
    
    w_evals_homo = np.zeros(df_sorted.shape[0])
    w_evals_hetero = np.zeros(df_sorted.shape[0])
    homo_xi = np.random.uniform(size=1)[0]
    for j in range(df_sorted.shape[0]):
        if df_sorted.iloc[j,2] == 0: # test point
            hat_fdps = np.array((df_sorted['cw_calib'] + df_sorted.iloc[j,1]) * ntest / (df_sorted['cw_test'] * (sum_calib_weight + df_sorted.iloc[j, 1])))
            smaller = [s for s in range(df_sorted.shape[0]) if hat_fdps[s]<= c]
            if len(smaller) > 0: 
                Tj = df_sorted.iloc[np.max(smaller), 0] 
                if df_sorted.iloc[j, 0] <= Tj:
                    w_evals_raw = (sum_calib_weight + arr_sorted[j,1]) / (df_sorted.iloc[np.max(smaller), 3] + arr_sorted[j,1])
                    w_evals_hetero[j] = w_evals_raw / np.random.uniform(size=1)[0]
                    w_evals_homo[j] = w_evals_raw / homo_xi
        # else: 
        #     w_evals_hetero[j] = -1
        #     w_evals_hetero[j] = -1
    
    df_sorted['eval_hetero'] = w_evals_hetero
    df_sorted['eval_homo'] = w_evals_homo
    df_test = df_sorted[df_sorted['cal']==0]
    
    # eBH(q) with heterogenous boosting
    df_test = df_test.sort_values(by='eval_hetero', ascending=False)
    df_test['threshold'] = ntest / (q * np.linspace(1, ntest, num=ntest))
    ebh_smaller = [j for j in range(ntest) if df_test.iloc[j, 5] >= df_test.iloc[j,7]]
    
    if len(ebh_smaller) == 0:
        sel_hetero = np.array([]) #return(np.array([]))
    else:
        sel_hetero = np.array(df_test.index)[range(np.max(ebh_smaller)+1)] 
        
    # eBH(q) with homo boosting
    df_test = df_test.sort_values(by='eval_homo', ascending=False)
    df_test['threshold'] = ntest / (q * np.linspace(1, ntest, num=ntest))
    ebh_smaller = [j for j in range(ntest) if df_test.iloc[j, 6] >= df_test.iloc[j,7]]
    
    if len(ebh_smaller) == 0:
        sel_homo = np.array([]) #return(np.array([]))
    else:
        sel_homo = np.array(df_test.index)[range(np.max(ebh_smaller)+1)] 
    
    return sel_homo, sel_hetero, w_evals_raw
        
        
    
