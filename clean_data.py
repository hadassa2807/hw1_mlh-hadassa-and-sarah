# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    #c_ctg = CTG_features.copy()
    #c_ctg = c_ctg.apply(pd.to_numeric, errors='coerce')
    #c_ctg = pd.to_numeric(c_ctg, errors='coerce')
    #c_ctg.drop(columns=['DR'], inplace=True)
    c_ctg = {}
    lft = list(CTG_features.columns.values)
    for ft in lft:
        if ft == extra_feature:
            continue
        else:
            col = CTG_features[ft].copy()
            col = pd.to_numeric(col, errors='coerce')
            col = col.dropna()
            c_ctg[ft] = col
        # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    lft = list(CTG_features.columns.values)
    for ft in lft:
        if ft == extra_feature:
            continue
        else:
            col = CTG_features[ft].copy()
            col = pd.to_numeric(col, errors='coerce')

            nan_list = np.array(col.isnull())
            val_list = col[col.notnull()].array
            for i in range(len(nan_list)):
                if nan_list[i] == True:
                    col.iloc[i] = np.random.choice(val_list)
            c_cdf[ft] = col
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {}
    lft = list(c_feat.columns.values)
    for ft in lft:
        col = c_feat[ft]
        ftdict = {"min": col.min(),
                    "Q1": col.quantile(0.25),
                    "median": col.quantile(0.5),
                    "Q3": col.quantile(0.75),
                    "max":col.max()}
        d_summary[ft] = ftdict
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    lft = list(c_feat.columns.values)
    for ft in lft:
        ft_dict = d_summary[ft]
        col = c_feat[ft].copy()
        cutoff = (ft_dict['Q3'] - ft_dict['Q1']) * 1.5
        down, up = ft_dict['Q1'] - cutoff, ft_dict['Q3'] + cutoff
        col.loc[(col > up) | (col < down)] = np.nan
        c_no_outlier[ft] = col
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    non_filt = c_cdf[feature]
    filt_feature = []
    for item in non_filt:
        if item <= thresh:
            filt_feature.append(item)
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    nsd_res = CTG_features.copy()
    lft = list(nsd_res.columns.values)
    for ft in lft:
        col = nsd_res[ft]
        if mode == 'standard':
            col = np.divide(col - col.mean(), col.std())
        elif mode == 'MinMax':
            col = np.divide(col - col.min(), (col.max() - col.min()))
        elif mode == 'mean':
            col = np.divide(col - col.mean(), (col.max() - col.min()))
        nsd_res[ft] = col
    if flag==True:  
        xlbl = ['Units {x} before','Units {y} before']
        axarr = CTG_features.hist(column=[x, y], bins=100,layout = (1, 2),figsize=(10, 5))
        for i,ax in enumerate(axarr.flatten()):
            ax.set_xlabel(xlbl[i])
            ax.set_ylabel("Count")
        xlbl2 = ['Units {x} after','Units {y} after']
        axarr2 = nsd_res.hist(column=[x, y], bins=100,layout = (1, 2),figsize=(10, 5))
        for i,ax in enumerate(axarr2.flatten()):
            ax.set_xlabel(xlbl2[i])
            ax.set_ylabel("Count")
        plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
