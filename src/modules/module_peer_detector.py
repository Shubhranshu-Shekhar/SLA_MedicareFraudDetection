"""
Module: Peer-based Anomaly Detector
"""
__author__ = "Shubhranshu Shekhar"
__date__ = "12/8/23"

import matplotlib
import numpy as np
import pandas as pd
import logging
import os
import json
import pickle
import time
import bz2
from sklearn.neighbors import NearestNeighbors

from scipy.linalg import norm
from scipy.spatial.distance import euclidean


# defining distance measures to be used in finding peers
_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64


def hellinger1(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2


def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def hellinger3(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2



import numba as nb
@nb.njit(parallel=True)
def _snn_imp_ss(ind, ref_set_):
    """Internal function for fast snn calculation

    Parameters
    ----------
    ind : int
        Indices return by kNN.

    ref_set_ : int, optional (default=10)
        specifies the number of shared nearest neighbors to create the
        reference set. Note that ref_set must be smaller than n_neighbors.

    """
    n = ind.shape[0]
    _count = np.zeros(shape=(n, ref_set_), dtype=np.uint32)
    _dist = np.zeros(shape=(n, n), dtype=np.uint32)
    for i in nb.prange(n):
        temp = np.empty(n, dtype=np.uint32)
        test_element_set = set(ind[i])
        for j in nb.prange(n):
            temp[j] = len(set(ind[j]).intersection(test_element_set))
        temp[i] = np.iinfo(np.uint32).max
        _dist[i] = temp
        _count[i] = np.argsort(temp)[::-1][1:ref_set_ + 1]

    return _count, _dist


def compute_peers(prvdr_mdc_df=None, prvdr_chronic_cdns_df=None, metric=None):
    if prvdr_mdc_df is not None:
        print('Uisng labels from prvdr_mdc_df')
        labels = list(prvdr_mdc_df.index)  # provider ids to be used here
    else:
        labels = list(prvdr_chronic_cdns_df.index)

    # mdc_columns = prvdr_mdc_dist_df.columns
    # mdc_index = dict(zip(list(mdc_columns), range(len(mdc_columns)))) # for book keeping stuff

    if prvdr_chronic_cdns_df is None: # peer type = 'MDC'
        # normalize
        prvdr_mdc_dist_df = prvdr_mdc_df.div(prvdr_mdc_df.sum(axis=1), axis=0)
        # create peers dict based on MDC representation -- snn is better than distance restriction
        # neigh = NearestNeighbors(n_neighbors=20, radius=0.2)
        # nbrs = neigh.fit(prvdr_mdc_arr)
        # distances, indices = nbrs.kneighbors(prvdr_mdc_arr)

        # nn_selected = np.where(distances <= 0.2, True, False)

        neigh = NearestNeighbors(n_neighbors=500, metric=metric)
        nbrs = neigh.fit(prvdr_mdc_dist_df.values)
        distances, indices = nbrs.kneighbors(prvdr_mdc_dist_df.values)

        shared_nn, shared_dist = _snn_imp_ss(indices, 10)

        # prvdr_codes = prvdr_mdc_dist_df.index
        # mdc_peers_dict = {} # used when using distance thresholding
        # for i in range(len(prvdr_codes)):
        #         mdc_peers_dict[prvdr_codes[i]
        #                     ] = prvdr_codes[indices[i][nn_selected[i]][1:]]

        mdc_peers_snn = {}
        for i, n_lst in enumerate(shared_nn):
            tmp = [labels[j] for j in n_lst]
            mdc_peers_snn[labels[i]] = tmp

        return mdc_peers_snn
    
    elif prvdr_mdc_df is None:  # peer type = 'chronic'
        prvdr_chronic_dist_df = prvdr_chronic_cdns_df.div(prvdr_chronic_cdns_df.sum(axis=1), axis=0)
         # compute near neighbor search
        neigh = NearestNeighbors(n_neighbors=500, metric=metric)
        nbrs = neigh.fit(prvdr_chronic_dist_df.values)
        distances, indices = nbrs.kneighbors(prvdr_chronic_dist_df.values)

        shared_nn, shared_dist = _snn_imp_ss(indices, 10)

        chronic_peers_snn = {}
        for i, n_lst in enumerate(shared_nn):
                tmp = [labels[j] for j in n_lst]
                chronic_peers_snn[labels[i]] = tmp

        return chronic_peers_snn
    
    else: # peer type = 'combined. I prefer this one
        # First individually normalize the two dataframes
        # normalize
        prvdr_mdc_dist_df = prvdr_mdc_df.div(prvdr_mdc_df.sum(axis=1), axis=0)
        prvdr_chronic_dist_df = prvdr_chronic_cdns_df.div(
            prvdr_chronic_cdns_df.sum(axis=1), axis=0)

        # 1. combine the two normalized dataframe
        combined_df = prvdr_mdc_dist_df.merge(
            prvdr_chronic_dist_df, left_index=True, right_index=True)
        
        # 2. compute near neighbor search on comined df
        neigh = NearestNeighbors(n_neighbors=500, metric=metric)
        nbrs = neigh.fit(combined_df.values)
        distances, indices = nbrs.kneighbors(combined_df.values)

        shared_nn, shared_dist = _snn_imp_ss(indices, 10)

        combined_peers_snn = {}
        for i, n_lst in enumerate(shared_nn):
            tmp = [labels[j] for j in n_lst]
            combined_peers_snn[labels[i]] = tmp

        return combined_peers_snn


def get_peers_avg_dstn(peers_dstn, peers_claims_counts, peer_wts=None):
    avg_cf_peers_ = peers_dstn[0]*peers_claims_counts[0]

    if not peer_wts:
        for i in range(1, peers_dstn.shape[0]):
            avg_cf_peers_ += peers_dstn[i]*peers_claims_counts[i]
    else:
        for i in range(1, peers_dstn.shape[0]):
            avg_cf_peers_ += peers_dstn[i]*peers_claims_counts[i]*peer_wts[i-1]

    avg_cf_peers_ /= np.sum(avg_cf_peers_)
    return avg_cf_peers_


def get_excess(dstn, cf_dstn, drg_lst, drg_price_dct):
    # normal flow
    exp_p = 0
    for i, d_ in enumerate(drg_lst):
        d_ = str(int(d_)).zfill(3)
        if d_ in drg_price_dct:
            exp_p += drg_price_dct[d_]*dstn[i]

    exp_q = 0
    for i, d_ in enumerate(drg_lst):
        d_ = str(int(d_)).zfill(3)
        if d_ in drg_price_dct:
            exp_q += drg_price_dct[d_]*cf_dstn[i]

    # EMD distance flow
    # weights_ = [drg_price_dct[d_] if d_ in drg_price_dct else 0 for d_ in drg_lst]

    return exp_p - exp_q
    # return wasserstein_distance(dstn, cf_dstn, u_weights=weights_, v_weights=weights_)


def get_peer_excess(prvdr_dstn, prvdr_names, prvdr_peers_dct, drg_lst, drg_price_dct, prvdr_counts, if_avg_dstn=True,
                    filter_prvdrs=None, peer_weights=None):
    prvdr_excess = []
    idx_dct = dict(zip(prvdr_names, range(len(prvdr_names))))

    for i, p_dstn in enumerate(prvdr_dstn):
        p_ = prvdr_names[i]
        p_peers = prvdr_peers_dct[p_]
        p_peers = [k_ for k_ in p_peers if k_ in prvdr_counts]

        p_counts = prvdr_counts
        if filter_prvdrs:
            p_peers = [k_ for k_ in p_peers if k_ not in filter_prvdrs]
        p_peers_cnts = [prvdr_counts[k_] for k_ in p_peers]
        idxs = [idx_dct[k_] for k_ in p_peers if k_ in idx_dct]
        peers_dstn = prvdr_dstn[idxs]
        if peers_dstn.shape[0] < 5:
            ex_ = 0
            print('Very few peers')
        else:
            if if_avg_dstn:
                wts = None
                if peer_weights:
                    wts = [peer_weights[k_] for k_ in p_peers]

                q_dstn = get_peers_avg_dstn(peers_dstn, p_peers_cnts, wts)
                ex_ = get_excess(p_dstn, q_dstn, drg_lst, drg_price_dct)
            else:
                n_peers = len(peers_dstn)
                sum_excess = 0
                for q_dstn in peers_dstn:
                    sum_excess += get_excess(p_dstn,
                                             q_dstn, drg_lst, drg_price_dct)
                ex_ = sum_excess / n_peers
        prvdr_excess.append(ex_)

    return prvdr_excess


def get_DRG_rep(drg_cnts, global_drg_lst, drg_idx):
    ret = np.zeros(len(global_drg_lst))
    for d_ in drg_cnts:
        i = drg_idx[d_]
        ret[i] = drg_cnts[d_]
    return ret


def convert_drg_counts_to_drg_dstn(provider_DRG_count, global_drg_lst, global_drg_idx):
    provider_DRG_dstn = {}
    for p in provider_DRG_count.keys():
        drg_cnts = provider_DRG_count[p]
        drg_rep = get_DRG_rep(drg_cnts, global_drg_lst, global_drg_idx)
        if np.sum(drg_rep) > 0:
            drg_dstn = drg_rep/np.sum(drg_rep)

        # save the dstn
        provider_DRG_dstn[p] = drg_dstn
    return provider_DRG_dstn


def get_sum_counts_prvdr_drglst(prvdr_drg_count, prvdr, drg_lst):
    s = 0
    if prvdr in prvdr_drg_count:
        dct_ = prvdr_drg_count[prvdr]
        for d_ in drg_lst:
            d_ = str(int(d_))
            if d_ in dct_:
                s += dct_[d_]

    return s


def find_peer_outliers(yr='2017', data_load_base_path='../output_ER/'):
    # load mdc counts per provider
    prvdr_mdc_counts = json.load(open(data_load_base_path + yr + '/' + 'prvdr_mdc_counts_dct.json'))

    # create dataframe out of it
    prvdr_mdc_df = pd.DataFrame.from_dict(prvdr_mdc_counts, orient='index').fillna(0)
    prvdr_mdc_df.columns = prvdr_mdc_df.columns.astype(str)
    prvdr_mdc_df = prvdr_mdc_df.reindex(sorted(prvdr_mdc_df.columns), axis=1) # MDC columns are sorted

    # load provider chronic conditions counts dataframe
    with bz2.BZ2File(data_load_base_path + yr + '/' + "prvdr_chronic_counts_df.bz2.pkl", 'rb') as f:
        prvdr_chronic_counts_df = pickle.load(f)
    
    # given two dataframes - get peers dict for each provider
    

    if os.path.exists(data_load_base_path + 'ER_combined_prvdr_peer.bz2.pkl'):
        with bz2.BZ2File(data_load_base_path + 'ER_combined_prvdr_peer.bz2.pkl', 'rb') as f:
            prvdr_peers = pickle.load(f)
    else:
        prvdr_peers = compute_peers(
            prvdr_mdc_df=prvdr_mdc_df, prvdr_chronic_cdns_df=prvdr_chronic_counts_df, metric=hellinger1)
    
        with bz2.BZ2File(data_load_base_path + 'ER_combined_prvdr_peer.bz2.pkl', 'wb') as f:
            pickle.dump(prvdr_peers, f)

    print("Peers computed and saved! \n Starting the excess amount calculation...")


    ## Now use DRG distributions and their base prices to find excess expenditure to find ranked outliers
    # start by first creating global DRG lst
    prvdr_drg_counts = json.load(open(data_load_base_path + yr +
              '/' + 'provider_drg_counts.json'))
    drg_lst_total = list(
        set([k for i, dct_ in prvdr_drg_counts.items() for k in dct_]))
    drg_lst = sorted(list(set(drg_lst_total)))
    drg_idx = dict(zip(drg_lst, range(len(drg_lst))))

    # load provider DRG related details
    drg_median_price = json.load(
        open(data_load_base_path + yr + '/' + "DRG_median_base.json"))
    # create provider distribution over DRGs
    # below array follows indexing by prvdr_lst
    prvdr_lst = [k for k in prvdr_drg_counts.keys()]
    
    dstn_ = convert_drg_counts_to_drg_dstn(prvdr_drg_counts, drg_lst, drg_idx) 
    dstn_ = list(dstn_.values())
    dstn_ = np.array(dstn_)
    prvdr_dstn = dstn_ # as np array

    # compute claim counts for each provider
    provider_claim_counts = [get_sum_counts_prvdr_drglst(prvdr_drg_counts, p_, drg_lst)
                             for p_ in prvdr_lst]
    provider_claim_counts = dict(zip(prvdr_lst, provider_claim_counts))

    # now finally given all of the meta data compute excess amount
    provider_excess = get_peer_excess(prvdr_dstn, prvdr_lst,  # prvdr_lst_common,
                                      prvdr_peers, drg_lst,
                                      drg_median_price,
                                      provider_claim_counts, if_avg_dstn=True)
    
    provider_excess_dct = dict(zip(prvdr_lst, provider_excess))
    # provider_excess_dct = sorted(
    #     provider_excess_dct.items(), key=lambda x: x[1], reverse=True)
    
    i = 0
    for key, value in provider_excess_dct.items():
        print(key, ':', value)
        i = i + 1
        if i ==10:
            break


    # store the results -- change path for ER/non-ER result storeing
    with bz2.BZ2File(data_load_base_path + 'ER_peer_excess_amount.bz2.pkl', 'wb') as f:
        pickle.dump(provider_excess_dct, f)
    print("Done.")


if __name__ == '__main__':
    start = time.time()
    print("Starting peer based modeling...")

    # running on all data
    # find_peer_outliers(yr='2017', data_load_base_path='../output/')

    # running for ER data
    # ER data is already preprocessed and stired in output_ER directory
    find_peer_outliers(yr='2017', data_load_base_path='../output_ER/')

    

    end = time.time()
    print('Peer-based model trained and scored. Total elapsed time: {} seconds'.format(
        end - start))



        










