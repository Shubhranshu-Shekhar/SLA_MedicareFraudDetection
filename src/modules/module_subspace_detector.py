from sklearn.linear_model import LassoCV, LinearRegression
import time
import bz2
import pickle
import json
import os
import numpy as np
import pandas as pd
from pyod.models.sod import SOD
from pyod.models.loda import LODA
from pyod.models.iforest import IForest
from pysad.models import RobustRandomCutForest #, RSHash
import dill

# based on the code from https://github.com/PalinkasAljoscha/RShash_outliers


np.float = float

class RShash():
    """ score datapoints as outliers 
    
    Implementation of rs-hash algorithm for outlier scoring according to 
    Sathe and Aggarwal "Subspace Outlier Detection in Linear Time with Randomized Hashing", DOI: 10.1109/ICDM.2016.0057
    """

    def __init__(self, n_hashes=4, n_samples=1000, n_runs=300, seed=None):
        self.n_hashes = n_hashes
        self.n_samples = n_samples
        self.n_runs = n_runs
        np.random.seed(seed)
        self.seeds_per_run = np.random.randint(
            low=0, high=np.power(2, 30), size=n_runs)

    def score(self, data):
        n_data = data.shape[0]
        # initilise array to store scores of all runs (for insight into outliers)
        scores_all_runs = np.zeros((self.n_runs, n_data))
        # alternatively only get the average score, updated in each run
        # avg_score = 0
        for k, seed in enumerate(self.seeds_per_run):
            # call function to put all data points into a random defined grid using the seed passed to the function
            y_bar, sample_obs, dim_keep = self._put_data_in_grid(data, seed)
            # in order to apply n_hash differnt hashings, append a number from 0 to n_hash-1 to each element of y_bar
            data_arrays_to_hash = np.hstack((np.tile(y_bar, (self.n_hashes, 1)),
                                             np.reshape(np.repeat(range(self.n_hashes), n_data),
                                                        (self.n_hashes*n_data, 1))
                                             )
                                            )
            # the counts in the hash table are done for the observation sample only, select these elements
            sample_to_hash = data_arrays_to_hash[np.concatenate(
                [sample_obs+k*n_data for k in range(self.n_hashes)]),]
            # and then cerate hashtable with counts
            hashtab = {}
            for arr in sample_to_hash:
                hashtab[arr.data.tobytes()] = hashtab.get(
                    arr.data.tobytes(), 0) + 1
            # then assign these counts to the whole population
            all_counts = np.array([hashtab.get(data_array.data.tobytes(), 0)
                                  for data_array in data_arrays_to_hash])
            # get score, i.e., take the minimum of the counts in all n_hash hash tables
            # and add +1 to the counts for out of sample points and then take the log
            score = np.log(
                np.reshape(all_counts, (self.n_hashes, n_data)).min(axis=0)
                + (1 - np.isin(np.array(range(n_data)),
                   sample_obs, assume_unique=True))
            )
            # write this score into results array
            scores_all_runs[k, :] = score
            # below line of not all runs are stored and average updated in each run
            # avg_score  = avg_score*(k/(k+1)) + score/(k+1)

        return scores_all_runs.mean(axis=0)

    # put each data point in a random defined grid
    def _put_data_in_grid(self, data, seed):
        n_data = data.shape[0]
        n_dim = data.shape[1]
        np.random.seed(seed)
        # sample locality parameter f (step 1 in paper)
        f = np.random.uniform(np.power(self.n_samples, -0.5),
                              1-np.power(self.n_samples, -0.5))
        assert (f > np.power(self.n_samples, -0.5))
        # get r, number of dimensions to use in this hash run
        log_of_s = np.log(self.n_samples) / np.log(np.maximum(2, (1/f)))
        r = np.random.randint(low=np.round(1+0.5*log_of_s),
                              high=np.ceil(log_of_s), size=1)
        r = np.minimum(r, n_dim)
        # get sample of dimensions to use
        sample_dims = np.random.choice(
            range(n_dim), size=np.minimum(r, n_dim), replace=False)
        # and sample of observations to use
        sample_obs = np.random.choice(
            range(1, n_data), size=self.n_samples, replace=False)
        # get min and max overservation sample in each dimension
        dim_min = data[sample_obs].min(axis=0)
        dim_max = data[sample_obs].max(axis=0)
        # drop from sampled dimensions those that are constant over the observation sample
        dim_keep = np.intersect1d(np.where(dim_min < dim_max)[0], sample_dims)
        # from obs to y_bar and to hash dict
        # linear affine transformmation of the observation to y_bar (notation from paper), i.e.,
        # first step, so that [0,1] is the range in each kept dimension
        # then scaled by 1/f and shifted with a random number from [0,1] in each dimension
        y_bar = np.floor(
            ((np.take(data, dim_keep, axis=1) - dim_min[dim_keep])
             / (dim_max[dim_keep] - dim_min[dim_keep])
             )/f
            + np.random.rand(dim_keep.shape[0],)
        )
        return y_bar, sample_obs, dim_keep


def process_ER_data():
    # check if processed data is laready located at "'../data/ER/prvdr_icd10_jaccard_df.bz2.pkl'"
    if os.path.exists('../data/ER/prvdr_icd10_jaccard_df.bz2.pkl'):
        with bz2.BZ2File('../data/ER/prvdr_icd10_jaccard_df.bz2.pkl', 'rb') as f:
            df_jaccard = pickle.load(f)
        print("Returning preloaded data.")
        return df_jaccard

    # if not then
    # Use data from output_ER

    # Load icd9 data for year 2017 -- Using ER data only
    provider_icd9_dgns = json.load(open('../output_ER/2017/provider_icd9_dgns.json'))
    provider_icd9_prcdr = json.load(open('../output_ER/2017/provider_icd9_prcdr.json'))
    
    # create dataframe from dict
    data, index = list(provider_icd9_dgns.values()), list(provider_icd9_dgns.keys())
    df1 = pd.DataFrame(data, index=index).fillna(0)

    print("Raw ER data loaded.")

    # to compute the jccard sim incorporated matrix, load icd similarity
    with bz2.BZ2File("../data/icd10icd10sim.bz2.pkl", 'rb') as f:
        prior_computed_sim = pickle.load(f)
    
    (icd10icd10sim, icd10_codes) = (prior_computed_sim[0], prior_computed_sim[1])
    print("Precomputed ICD data loaded.")

    # now I need to incorporate ICDcode sismilarity. Remeber not all codes will be used in ER data,
    # therefore I first select ICDcodes that are used, and then do a matrix multiplication
    mask_selected_icd10 = np.ones_like(icd10_codes, dtype=bool)
    df1c = df1.columns
    for i, c_ in enumerate(icd10_codes):
        if c_ not in df1c:
            mask_selected_icd10[i] = False

    selected_icd10_codes = icd10_codes[mask_selected_icd10]
    selected_icd10icd10sim = icd10icd10sim[np.ix_(mask_selected_icd10, mask_selected_icd10)]

    df1_jaccard = df1[selected_icd10_codes] @ selected_icd10icd10sim # note I'm using selected codes
    df1_jaccard.columns = selected_icd10_codes # rearrange coloumn names abased on selected codes
    print("ICD similarity incorporated loaded.")
    
    data, index = list(provider_icd9_prcdr.values()), list(provider_icd9_prcdr.keys())
    df2 = pd.DataFrame(data, index=index).fillna(0)


    df_jaccard = pd.merge(df1_jaccard, df2, left_index=True, right_index=True)
    print("Data prepared.")
    
    # save jaccard df -- for later user
    with bz2.BZ2File('../data/ER/prvdr_icd10_jaccard_df.bz2.pkl', 'wb') as f:
        pickle.dump(df_jaccard, f)
    print("Data saved.")
    return df_jaccard


def run_ensemble(df_jaccard, results_file_name=""):
    # with bz2.BZ2File('../data/prvdr_icd10_jaccard_df.bz2.pkl', 'rb') as f:
    #     df_jaccard = pickle.load(f)


    X = df_jaccard.values  # df.values
    # feature_mins, feature_maxes = X.min(axis=0), X.max(axis=1)

    print("Data shape:", X.shape)

    # fit and save model

    methods = ['sod', 'ifr', 'loda', 'rshash', 'rcf']
    models = []
    scores = []
    seed = 42
    for m_ in methods: 
        print("Processing model: ", m_)
        if m_ == 'sod':
            model = SOD(contamination=0.05, n_neighbors=20, ref_set=10, alpha=0.8)
        elif m_ == 'ifr':
            model = IForest(n_estimators=1000, n_jobs=-2, behaviour='new', random_state=seed)
        elif m_ == 'loda':
            model = LODA(contamination=0.05, n_bins="auto", n_random_cuts=100)
        elif m_ == 'rshash':
            # model = RSHash_my(feature_mins, feature_maxes, sampling_points=1000
            #                , decay=0.015, num_components=100, num_hash_fns=8)
            model = RShash(n_hashes=8, n_samples=1000, n_runs=1000, seed=seed)
        elif m_ == 'rcf':
            model = RobustRandomCutForest(num_trees=100, shingle_size=8, tree_size=256)

        # begin training
        if m_ in ['sod', 'ifr', 'loda']:  # pyod models
            model = model.fit(X)
            score_ = model.decision_scores_

            models.append(model)
            scores.append(score_)
            print("Fitted and scored!")
        elif m_ == "rshash":
            print("Training RSHash...")
            score_ = model.score(X)

            models.append(model)
            scores.append(score_)
        else:
            print("Training RRCF...")
            model = model.fit(X)
            score_ = model.score(X)
            
            models.append(models)
            scores.append(score_)
            print("Fitted and scored!")

    # SAVE THE RESULTS
    try:
        with bz2.BZ2File('../output/' + results_file_name, 'wb') as f:
            dill.dump([methods, models, scores], f)
    except:
        with bz2.BZ2File(results_file_name, 'wb') as f:
            dill.dump([methods, models, scores], f)





if __name__ == '__main__':
    # model = RShash(n_hashes=8, n_samples=1000, n_runs=1000, seed=42)
    start = time.time()
    print("Starting subspace fitting...")
    
    with bz2.BZ2File('../data/prvdr_icd10_jaccard_df.bz2.pkl', 'rb') as f:
        df_jaccard = pickle.load(f)
    
    print("Begin modeule subspace...")
    run_ensemble(df_jaccard=df_jaccard,
         results_file_name="subspace_module_rshash_rrcf.bz.dill")
    
    end = time.time()
    print('Subspace models trained. Total elapsed time: {} seconds'.format(
        end - start))

