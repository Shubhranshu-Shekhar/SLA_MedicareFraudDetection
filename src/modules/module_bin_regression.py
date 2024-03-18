from sklearn.linear_model import LinearRegression
import time
import bz2
import pickle
import numpy as np

def do_visit_to_hopsital_regression(X, y):
    '''
    Param X: Comntains the features including patient history encoded by
            counts of different codes for last five yesrs, and also 
            includes counts for hospital visits. 
    Param y: Expenditure on inpatient hospitalization for a given beneficiary in the current year
    '''
    

    y = np.array(y)
    index_of_importance = np.where(y > 0)[0] # filter out only positive expenses

    reg_ = LinearRegression()
    reg_.fit(X, y)

    print("R^2 = ", reg_.score(X, y))

    return reg_


if __name__ == '__main__':
    start = time.time()
    with bz2.BZ2File('../data/sum_mat_target.bz2.pkl', 'rb') as f:
        [X, y] = pickle.load(f)
    num_hospitals = 3855 # last 3855 columns of X ontain visit counts
    print("Starting regression...")
    reg_model = do_visit_to_hopsital_regression(X, y)
    hospital_coefficients = reg_model.coef_[:-num_hospitals]
    end = time.time()
    print('Regression completed. Total elapsed time: {} seconds'.format(end - start))

