import pandas as pd
import numpy as np
import random
import math
import scipy
from scipy import stats

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # for train/test split
from statkit.decision import net_benefit
from scipy import integrate

import dca_fs as dcafs
from sklearn.metrics import roc_auc_score
import torch

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt  #For representation
from sklearn.svm import l1_min_c  # for L1 regluarization path
from sklearn.linear_model import LogisticRegression

def make_class_dataset(n_sample = 100,
                       n_features = 3,
                       n_redundant = 0,
                       random_state = 2001,
                       n_informative = 3,
                       n_clusters_per_class = 1,
                       n_classes = 2):

    """
    Use sklear.datasets make_classification to make a synthetic data set for testing
    
    Parameters:
    - n_sample (integer): numer of samples.
    -

    Returns:
    - pd.dataframe: Training data set.
    - pd.dataframe: Testing data set/
    """
        
    # generate data set
    X, Y = make_classification(n_samples=n_sample, 
                               n_features=n_features, 
                               n_redundant=n_redundant, 
                               random_state=random_state, 
                               n_informative=n_informative, 
                               n_clusters_per_class=n_clusters_per_class,
                                n_classes=n_classes)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Names of independent variables
    ind_var_names = ["x"+str(i) for i in range(n_features)]

    # Rename output variables
    X = pd.DataFrame(X, columns = ind_var_names)
    Y = pd.DataFrame(Y, columns = ["y"])

    # compose output dataframe 
    df = pd.concat([Y.reset_index(drop=True), X], axis=1)

    # Train test split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1234)

    return df_train,df_test, ind_var_names


# Make synthetic data #
#=====================#


def make_synth_data(n_features = 3, 
                    n_sample = 1000, 
                    true_coefs = [1.,2.,3.], 
                    noise = 'none', 
                    true_bias = 0.,
                    heteros = 0.):
    # initialise
    df = pd.DataFrame()

    # Standardization of variables
    mean_vars = [0 for i in range(n_features)]
    var_vars = [1 for i in range(n_features)]

    # Amount of noise to add to  each variable
    #noise_scale = [i**2 for i in range(n_features)]
    #noise_scale = [0.05*i/(n_features-1)**2 for i in noise_scale]
    if noise == 'none':
        noise_scale = [0 for i in range(n_features)]
    else:
        noise_scale = noise

    # create normal variables
    for n in range(n_features):
        df[n] =  np.random.normal(loc=mean_vars[n], scale=var_vars[n], size=n_sample)

    # create linear combination
    df["x"] = 0
    for n in range(n_features):
        df["x"] +=  true_coefs[n]*df[n]

    # add heteroscedacity
    df["x"] = df.apply(lambda row: row["x"] + heteros * np.random.normal(loc=0, scale = 1./(1.+np.exp(-row["x"])) ), axis=1)

    
    # include the bias
    df["x"] += true_bias

    
    # Generate outcome

    df["p"] = df.apply(lambda row: 1./(1. + math.exp(-row["x"])), axis=1)
    df["y"] = df.apply(lambda row: np.random.binomial(1, row["p"] ), axis=1)

    # Add noise to each variable
    df_noisy = df.copy()

    for n in range(n_features):
        df_noisy[n] += np.random.normal(loc=0.0, scale=noise_scale[n], size=n_sample)

    df = df.drop(["x", "p"], axis=1)
    df_noisy = df_noisy.drop(["x", "p"], axis=1)

    # rename the indpendent variables
    ind_var_rename = dict()
    for ind_var_in in range(n_features) :
        ind_var_rename[ind_var_in] = "x"+f"{ind_var_in}"
    df = df.rename(columns=ind_var_rename)
    df_noisy = df_noisy.rename(columns=ind_var_rename)

    ind_var_names = list(ind_var_rename.values())


    # Split the data
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    df_train, df_test, df_noisy_train, df_noisy_test = train_test_split(df, df_noisy, test_size=0.2, random_state=1234)

    # Standardize each split

    for indep_var in ind_var_names:
        df_train[indep_var] = stats.zscore(df_train[indep_var]) 
        df_test[indep_var] = stats.zscore(df_test[indep_var]) 

        df_noisy_train[indep_var] = stats.zscore(df_noisy_train[indep_var]) 
        df_noisy_test[indep_var] = stats.zscore(df_noisy_test[indep_var]) 

    # scale data
    #sc = StandardScaler()
    #df_train = sc.fit_transform(df_train)
    #X_test = sc.transform(X_test)

    return df_train, df_test, df_noisy_train, df_noisy_test, ind_var_names, n_features, true_coefs




def mean_net_benefit(y_true, y_pred, n_thresh):
    thresh = np.linspace(0.0, 1.0, num=n_thresh)

    nb = net_benefit(y_true, y_pred , thresholds=thresh)

    mnb = integrate.simpson(nb[1], x=nb[0])

    return({
        'mnb': mnb,
        'benefit': nb[1],
        'thresholds': nb[0] 
    })





def lr_skl_boot(lr_object,
                df_boot,
                df_test,
                dependent = "y",
                independent_in = None,
                label = "scikit_learn"):
    
    if independent_in == None:
        independent = list(df_boot.drop([dependent], axis=1).columns)
    else:
        independent = independent_in

    lr_object.fit(df_boot[independent], df_boot[dependent])

    #coefs
    coefs = pd.DataFrame(lr_object.coef_, columns=independent)

    #preds
    pred = lr_object.predict_proba(df_test[independent])

    # auc
    auc = roc_auc_score(df_test[dependent],pred[:, 1])

    # mnb
    mnb = mean_net_benefit(df_test[dependent], pred[:, 1], n_thresh=100)['mnb']

    out = pd.DataFrame({
        "label": [label],
        "auc" : [auc],
        "mnb": [mnb]
    })

    out = pd.concat([out, coefs], axis=1)

    return out.reset_index().drop("index", axis=1)





def lr_boot(df_boot,
            df_test,
            n_epochs=1000, 
            learn_rate=  1.0, 
            loss_fun = 'log',
            dependent = "y",
            independent_in = None,
            label = "torch"):
    
    if independent_in == None:
        independent = list(df_boot.drop([dependent], axis=1).columns)
    else:
        independent = independent_in

    #FIT
    torch_run = dcafs.lr_train(df_boot, n_epochs=n_epochs, learn_rate=learn_rate, loss_fun = loss_fun)['net']

    #coefs
    coefs = pd.DataFrame(torch_run.linear.weight.detach().numpy(), columns=independent)

    #preds
    test_train_dataset_orig = dcafs.SynthDataset(df_test, "y", None)
    test_x = torch.from_numpy(test_train_dataset_orig.x.to_numpy().astype(np.float32) )
    pred = torch.sigmoid(torch_run(test_x )).detach().numpy()

    # auc
    auc = roc_auc_score(df_test['y'],pred)

    #mnb
    mnb = mean_net_benefit(df_test['y'], pred, n_thresh=100)['mnb']

    out = pd.DataFrame({
        "label": [label],
        "auc" : [auc],
        "mnb": [mnb]
    })

    out = pd.concat([out, coefs], axis=1).reset_index().drop("index", axis=1)

    return out



def make_net_prediction(net, data, dependent = "y", independent = None):
    """
    Given a torch logistic regression network make predictions on a dataset
    
    Parameters:
    - net (torch network): network.
    - data (pandas df) : data set including independent and dependent variables
    - dependent (string) : Name of independent variable
    - indepenetent (list of strings) : Names of independent variables

    Returns:
    - np.array: prediction probabilities of positive outcome.
    """

    preped_data = dcafs.SynthDataset(data, dependent, independent)

    preped_data = torch.from_numpy(preped_data.x.to_numpy().astype(np.float32) )

    preds = torch.sigmoid(net(preped_data).detach()).numpy()

    return preds


def my_multi_hist(df_in, col, ax, leg_label, color,  kwargs):    
    first = True
    for l in df_in['label'].unique():

        df=df_in.copy()
        df = df[df['label']==l]
        df = df[col]

        if first==True:
            lab = leg_label
            first = False
        else:
            lab = ""

        ax.hist(df, **kwargs, label = lab, histtype=u'step', density=True, color=color)



def plot_bootstrap(skl_boot, torch_boot_log, torch_boot_mnb):
    kwargs = dict(alpha=0.5, bins=100)

    color_1 = "blue"
    color_2 = "red"
    color_3 = "green"

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(6, 2)

    ax1 = fig.add_subplot(gs[0:2, 0])
    ax2 = fig.add_subplot(gs[3:5, 0])
    ax3 = fig.add_subplot(gs[0:1, 1])
    ax4 = fig.add_subplot(gs[2:3, 1])
    ax5 = fig.add_subplot(gs[4:5, 1])

    # AUC distribution
    #ax1.hist(skl_boot["auc"], **kwargs, label='scikit learn', histtype=u'step', density=False)
    #ax1.hist(torch_boot_log["auc"], **kwargs, label='pytorch implementation', histtype=u'step', density=False)
    #ax1.hist(torch_boot_mnb["auc"], **kwargs, label='pytorch MNB', histtype=u'step', density=False)

    my_multi_hist(skl_boot, "auc", ax1, 'scikit learn', color_1, kwargs)
    my_multi_hist(torch_boot_log, "auc", ax1, 'pytorch implementation', color_2, kwargs)
    my_multi_hist(torch_boot_mnb, "auc", ax1, 'pytorch MNB', color_3, kwargs)

    ax1.legend(loc='upper left')
    ax1.set_xlabel("AUC")
    ax1.set_ylabel("Count")
    ax1.set_title("AUC")

    # Mean Net-benefit distribution
    #ax2.hist(skl_boot["mnb"], **kwargs, label='scikit learn', histtype=u'step', density=False)
    #ax2.hist(torch_boot_log["mnb"], **kwargs, label='pytorch implementation', histtype=u'step', density=False)
    #ax2.hist(torch_boot_mnb["mnb"], **kwargs, label='pytorch MNB', histtype=u'step', density=False)

    my_multi_hist(skl_boot, "mnb", ax2, 'scikit learn', color_1, kwargs)
    my_multi_hist(torch_boot_log, "mnb", ax2, 'pytorch implementation', color_2, kwargs)
    my_multi_hist(torch_boot_mnb, "mnb", ax2, 'pytorch MNB', color_3, kwargs)


    #ax2.legend(loc='upper left')
    ax2.set_xlabel("Mean Net-benefit")
    ax2.set_ylabel("Count")
    ax2.set_title("Mean Net-benefit")


    # Parameters
    #-------------#
    #x0
    #---
    #ax3.hist(skl_boot["x0"], **kwargs,  label='scikit learn', histtype=u'step', density=False)
    #ax3.hist(torch_boot_log["x0"], **kwargs, label='pytorch implementation', histtype=u'step', density=False)
    #ax3.hist(torch_boot_mnb["x0"], **kwargs, label='pytorch MNB', histtype=u'step', density=False)

    my_multi_hist(skl_boot, "x0", ax3, 'scikit learn', color_1, kwargs)
    my_multi_hist(torch_boot_log, "x0", ax3, 'pytorch implementation', color_2, kwargs)
    my_multi_hist(torch_boot_mnb, "x0", ax3, 'pytorch MNB', color_3, kwargs)

    #ax3.legend(loc='upper left')
    ax3.set_xlabel("Parameter x0")
    ax3.set_ylabel("Count")
    ax3.set_title("Parameter: x0")

    #x1
    #---
    #ax4.hist(skl_boot["x1"], **kwargs,  label='scikit learn', histtype=u'step', density=False)
    #ax4.hist(torch_boot_log["x1"], **kwargs, label='pytorch implementation', histtype=u'step', density=False)
    #ax4.hist(torch_boot_mnb["x1"], **kwargs, label='pytorch MNB', histtype=u'step', density=False)

    my_multi_hist(skl_boot, "x1", ax4, 'scikit learn', color_1, kwargs)
    my_multi_hist(torch_boot_log, "x1", ax4, 'pytorch implementation', color_2, kwargs)
    my_multi_hist(torch_boot_mnb, "x1", ax4, 'pytorch MNB', color_3, kwargs)

    #ax3.legend(loc='upper left')
    ax4.set_xlabel("Parameter x1")
    ax4.set_ylabel("Count")
    ax4.set_title("Parameter: x1")

    #x2
    #---
    #ax5.hist(skl_boot["x2"], **kwargs,  label='scikit learn', histtype=u'step', density=False)
    #ax5.hist(torch_boot_log["x2"], **kwargs, label='pytorch implementation', histtype=u'step', density=False)
    #ax5.hist(torch_boot_mnb["x2"], **kwargs, label='pytorch MNB', histtype=u'step', density=False)

    my_multi_hist(skl_boot, "x2", ax5, 'scikit learn', color_1, kwargs)
    my_multi_hist(torch_boot_log, "x2", ax5, 'pytorch implementation', color_2, kwargs)
    my_multi_hist(torch_boot_mnb, "x2", ax5, 'pytorch MNB', color_3, kwargs)

    #ax3.legend(loc='upper left')
    ax5.set_xlabel("Parameter x2")
    ax5.set_ylabel("Count")
    ax5.set_title("Parameter: x2")

    return plt.show()




def skl_reg_path(data,
                dependent = "y",
                independent_in = None,
                log_space_min = 0,
                log_space_max = 10,
                log_space_steps = 16,
                warm_start = False):

    
    if independent_in == None:
        independent = list(data.drop([dependent], axis=1).columns)
    else:
        independent = independent_in

    c_steps = l1_min_c(data[independent], data[dependent], loss="log") * np.logspace(log_space_min, log_space_max, log_space_steps)

    lr = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        tol=1e-6,
        max_iter=int(1e6),
        warm_start=warm_start,
        intercept_scaling=10000.0,
    )
    coefs_ = []
    for c in c_steps:
        lr.set_params(C=c)
        lr.fit(data[independent], data[dependent] )
        coefs_.append(lr.coef_.ravel().copy())

    coefs_ = np.array(coefs_)

    out = pd.DataFrame(coefs_, columns = independent)
    out["c"] = c_steps

    n_sample = data.shape[0]
    out["lambda"] = [1./(c*n_sample) for c in c_steps]


    return out



