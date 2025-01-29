import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
from copy import deepcopy
from functools import partial
from itertools import combinations
import random
import gc


# Import sklearn classes for model selection, cross validation, and performance evaluation
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
from category_encoders import OneHotEncoder, OrdinalEncoder, CountEncoder, CatBoostEncoder
from imblearn.under_sampling import RandomUnderSampler

# Import libraries for Hypertuning
import optuna

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor, LassoCV
from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV, ElasticNetCV
from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier
from catboost import Pool

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

df_train = pd.read_csv('train.csv', index_col=[0])
df_test = pd.read_csv('test.csv', index_col=[0])
original = pd.read_csv('WildBlueberryPollinationSimulationData.csv', index_col=[0])

df_train['is_generated'] = 1
df_test['is_generated'] = 1
original['is_generated'] = 0

original = original.reset_index()
original['id'] = original['Row#'] + df_test.index[-1] + 1
original = original.drop(columns = ['Row#']).set_index('id')

target_col = 'yield'

print(f"df_train shape :{df_train.shape}")
print(f"df_test shape :{df_test.shape}")
print(f"original shape :{original.shape}")

original

def plot_histograms(df_train, df_test, original, target_col, n_cols=3):
    n_cols = 3
    n_rows = (len(df_train.columns) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))
    axes = axes.flatten()

    for i, var_name in enumerate(df_train.columns.tolist()):
        if var_name != 'is_generated':
            ax = axes[i]
            sns.distplot(df_train[var_name], kde=True, ax=ax, label='Train')
            if var_name != target_col:
                sns.distplot(df_test[var_name], kde=True, ax=ax, label='Test')
            sns.distplot(original[var_name], kde=True, ax=ax, label='Original')
            ax.set_title(f'{var_name} Distribution (Train vs Test)')
            ax.legend()

    plt.tight_layout()
    plt.show()
    
plot_histograms(df_train, df_test, original, target_col, n_cols=3)

def plot_heatmap(df, title):
    # Create a mask for the diagonal elements
    mask = np.zeros_like(df.astype(float).corr())
    mask[np.triu_indices_from(mask)] = True

    # Set the colormap and figure size
    colormap = plt.cm.RdBu_r
    plt.figure(figsize=(28, 28))

    # Set the title and font properties
    plt.title(f'{title} Correlation of Features', fontweight='bold', y=1.02, size=20)

    # Plot the heatmap with the masked diagonal elements
    sns.heatmap(df.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1.0, 
                square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 14, "weight": "bold"},
                mask=mask)

plot_heatmap(df_train.drop('is_generated', axis=1), title='Train')
plot_heatmap(df_test.drop('is_generated', axis=1), title='Test')
plot_heatmap(original.drop('is_generated', axis=1), title='original')

bins=[-np.inf, 40, 45, 50, 55, +np.inf]
labels=["39.0", "42.1", "46.8", "52.0", "57.2"]
bins_col = 'MinOfUpperTRange'

df_train[f'{bins_col}_bins'] = pd.cut(df_train[bins_col], bins=bins, labels=labels)
original[f'{bins_col}_bins'] = pd.cut(original[bins_col], bins=bins, labels=labels)

df_train

def plot_scatter_with_fixed_col(df, fixed_col, hue=False, drop_cols=[], size=30, title=''):
    sns.set_style('whitegrid')
    
    if hue:
        cols = df.columns.drop([hue, fixed_col] + drop_cols)
    else:
        cols = df.columns.drop([fixed_col] + drop_cols)
    n_cols = 4
    n_rows = (len(cols) - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(size, size/n_cols*n_rows), sharex=False, sharey=False)
    fig.suptitle(f'{title} Set Scatter Plot with Fixed Column', fontsize=24, fontweight='bold', y=1.01)

    for i, col in enumerate(cols):
        n_row = i // n_cols
        n_col = i % n_cols
        ax = axes[n_row, n_col]

        ax.set_xlabel(f'{col}', fontsize=14)
        ax.set_ylabel(f'{fixed_col}', fontsize=14)

        # Plot the scatterplot
        if hue:
            sns.scatterplot(data=df, x=col, y=fixed_col, hue=hue, ax=ax,
                            s=80, edgecolor='gray', alpha=0.35, palette='bright')
            ax.legend(title=hue, title_fontsize=12, fontsize=12) # loc='upper right'
        else:
            sns.scatterplot(data=df, x=col, y=fixed_col, ax=ax,
                            s=80, edgecolor='gray', alpha=0.35)

        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_title(f'{col}', fontsize=18)
    
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.show()
    
plot_scatter_with_fixed_col(df_train, fixed_col=target_col, hue=f'{bins_col}_bins', drop_cols=['is_generated'], size=24, title='Train')
plot_scatter_with_fixed_col(original, fixed_col=target_col, hue=f'{bins_col}_bins', drop_cols=['is_generated'], size=24, title='Original')

def plot_distribution(df, hue, title='', drop_cols=[]):
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
    n_cols = 2
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.histplot(data=df, x=var_name, kde=True, ax=ax, hue=hue) # sns.distplot(df_train[var_name], kde=True, ax=ax, label='Train')
        ax.set_title(f'{var_name} Distribution')

    fig.suptitle(f'{title} Distribution Plot', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_distribution(df_train[['fruitset', 'fruitmass', 'seeds', 'yield']+['MinOfUpperTRange_bins']], hue=f'{bins_col}_bins', title='Train Set', drop_cols=[])
plot_distribution(original[['fruitset', 'fruitmass', 'seeds', 'yield']+['MinOfUpperTRange_bins']], hue=f'{bins_col}_bins', title='Original Set', drop_cols=[])

def plot_boxplot(df, hue, title='', drop_cols=[]):
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
    n_cols = 2
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.boxplot(data=df, x=hue, y=var_name, ax=ax, showmeans=True, 
                    meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue", "markersize":"5"})
        ax.set_title(f'{var_name} by {hue}')

    fig.suptitle(f'{title} Boxplot', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()

plot_boxplot(df_train[['fruitset', 'fruitmass', 'seeds', 'yield']+['MinOfUpperTRange_bins']], hue=f'{bins_col}_bins', title='Train Set', drop_cols=[])
plot_boxplot(original[['fruitset', 'fruitmass', 'seeds', 'yield']+['MinOfUpperTRange_bins']], hue=f'{bins_col}_bins', title='Original Set', drop_cols=[])

def plot_violinplot(df, hue, title='', drop_cols=[]):
    sns.set_style('whitegrid')

    cols = df.columns.drop([hue] + drop_cols)
    n_cols = 2
    n_rows = (len(cols) - 1) // n_cols + 1

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 4*n_rows))

    for i, var_name in enumerate(cols):
        row = i // n_cols
        col = i % n_cols

        ax = axes[row, col]
        sns.violinplot(data=df, x=hue, y=var_name, ax=ax, inner='quartile')
        ax.set_title(f'{var_name} Distribution')

    fig.suptitle(f'{title} Violin Plot', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()


plot_violinplot(df_train[['fruitset', 'fruitmass', 'seeds', 'yield']+['MinOfUpperTRange_bins']], hue=f'{bins_col}_bins', title='Train Set', drop_cols=[])
plot_violinplot(original[['fruitset', 'fruitmass', 'seeds', 'yield']+['MinOfUpperTRange_bins']], hue=f'{bins_col}_bins', title='Original Set', drop_cols=[])

# Concatenate train and original dataframes, and prepare train and test sets
df_train = pd.concat([df_train, original])
X_train = df_train.drop([f'{target_col}'],axis=1).reset_index(drop=True)
y_train = df_train[f'{target_col}'].reset_index(drop=True)
X_test = df_test.reset_index(drop=True)

# Add features
X_train["fruitset_seed_mul"] = X_train["fruitset"] * X_train["seeds"]
X_test["fruitset_seed_mul"] = X_test["fruitset"] * X_test["seeds"]
X_train["fruitmass_seed_mul"] = X_train["fruitmass"] * X_train["seeds"]
X_test["fruitmass_seed_mul"] = X_test["fruitmass"] * X_test["seeds"]
X_train["fruitmass_fruitmass_mul"] = X_train["fruitset"] * X_train["fruitmass"]
X_test["fruitmass_fruitmass_mul"] = X_test["fruitset"] * X_test["fruitmass"]

X_train["fruitset_seed_div"] = X_train["fruitset"] / X_train["seeds"]
X_test["fruitset_seed_div"] = X_test["fruitset"] / X_test["seeds"]
X_train["fruitmass_seed_div"] = X_train["fruitmass"] / X_train["seeds"]
X_test["fruitmass_seed_div"] = X_test["fruitmass"] / X_test["seeds"]
X_train["fruitmass_fruitmass_div"] = X_train["fruitset"] / X_train["fruitmass"]
X_test["fruitmass_fruitmass_div"] = X_test["fruitset"] / X_test["fruitmass"]

X_train["RainingDays_mul"] = X_train["RainingDays"] * X_train["AverageRainingDays"]
X_test["RainingDays_mul"] = X_test["RainingDays"] * X_test["AverageRainingDays"]
X_train["RainingDays_div"] = X_train["RainingDays"] / X_train["AverageRainingDays"]
X_test["RainingDays_div"] = X_test["RainingDays"] / X_test["AverageRainingDays"]

X_train['SumOfTRange'] = [0 for _ in range(len(X_train['MinOfLowerTRange']))]
X_test['SumOfTRange'] = [0 for _ in range(len(X_test['MinOfLowerTRange']))]
for _ in ['MinOfLowerTRange', 'MinOfUpperTRange','AverageOfUpperTRange', 'AverageOfLowerTRange', 'MaxOfUpperTRange', 'MaxOfLowerTRange']:
    X_train['SumOfTRange'] += X_train[_]
    X_test['SumOfTRange'] += X_test[_]


# Drop_col
drop_cols = [
    # 'honeybee',
    'MinOfLowerTRange', 'MinOfUpperTRange','AverageOfUpperTRange', 'AverageOfLowerTRange',
    'MaxOfUpperTRange', 'MaxOfLowerTRange', 
    # 'RainingDays', 'AverageRainingDays',
    'is_generated'
]

X_train.drop(drop_cols+[f'{bins_col}_bins'], axis=1, inplace=True)
X_test.drop(drop_cols, axis=1, inplace=True)

# StandardScaler
categorical_columns = ['is_generated']
numeric_columns = [_ for _ in X_train.columns if _ not in categorical_columns]
sc = StandardScaler() # MinMaxScaler or StandardScaler
X_train[numeric_columns] = sc.fit_transform(X_train[numeric_columns])
X_test[numeric_columns] = sc.transform(X_test[numeric_columns])

print(f"X_train shape :{X_train.shape} , y_train shape :{y_train.shape}")
print(f"X_test shape :{X_test.shape}")

# Delete the train and test dataframes to free up memory
del df_train, df_test, original

X_train.head(5)

class Splitter:
    def __init__(self, fold_method=True, n_splits=5):
        self.n_splits = n_splits
        self.fold_method = fold_method # True = KFold, False=StratifiedKFold

    def split_data(self, X, y, random_state_list):
        for random_state in random_state_list:
            if fold_method:
                kf = KFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in kf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val
            else:
                skf = StratifiedKFold(n_splits=self.n_splits, random_state=random_state, shuffle=True)
                for train_index, val_index in skf.split(X, y):
                    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
                    yield X_train, X_val, y_train, y_val

n_estimators = 9999
early_stopping_rounds = 800
verbose = False

def lgbm_objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 2, 50),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'n_estimators': n_estimators,
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-10, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-10, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        'objective': 'regression_l1',
        'metric': 'mean_absolute_error',
        'boosting_type': 'gbdt',
        'random_state': 42
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train_opt, y_train_opt, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
    preds = model.predict(X_test_opt)
    return mean_absolute_error(y_test_opt, preds)

# Define objective functions for CatBoostRegressor
def catboost_objective(trial):
    params = {
        'iterations': n_estimators,
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'depth': trial.suggest_int('depth', 3, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-10, 10.0),
        'random_strength': trial.suggest_uniform('random_strength', 0.0, 100.0),
        'max_bin': trial.suggest_int('max_bin', 1, 500), 
        'od_wait': trial.suggest_int('od_wait', 1, 100), 
        'grow_policy': 'Lossguide',
        'bootstrap_type': 'Bayesian',
        'od_type': 'Iter',
        'eval_metric': 'MAE',
        'loss_function': 'MAE',
        'verbose': False
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train_opt, y_train_opt, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
    preds = model.predict(X_test_opt)
    return mean_absolute_error(y_test_opt, preds)

optuna.logging.set_verbosity(optuna.logging.WARNING)

random_state = 42

X_train_, X_test_opt, y_train_, y_test_opt = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train_, y_train_, test_size=0.2, random_state=random_state)

# Optimize LGBMRegressor    
study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(lgbm_objective, n_trials=100)

# Print best parameters and best score for LGBMRegressor
print("LGBMRegressor - Best trial:")
best_lgbm_params = study_lgbm.best_trial.params
print("Params: ", best_lgbm_params)
print(f'SEED-42 lgbm MAE score: ', study_lgbm.best_value)

# Optimize CatBoostRegressor
study_catboost = optuna.create_study(direction='minimize')
study_catboost.optimize(catboost_objective, n_trials=100)

# Print best parameters and best score for CatBoostRegressor
print("\nCatBoostRegressor - Best trial:")
best_catboost_params = study_catboost.best_trial.params
print("Params: ", best_catboost_params)
print(f'SEED-42 catbm MAE score: ', study_catboost.best_value)

random_state_list = [42]
fold_method = True
n_splits = 5

splitter = Splitter(fold_method=fold_method, n_splits=n_splits)

for m, random_state in enumerate(random_state_list):
    for n, (X_train_, X_test_opt, y_train_, y_test_opt) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
        X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train_, y_train_, test_size=0.2, random_state=random_state)
        # Optimize LGBMRegressor    
        study_lgbm = optuna.create_study(direction='minimize')
        study_lgbm.optimize(lgbm_objective, n_trials=100)

        # Print best parameters and best score for LGBMRegressor
        print("LGBMRegressor - Best trial:")
        best_lgbm_params = study_lgbm.best_trial.params
        print("Params: ", best_lgbm_params)
        print(f'[FOLD-{n} SEED-{random_state_list[m]}]lgbm MAE score: ', study_lgbm.best_value)
        
        # Optimize CatBoostRegressor
        study_catboost = optuna.create_study(direction='minimize')
        study_catboost.optimize(catboost_objective, n_trials=100)

        # Print best parameters and best score for CatBoostRegressor
        print("\nCatBoostRegressor - Best trial:")
        best_catboost_params = study_catboost.best_trial.params
        print("Params: ", best_catboost_params)
        print(f'[FOLD-{n} SEED-{random_state_list[m]}] MAE score: ', study_catboost.best_value)
        print('-'*150)

optuna.logging.set_verbosity(optuna.logging.WARNING)

class Regressor:
    def __init__(self, n_estimators=200, device="cpu", random_state=42):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.models = self._define_model()
        self.models_name = list(self._define_model().keys())
        self.len_models = len(self.models)
        
    def _define_model(self):
#         lgb_params = {
#             'n_estimators': self.n_estimators,
#             'random_state': self.random_state,
#             'num_leaves': 37, 
#             'learning_rate': 0.0019918819279222664, 
#             'max_depth': 9, 
#             'reg_alpha': 8.020069676334337e-09, 
#             'reg_lambda': 0.004722689662849701, 
#             'min_child_samples': 8, 
#             'subsample': 0.8378965417505742, 
#             'colsample_bytree': 0.6442716928004532,
#             'device': self.device,
#             'force_col_wise': True
#         }
#         lgb1_params = {
#             'n_estimators': self.n_estimators,
#             'random_state': self.random_state,
#             'num_leaves': 24, 
#             'learning_rate': 0.020821067551898657, 
#             'max_depth': 12, 
#             'reg_alpha': 8.035902004358891e-09, 
#             'reg_lambda': 4.8710004902224706e-08, 
#             'min_child_samples': 21, 
#             'subsample': 0.9233068723671619, 
#             'colsample_bytree': 0.8880898047713869,
#             'device': self.device,
#             'force_col_wise': True
#         }
#         lgb2_params = {
#             'n_estimators': self.n_estimators,
#             'random_state': self.random_state,
#             'num_leaves': 9, 
#             'learning_rate': 0.0044791409804770515, 
#             'max_depth': 15, 
#             'reg_alpha': 8.350931261229492, 
#             'reg_lambda': 5.055112229054667e-09, 
#             'min_child_samples': 39, 
#             'subsample': 0.7649386012510687, 
#             'colsample_bytree': 0.6841502514186323,
#             'device': self.device,
#             'force_col_wise': True
#         }
#         lgb3_params = {
#             'n_estimators': self.n_estimators,
#             'random_state': self.random_state,
#             'num_leaves': 18, 
#             'learning_rate': 0.005538226939909442, 
#             'max_depth': 8, 
#             'reg_alpha': 0.0008552332437461625, 
#             'reg_lambda': 0.025302451751690273, 
#             'min_child_samples': 82, 
#             'subsample': 0.7224177616950065, 
#             'colsample_bytree': 0.6120489223177372,
#             'device': self.device,
#             'force_col_wise': True
#         }
#         lgb4_params = {
#             'n_estimators': self.n_estimators,
#             'random_state': self.random_state,
#             'num_leaves': 43, 
#             'learning_rate': 0.030615936244234414, 
#             'max_depth': 11, 
#             'reg_alpha': 1.982566963868292e-05, 
#             'reg_lambda': 0.003442495293481783, 
#             'min_child_samples': 6, 
#             'subsample': 0.9153765281052757, 
#             'colsample_bytree': 0.9279264131470142,
#             'device': self.device,
#             'force_col_wise': True
#         }
#         lgb5_params = {
#             'n_estimators': self.n_estimators,
#             'random_state': self.random_state,
#             'num_leaves': 25, 
#             'learning_rate': 0.012732040200365323, 
#             'max_depth': 13, 
#             'reg_alpha': 0.013605855851193602, 
#             'reg_lambda': 7.033667306962803e-10, 
#             'min_child_samples': 22, 
#             'subsample': 0.6866989417861171, 
#             'colsample_bytree': 0.7417471356504809,
#             'device': self.device,
#             'force_col_wise': True
#         }
#         cb_params = {
#             'iterations': self.n_estimators,
#             'random_state': self.random_state,
#             'learning_rate': 0.004791176151246176, 
#             'depth': 9, 
#             'l2_leaf_reg': 1.3890582850862629e-08, 
#             'random_strength': 1.0529915605456992, 
#             'bagging_temperature': 1.5251974131917176, 
#             'max_bin': 430, 
#             'od_wait': 19,
#             'grow_policy': 'Lossguide',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'Iter',
#             'eval_metric': 'MAE',
#             'loss_function': 'MAE',
#             'verbose': False
#         }        
#         cb1_params = {
#             'iterations': self.n_estimators,
#             'random_state': self.random_state,
#             'learning_rate': 0.008392320543134638, 
#             'depth': 6, 
#             'l2_leaf_reg': 7.792031999964924, 
#             'random_strength': 13.03916111826091, 
#             'bagging_temperature': 2.62715902034164, 
#             'max_bin': 393, 
#             'od_wait': 52,
#             'grow_policy': 'Lossguide',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'Iter',
#             'eval_metric': 'MAE',
#             'loss_function': 'MAE',
#             'verbose': False
#         }
#         cb2_params = {
#             'iterations': self.n_estimators,
#             'random_state': self.random_state,
#             'learning_rate': 0.00591296765664144, 
#             'depth': 10, 
#             'l2_leaf_reg': 0.005730801589745849, 
#             'random_strength': 34.216054695132115, 
#             'bagging_temperature': 0.011828668259275893, 
#             'max_bin': 496, 
#             'od_wait': 75,
#             'grow_policy': 'Lossguide',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'Iter',
#             'eval_metric': 'MAE',
#             'loss_function': 'MAE',
#             'verbose': False

#         }
#         cb3_params = {
#             'iterations': self.n_estimators,
#             'random_state': self.random_state,
#             'learning_rate': 0.009949044944912614, 
#             'depth': 6, 
#             'l2_leaf_reg': 7.986322202890632, 
#             'random_strength': 31.673837540558328, 
#             'bagging_temperature': 0.5467701426141005, 
#             'max_bin': 407, 
#             'od_wait': 49,
#             'grow_policy': 'Lossguide',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'Iter',
#             'eval_metric': 'MAE',
#             'loss_function': 'MAE',
#             'verbose': False
#         }
        
#         cb4_params = {
#             'iterations': self.n_estimators,
#             'random_state': self.random_state,
#             'learning_rate': 0.003701035194043024, 
#             'depth': 8, 
#             'l2_leaf_reg': 0.005647910142796616, 
#             'random_strength': 1.2461850373887449, 
#             'bagging_temperature': 2.5317718637035647, 
#             'max_bin': 445, 
#             'od_wait': 44,
#             'grow_policy': 'Lossguide',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'Iter',
#             'eval_metric': 'MAE',
#             'loss_function': 'MAE',
#             'verbose': False
#         }
        
#         cb5_params = {
#             'iterations': self.n_estimators,
#             'random_state': self.random_state,
#             'learning_rate': 0.004833455156883491, 
#             'depth': 9, 
#             'l2_leaf_reg': 0.11144124929595065, 
#             'random_strength': 2.0608853758637635, 
#             'bagging_temperature': 2.1978583447053732, 
#             'max_bin': 435, 
#             'od_wait': 19,
#             'grow_policy': 'Lossguide',
#             'bootstrap_type': 'Bayesian',
#             'od_type': 'Iter',
#             'eval_metric': 'MAE',
#             'loss_function': 'MAE',
#             'verbose': False
#         }

        lgb1_params = {
            'n_estimators': self.n_estimators,
            'num_leaves': 16,
            'learning_rate': 0.05,
            'subsample': 0.60,
            'colsample_bytree': 1,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-07,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb2_params = {
            'n_estimators': self.n_estimators,
            'num_leaves': 93, 
            'min_child_samples': 20, 
            'learning_rate': 0.05533790147941807, 
            'colsample_bytree': 0.8809128870084636, 
            'reg_alpha': 0.0009765625, 
            'reg_lambda': 0.015589408048174165,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state
        }
        lgb3_params = {
            'n_estimators': self.n_estimators,
            'num_leaves': 45,
            'max_depth': 13,
            'learning_rate': 0.0684383311038932,
            'subsample': 0.5758412171285148,
            'colsample_bytree': 0.8599714680300794,
            'reg_lambda': 1.597717830931487e-08,
            'objective': 'regression_l1',
            'metric': 'mean_absolute_error',
            'boosting_type': 'gbdt',
            'device': self.device,
            'random_state': self.random_state,
            'force_col_wise': True
        }
                
        cb1_params = {
            'iterations': self.n_estimators,
            'depth': 8,
            'learning_rate': 0.01,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 70,
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb2_params = {
            'iterations': self.n_estimators,
            'depth': 9, 
            'learning_rate': 0.456,
            'l2_leaf_reg': 8.41,
            'random_strength': 0.18,
            'max_bin': 225, 
            'od_wait': 58, 
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state
        }
        cb3_params = {
            'n_estimators': self.n_estimators,
            'depth': 10,
            'learning_rate': 0.08827842054729117,
            'l2_leaf_reg': 4.8351074756668864e-05,
            'random_strength': 0.21306687539993183,
            'max_bin': 483,
            'od_wait': 97,
            'grow_policy': 'Lossguide',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MAE',
            'loss_function': 'MAE',
            'task_type': self.device.upper(),
            'random_state': self.random_state,
            'silent': True
        }

        models = {
            "LGBM1": lgb.LGBMRegressor(**lgb1_params),
            "LGBM2": lgb.LGBMRegressor(**lgb2_params),
            # "lgb3": lgb.LGBMRegressor(**lgb3_params),
            "CatBoost1": CatBoostRegressor(**cb1_params),
            # "CatBM2": CatBoostRegressor(**cb2_params),
            "CatBoost2": CatBoostRegressor(**cb3_params),
            # "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=self.n_estimators, learning_rate=0.01, loss="least_absolute_deviation", n_iter_no_change=300,random_state=self.random_state),
        }
        
        return models

class OptunaWeights:
    def __init__(self, random_state, n_trials=4000):
        self.study = None
        self.weights = None
        self.random_state = random_state
        self.n_trials = n_trials

    def _objective(self, trial, y_true, y_preds):
        # Define the weights for the predictions from each model
        weights = [trial.suggest_float(f"weight{n}", 1e-14, 1) for n in range(len(y_preds))]

        # Calculate the weighted prediction
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=weights)

        # Calculate the score for the weighted prediction
        score = mean_absolute_error(y_true, weighted_pred)
        return score

    def fit(self, y_true, y_preds):
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        sampler = optuna.samplers.CmaEsSampler(seed=self.random_state)
        pruner = optuna.pruners.HyperbandPruner()
        self.study = optuna.create_study(sampler=sampler, pruner=pruner, study_name="OptunaWeights", direction='minimize')
        objective_partial = partial(self._objective, y_true=y_true, y_preds=y_preds)
        self.study.optimize(objective_partial, n_trials=self.n_trials)
        self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))]

    def predict(self, y_preds):
        assert self.weights is not None, 'OptunaWeights error, must be fitted before predict'
        weighted_pred = np.average(np.array(y_preds).T, axis=1, weights=self.weights)
        return weighted_pred

    def fit_predict(self, y_true, y_preds):
        self.fit(y_true, y_preds)
        return self.predict(y_preds)
    
    def weights(self):
        return self.weights

%%time

fold_method = True
n_splits = 5
random_state = 42
random_state_list = [42]
n_estimators = 9999
early_stopping_rounds = 800
verbose = False
device = 'cpu'

splitter = Splitter(fold_method=fold_method, n_splits=n_splits)

# Initialize an array for storing test predictions
regressor = Regressor(n_estimators, device, random_state)
test_predss = np.zeros((X_test.shape[0]))
oof_predss = np.zeros((X_train.shape[0]))
ensemble_score = []
weights = []
trained_models = {'LGBM':[], 'Cat':[], 'RandomForestRegressor':[]}
score_dict = dict(zip(regressor.models_name, [[] for _ in range(regressor.len_models)]))
optweightss = []
    
for i, (X_train_, X_val, y_train_, y_val) in enumerate(splitter.split_data(X_train, y_train, random_state_list=random_state_list)):
    n = i % n_splits
    m = i // n_splits
            
    # Get a set of Regressor models
    regressor = Regressor(n_estimators, device, random_state)
    models = regressor.models
    
    # Initialize lists to store oof and test predictions for each base model
    oof_preds = []
    test_preds = []
    
    # Loop over each base model and fit it to the training data, evaluate on validation data, and store predictions
    for name, model in models.items():
        # model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        
        if ('xgb' in name) or ('LGBM' in name) or ('Cat' in name):
            model.fit(X_train_, y_train_, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=verbose)
        else:
            model.fit(X_train_, y_train_)
            
        if name in trained_models.keys():
            pass
            # trained_models[f'{name}'].append(deepcopy(model))
        
        test_pred = model.predict(X_test).reshape(-1)
        y_val_pred = model.predict(X_val).reshape(-1)
        
        score = mean_absolute_error(y_val, y_val_pred)
        score_dict[name].append(score)
        print(f'{name} [FOLD-{n} SEED-{random_state_list[m]}] MAE score: {score:.5f}')
        
        oof_preds.append(y_val_pred)
        test_preds.append(test_pred)
    
    # Use Optuna to find the best ensemble weights
    optweights = OptunaWeights(random_state=random_state)
    y_val_pred = optweights.fit_predict(y_val.values, oof_preds)
    
    score = mean_absolute_error(y_val, y_val_pred)
    print(f'Ensemble [FOLD-{n} SEED-{random_state_list[m]}] MAE score {score:.5f}')
    ensemble_score.append(score)
    weights.append(optweights.weights)
    
    # Predict to X_test by the best ensemble weights
    test_predss += optweights.predict(test_preds) / (n_splits * len(random_state_list))
    oof_predss[X_val.index] = optweights.predict(oof_preds)
    
    optweightss.append(optweights)
    
    gc.collect()

def plot_score_from_dict(score_dict, title='Mean Absolute Error', ascending=True):
    score_df = pd.melt(pd.DataFrame(score_dict))
    score_df = score_df.sort_values('value', ascending=ascending)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='value', y='variable', data=score_df, palette='Blues_r')
    plt.xlabel(f'{title}', fontsize=14)
    plt.ylabel('')
    #plt.title(f'{title}', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim([320,340])
    plt.grid(True, axis='x')
    plt.show()
    
plot_score_from_dict(score_dict, title=f'Mean Absolute Error (n_splits:{n_splits})')

score_list = [v for k,v in score_dict.items()]

print('--- Model MAE ---')
mean_mae = np.mean(score_list, axis=0)
std_mae = np.std(score_list, axis=0)
for name, mean_mae, std_mae in zip(score_dict.keys(), mean_mae, std_mae):
    print(f'{name}: {mean_mae:.5f} ± {std_mae:.5f}')

# Calculate the mean LogLoss score of the ensemble
mean_score = np.mean(ensemble_score)
std_score = np.std(ensemble_score)
print(f'Ensemble MAE score {mean_score:.5f} ± {std_score:.5f}')

print('')
# Print the mean and standard deviation of the ensemble weights for each model
print('--- Model Weights ---')
mean_weights = np.mean(weights, axis=0)
std_weights = np.std(weights, axis=0)
for name, mean_weight, std_weight in zip(models.keys(), mean_weights, std_weights):
    print(f'{name}: {mean_weight:.5f} ± {std_weight:.5f}')
    
# weight_dict = dict(zip(list(score_dict.keys()), np.array(weights).T.tolist()))
# plot_score_from_dict(weight_dict, title='Model Weights', ascending=False)
normalize = [((weight - np.min(weight)) / (np.max(weight) - np.min(weight))).tolist() for weight in weights]
weight_dict = dict(zip(list(score_dict.keys()), np.array(normalize).T.tolist()))
plot_score_from_dict(weight_dict, title='Model Weights (Normalize 0 to 1)', ascending=False)

for n in range(n_splits):
    for name, weight in weight_dict.items():
        print(f'{name} [FOLD-{n} SEED-42] Model Weights: {weight[n]:.5f}')

plt.figure(figsize=(20, 6))
plt.subplot(1, 2, 1)
sns.histplot(oof_predss, kde=True, alpha=0.5, label='oof_preds')
sns.histplot(y_train.values, kde=True, alpha=0.5, label='y_train')
plt.title('Histogram of OOF Predictions and Train Values')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=y_train.values, y=oof_predss, alpha=0.5)
plt.xlabel('Actual Values')
plt.ylabel('OOF Predicted Values')
plt.title('Actual vs. OOF Predicted Values')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', alpha=0.5)
plt.show()

print('Ensemble MAE score : ',mean_absolute_error(y_train.values, oof_predss))

unique_targets = np.unique(y_train)
def mattop_post_process(preds):
     return np.array([min(unique_targets, key = lambda x: abs(x - pred)) for pred in preds])

sub = pd.read_csv('sample_submission.csv')
sub[f'{target_col}'] = mattop_post_process(test_predss)
sub.to_csv('ensemble_submission.csv', index=False)
sub
