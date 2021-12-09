
import pandas as pd
import numpy as np
import sklearn as skl
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble



url = "train.csv"
url2 = "unique_m.csv"
superconduct = pd.read_csv(url)
unique_m = pd.read_csv(url2)
form = unique_m.material
unique_mat = np.unique(form)
n = len(unique_m)
train, test = skl.model_selection.train_test_split(unique_mat, 
                                                            test_size=0.1, 
                                                            random_state=1)


train, val = skl.model_selection.train_test_split(train, 
                                                            test_size=0.111, 
                                                            random_state=1)

superconduct = superconduct.merge(unique_m.material, left_index=True, right_index=True)
super_train = superconduct.loc[superconduct["material"].isin(train)]
super_test = superconduct.loc[superconduct["material"].isin(test)]
super_val = superconduct.loc[superconduct["material"].isin(val)]


fold_num = train.shape[0]//10
df_train = pd.DataFrame(train)


df_train['fold'] = (df_train.index ) // fold_num
df_train.fold[df_train.fold==10] = 9
df_train = df_train.rename(columns={0: 'material'})
super_train = super_train.merge(df_train, 'inner', on= 'material')
folds = []
n = super_train.shape[0]
rows = np.arange(n)
for fold in range(10):
    cross_train = np.asarray(super_train['fold'] != fold).nonzero()[0]
    cross_test = np.asarray(super_train['fold'] == fold).nonzero()[0]
    folds.append((cross_train, cross_test))





x_train = super_train.loc[:, 'number_of_elements':'wtd_std_Valence'].to_numpy()
y_train = super_train.loc[:, 'critical_temp'].to_numpy()
n, p = x_train.shape





x_val = super_val.loc[:, 'number_of_elements':'wtd_std_Valence'].to_numpy()
y_val = super_val.loc[:, 'critical_temp'].to_numpy()

x_test = super_test.loc[:, 'number_of_elements':'wtd_std_Valence'].to_numpy()
y_test = super_test.loc[:, 'critical_temp'].to_numpy()

for fold, (train, test) in enumerate(folds):
    # training data
    x_in = x_train[train, :]
    y_in = y_train[train]
    n, p = x_in.shape
    #test data
    x_test = x_train[test, :]
    y_test = y_train[test]
    rf = skl.ensemble.RandomForestRegressor(
                n_estimators=nt,  # number of trees
                criterion='mse',
                max_depth=md,      # maximum number of splits
                max_features='sqrt',
                max_samples=0.5,   # smaller yields more regularization
                n_jobs=3
            )
    res_rf = rf.fit(x_in, y_in)
            
    y_pred = res_rf.predict(x_test)
    mse = skl.metrics.mean_squared_error(y_test, y_pred)
    metrics['mse'].append(mse)
