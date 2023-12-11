import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from drfp import DrfpEncoder
from functools import partial
from typing import Iterable
from sklearn.utils import shuffle
from sklearn import preprocessing
import multiprocessing
from typing import Tuple
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from statistics import mean


data = pd.read_csv("amidation_carboxylicAcid_primaryAmine_ECFP_2048.csv")

data_1 = data[data['Yield']<100]

# Add the reagents to the reactants and rejoin with >> 
data_1['Reactants'] = data_1['SMILES'].str.split('>', expand=True)[0]
data_1['Agents'] = data_1['SMILES'].str.split('>', expand=True)[1]
data_1['Products'] = data_1['SMILES'].str.split('>', expand=True)[2]
data_1['Reactants+Agents'] = data_1['Reactants'] + "." + data_1["Agents"]
data_1['Reaction'] = data_1['Reactants+Agents'] + ">>" + data_1["Products"]

def encode(smiles: Iterable, length: int = 2048, radius: int = 3) -> np.ndarray: 
    return DrfpEncoder.encode(
        smiles,
        n_folded_length=length,
        radius=radius,
        rings=True,
    )


def encode_dataset(smiles: Iterable, length: int, radius: int) -> np.ndarray:
    """Encode the reaction SMILES to drfp"""

    cpu_count = (
        multiprocessing.cpu_count()
    )  # Data gets too big for piping when splitting less in python < 2.8

    # Split reaction SMILES for multiprocessing
    k, m = divmod(len(smiles), cpu_count)
    smiles_chunks = (
        smiles[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
        for i in range(cpu_count)
    )

    # Run the fingerprint generation in parallel
    results = []
    with multiprocessing.Pool(cpu_count) as p:
        results = p.map(partial(encode, length=length, radius=radius), smiles_chunks)

    return np.array([item for s in results for item in s])

drfps = encode_dataset(data_1['Reaction'], length=2048, radius=3)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

CV_num = 1
for train_index, test_index in kf.split(drfps):
    
    # Make splits for drfp and ecfp training sets
    X_train_drfp, X_test_drfp = drfps[train_index], drfps[test_index]
    X_train_ecfp, X_test_ecfp = data_1.iloc[:,:-7].to_numpy()[train_index], data_1.iloc[:,:-7].to_numpy()[test_index]
    y_train, y_test = data_1.loc[:,'Yield'].to_numpy()[train_index], data_1.loc[:,'Yield'].to_numpy()[test_index]
    
    # Group together the X_y and dump into pickle files
    with open("CV{}_2048_3_DRFP_train.pkl".format(CV_num), "wb+") as f:
        pickle.dump((X_train_drfp, y_train), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("CV{}_2048_3_ECFP_train.pkl".format(CV_num), "wb+") as f:
        pickle.dump((X_train_ecfp, y_train), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("CV{}_2048_3_DRFP_test.pkl".format(CV_num), "wb+") as f:
        pickle.dump((X_test_drfp, y_test), f, protocol=pickle.HIGHEST_PROTOCOL)
    with open("CV{}_2048_3_ECFP_test.pkl".format(CV_num), "wb+") as f:
        pickle.dump((X_test_ecfp, y_test), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    CV_num +=1

def load_data( 
    path_train: str,
    path_test: str,
    valid_frac: str = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    
    X_train, y_train= pickle.load(
        open(
            path_train,
            "rb",
        )
    )

    subset_indices = np.random.choice(
        np.arange(len(X_train)), int(1.0 * len(X_train)), replace=False
    )
    X_train = X_train[subset_indices]
    y_train = y_train[subset_indices]

    X_test, y_test = pickle.load(
        open(
            path_test,
            "rb",
        )
    )

    valid_indices = np.random.choice(
        np.arange(len(X_train)), int(valid_frac * len(X_train)), replace=False
    )
    X_valid = X_train[valid_indices]
    y_valid = y_train[valid_indices]

    train_indices = list(set(range(len(X_train))) - set(valid_indices))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Define % within certain range 
def within_range(list1, list2, range2):
    x=0
    for i in range(len(list2)):
        if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
            x+=1
    return((float(x)/(len(list2)))*100)

# Define RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Function to store model metrics in file 
def save_metrics(set_name: str,
    split_id: str,
    r_squared: float,
    within_10: float,
    within_5: float,
    rmse: float)->None:
    with open(f"{set_name}_{split_id}_metrics.csv", "w+") as f:
        f.write(f"r_squared = {r_squared}\n")
        f.write(f"rmse = {rmse}\n")
        f.write(f"% within 10% = {within_10}\n")
        f.write(f"% within 5% = {within_5}\n")


def save_results( 
    set_name: str,
    split_id: str,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> None:
    with open(f"{set_name}_{split_id}.csv", "w+") as f:
        for gt, pred in zip(ground_truth, prediction):
            f.write(f"{set_name},{split_id},{gt},{pred}\n")

def predict_uspto(train, test, file, name): 
    

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        train,
        test,
        valid_frac=0.1)

    model = XGBRegressor(
        n_estimators=999999,
        learning_rate=0.1,
        max_depth=12,
        min_child_weight=6,
        colsample_bytree=0.6,
        subsample=0.8,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=20,
        verbose=False,
    )

    y_pred = model.predict(X_test, ntree_limit=model.best_ntree_limit)
    y_pred[y_pred < 0.0] = 0.0

    save_results(name, str(file), y_test, y_pred)
    
    r_squared = r2_score(y_test, y_pred)
    RMSE = rmse(y_test, y_pred)
    within_10 = within_range(y_test, y_pred, 10)
    within_5 = within_range(y_test, y_pred, 5)
    
    #save_metrics("uspto", "AcidAnhydrideP", r_squared,
     #            within_10, within_5, rmse)
                 
    
    return r_squared, RMSE, within_10, within_5

# DRFP metrics
r2s = []
RMSEs = []
within_10s = []
within_5s = []

# ECFP metrics
r2s_ = []
RMSEs_ = []
within_10s_ = []
within_5s_ = []


for i in [1,2,3,4,5,6,7,8,9,10]:
    
    r_squared, RMSE, within_10, within_5 =  predict_uspto("CV{}_2048_3_DRFP_train.pkl".format(i), "CV{}_2048_3_DRFP_test.pkl".format(i), str(i), 'DRFPCV')
    r2s.append(r_squared)
    RMSEs.append(RMSE)
    within_10s.append(within_10)
    within_5s.append(within_5)

    r_squared_, RMSE_, within_10_, within_5_ =  predict_uspto("CV{}_2048_3_ECFP_train.pkl".format(i), "CV{}_2048_3_ECFP_test.pkl".format(i), str(i), 'ECFPCV')
    r2s_.append(r_squared_)
    RMSEs_.append(RMSE_)
    within_10s_.append(within_10_)
    within_5s_.append(within_5_)

save_metrics("uspto_CA_pA_", "DRFPCV", mean(r2s), mean(within_10s), mean(within_5s), mean(RMSEs))
save_metrics("uspto_CA_pA_", "ECFPCV", mean(r2s_), mean(within_10s_), mean(within_5s_), mean(RMSEs_))
