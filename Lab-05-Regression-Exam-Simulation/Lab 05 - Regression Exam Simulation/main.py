import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,StandardScaler

SEED = 42
N = 100  

def main():
    random.seed(SEED)

    np.random.seed(SEED)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    indices = list(range(N))
    values = np.random.rand(N).tolist() 
    noise = torch.randn(N).numpy()   

    combined_values = [v + 0.1 * n for v, n in zip(values, noise)]

    df = pd.DataFrame({
        "index": indices,
        "value": combined_values
    })

    submission_path = "submission.csv"
    df.to_csv(submission_path, index=False)

    data = pd.read_csv("train_dataset.csv",sep=',')
    print(data.shape)

    miss_cols = data.isna().sum()/data.shape[0]
    miss_cols = miss_cols.sort_values(ascending=False).head()
    print(miss_cols) #we can probably eliminate these but for now lets keep them

    print(data["target"])

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    #-----ORDINAL PREPROCESSING---------------------- use ordinalencoder and impute mode in nan values

    print(X["ord_0"].value_counts().sort_index())

    ord_values = [ X[x].value_counts().sort_index().index.to_list() for x in X.columns if "ord" in x]
    print(ord_values)

    oe = OrdinalEncoder(categories=ord_values)
    X_ord = X[[x  for x in X.columns if "ord" in x]]
    '''
    2. What X_ord.mode().iloc[0] Does

        When you use .iloc[0], you are selecting the first row (at index 0) from that DataFrame. This gives you a Series:

        ord_0    'val_A'
        ord_1    'val_X'
        ord_2    'val_M'
        Name: 0, dtype: object

        This Series is now a perfect "map" where the index is the column name and the value is the mode to fill in.

        3. What fillna() Does

        When you pass this Series to fillna(): X_ord.fillna(X_ord.mode().iloc[0])

        Pandas interprets this as:

            "For the ord_0 column, fill all NaNs with 'val_A'."

            "For the ord_1 column, fill all NaNs with 'val_X'."

            "For the ord_2 column, fill all NaNs with 'val_M'."
    '''
    print(X_ord.mode().T)
    X_ord = X_ord.fillna(X_ord.mode().iloc[0])
    #--------------------------------------------------------------
    X[[x  for x in X.columns if "ord" in x]] = oe.fit_transform(X_ord)
    #---------------------------------------------------------------


    #-------------------------CATEGORICAL PREPROCESSING--------------------------------------------
    cat_values = [X[x].value_counts().sort_index().index.to_list() for x in X.columns if "cat" in  x]
    print(cat_values)
    cat_cols = [x for x in X.columns if "cat" in  x]
    X_cat = X[cat_cols]
    X_cat = X_cat.fillna(X_cat.mode().iloc[0])
    
    ohe = OneHotEncoder(sparse_output=False, dtype=int)

    X_ohe_array = ohe.fit_transform(X_cat)

    '''
    You converted a few categorical columns (e.g., 8 columns) into many binary columns (e.g., 50 columns) using One-Hot Encoding.
    You cannot assign 50 columns back into 8 columns using standard DataFrame assignment (X[cat_cols] = ...).
    '''
    ohe_feature_names = ohe.get_feature_names_out(cat_cols)
    X_ohe_df = pd.DataFrame(X_ohe_array, columns=ohe_feature_names, index=X.index)

    # drop the old (text) columns from X
    X = X.drop(columns=cat_cols)

    # concatenate the new (numerical) columns to X
    X = pd.concat([X, X_ohe_df], axis=1)

    #-------------------------------------------------------------------

    #--------------CONTINOUS DATA-----------------------------------------
    std = StandardScaler()

    cont_cols = [x for x in X.columns if "cont" in x]
    X_cont = X[cont_cols]
    X_cont = X_cont.fillna(X_cont.mean(axis=0))

    X_cont_arr = std.fit_transform(X_cont)

    X_cont_scaled_df = pd.DataFrame(
    X_cont_arr, 
    columns=cont_cols,  # Use the original names
    index=X.index       # Ensure the index aligns with the main X DataFrame
    )

    X = X.drop(columns=cont_cols)
    X = pd.concat([X,X_cont_scaled_df],axis=1)





    

if __name__ == "__main__":
    main()
