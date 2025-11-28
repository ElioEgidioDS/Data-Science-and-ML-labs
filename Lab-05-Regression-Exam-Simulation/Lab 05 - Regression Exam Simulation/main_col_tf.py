import random
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import torch
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PolynomialFeatures,StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

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

    # Split the raw data before any preprocessing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )




    '''#-----PLOTTING histograms of some features to see if there is some relation
    plot_df = X_train.copy()
    plot_df['target'] = y_train

    # --- Plot an Ordinal Feature ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x='ord_0', y='target')
    plt.title('Target Distribution by ord_0 Categories')
    plt.show()

    # --- Plot a Categorical Feature ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x='cat_0', y='target')
    plt.title('Target Distribution by cat_0 Categories')
    plt.show()

    #-----END of PLOTTING'''




    # Get the lists of column names
    cont_cols = [x for x in X_train.columns if "cont" in x]
    #ord_cols = [x for x in X_train.columns if "ord" in x]
    cat_cols = [x for x in X_train.columns if "cat" in x]

    #---CHECK on skewness of trained data
    skewness = X_train[cont_cols].skew()

    print("Skewness of continuous features:")
    print(skewness) 
    #standard scaling is the correct choice 
    #no need to apply any log scaling, needed in case of extreme skewness
    #---END OF SKEW CHECK




    #-------------START OF PREPROCESSING STEPS----------------

    # --- CONTINUOUS Pipeline ---
    # 1. Fill missing values with the mean
    # 2. Scale the data
    cont_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # --- ORDINAL Pipeline ---
    # 1. Fill missing values with the most frequent value (mode)
    # 2. Apply OrdinalEncoder
    # We must get the categories *only from the training set*
    ord_cols = [x for x in X_train.columns if "ord" in x]
    ord_values = [ X_train[x].value_counts().sort_index().index.to_list() for x in ord_cols]

    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=ord_values))
    ])

    # --- CATEGORICAL Pipeline ---
    # 1. Fill missing values with the most frequent value (mode)
    # 2. Apply OneHotEncoder
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    

    # Create the master preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cont', cont_pipeline, cont_cols),
            ('ord', ord_pipeline, ord_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder='passthrough' # Keep any columns we didn't specify
    )


    # 1. Fit the preprocessor on the TRAINING data
    # This "learns" the means, modes, and categories from X_train ONLY
    print("Fitting preprocessor on X_train...")
    preprocessor.fit(X_train)

    # 2. Transform BOTH X_train and X_test
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing complete with no data leakage.")
    print(f"X_train_processed shape: {X_train_processed.shape}")
    print(f"X_test_processed shape: {X_test_processed.shape}")

    
    #---------END OF PREPROCESSING STEP------------------------

    '''
    FIRST MANUAL TRY WITH SIMPLE LINEAR REGRESSION(LOW MODEL SCORES):

    reg = LinearRegression()
    reg.fit(X_train_processed,y_train)
    y_test_pred = reg.predict(X_test_processed)

    r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(r2)
    print(rmse)
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.xlabel("Actual Values (y_test)")
    plt.ylabel("Predicted Values (y_test_pred)")
    plt.title("Actual vs. Predicted Values")

    # Add the 45-degree "perfect prediction" line
    p1 = max(max(y_test), max(y_test_pred))
    p2 = min(min(y_test), min(y_test_pred))
    plt.plot([p1, p2], [p1, p2], 'r--') # 'r--' is a red dashed line

    # Set the axis limits to be the same, making it a square plot
    plt.xlim(p2, p1)
    plt.ylim(p2, p1)

    plt.grid(True)
    plt.show()'''



    #------START OF MODEL EVALUATION----------------

    models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }

    # 3. Loop, train, predict, and store results
    results = []

    for name, model in models.items():
        
        # Train the model
        model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Store results
        results.append({'Model': name, 'R-squared': r2, 'RMSE': rmse})

    # 4. Create a DataFrame to view all results at once
    results_df = pd.DataFrame(results).sort_values(by='RMSE', ascending=True)

    print(results_df)


    #\\\\\\\\ POLYNOMIAL REGRESSION ////////////

    deg=2

    cont_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly', PolynomialFeatures(degree=deg,include_bias=False)),
        ('scaler',StandardScaler())
       ])
    
    preprocessor_poly = ColumnTransformer(
        transformers=[
            ('cont', cont_pipeline, cont_cols),
            ('ord',ord_pipeline,ord_cols),
            ('cat', cat_pipeline, cat_cols)
        ],
        remainder="passthrough"
    )

    poly_ridge = Pipeline([
        ('preprocessor',preprocessor_poly),
        ('regressor',Ridge())
    ])

    poly_ridge.fit(X_train,y_train)
    y_pred_poly = poly_ridge.predict(X_test)

    r2_ply = r2_score(y_test,y_pred_poly)
    rmse_ply = np.sqrt(mean_squared_error(y_test, y_pred_poly))

    results.append({
    'Model': f'Polynomial (Deg={deg}) + Ridge', 
    'R-squared': r2_ply, 
    'RMSE': rmse_ply
    })

    results_df = pd.DataFrame(results).sort_values(by='RMSE', ascending=True)

    print(results_df)

    #\\\\\\\\\\\  END OF POLYNOMIAL REGRESSION ////////////


    #-------END OF MODEL EVALUATION
    #RESULTS: random forest wins
    #         ridge of degree 2 comes second, gradboost comes close 



    '''#-------START OF HYPERPARAMETER TUNING(RANDOM FOREST vs RIDGE vs GRADBOOST)
    param_grid_ridge = {'regressor__alpha' : [0.1, 0.2, 0.3], 
                        'regressor__fit_intercept' : [True, False]} 

    gridsearch = GridSearchCV(poly_ridge, param_grid_ridge, scoring='r2', cv = 5,n_jobs=-1)
    gridsearch.fit(X_train,y_train)
    
    best_configured_model = gridsearch.best_estimator_
    y_pred_ridge = best_configured_model.predict(X_test)
    r2_ridge = r2_score(y_test,y_pred_ridge)
    print(f"Tuned polynomial ridge: {r2_ridge}")


    param_grid_ranfor = {
        'n_estimators' : [100, 200, 300],
        'max_depth' : [10, 20, None],
        'min_samples_split' : [2, 5],
        'min_samples_leaf' : [1, 2]
    }

    gridsearch = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_ranfor,  scoring='r2', cv = 5,n_jobs=-1)
    gridsearch.fit(X_train_processed,y_train)
    best_configured_model = gridsearch.best_estimator_
    y_pred_ridge = best_configured_model.predict(X_test_processed)
    r2_ridge = r2_score(y_test,y_pred_ridge)
    print(f"Tuned Random Forest: {r2_ridge}")


    param_grid_gradboost = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2]
    }

    gridsearch = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gradboost, cv=5, scoring='r2', verbose=1, n_jobs=-1)
    gridsearch.fit(X_train_processed,y_train)
    best_configured_model = gridsearch.best_estimator_
    y_pred_ridge = best_configured_model.predict(X_test_processed)
    r2_ridge = r2_score(y_test,y_pred_ridge)
    print(f"Tuned gradient boost: {r2_ridge}")

    
    #RESULTS:
    #Tuned polynomial ridge: 0.7229202287957441
    #Tuned Random Forest: 0.808956579530163
    #Tuned gradient boost: 0.7797565713015698
    

    '''

    #------FINAL RESULTS-----------
    final_pipline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor',  RandomForestRegressor(random_state=42,n_estimators=200,max_depth=None, min_samples_split=5,min_samples_leaf=1,n_jobs=-1))
    ])
    
    final_pipline.fit(X,y)
    test_data = pd.read_csv(r"test_dataset.csv",sep=',')
    y_final_pred = final_pipline.predict(test_data)

    submission = pd.DataFrame()

    if 'id' in test_data.columns:
        submission['index'] = test_data['index']
    else:
        submission['index'] = test_data.index

    submission["value"] = y_final_pred

    submission.to_csv(submission_path,index=False)
    print("Submission saved successfully.")
    
    

if __name__ == "__main__":
    main()
