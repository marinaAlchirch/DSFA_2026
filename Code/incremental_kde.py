import copy
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Code.california_housing_dataset_cleaning import clean_data
from Code.kernels import CustomGaussianKernel, CustomEpanechnikovKernel
from src.trees import HoeffdingTreeRegressor, iSOUPTreeRegressor, HoeffdingAdaptiveTreeRegressor
from Code.model import IncrKDEModel



# Models to get tuned for different parameters used in kde
def create_models(lambda_list, h_list, window_size_list, y):
    kde_models = []

    lambda_list = lambda_list if lambda_list else [0]
    h_list = h_list if h_list else [1]
    window_size_list = window_size_list if window_size_list else [1]
    kernels=[CustomGaussianKernel(sigma=1), CustomEpanechnikovKernel()]
    #kernels = [CustomGaussianKernel(sigma=1)]
    #range_of_bin_list = [0, 1, 2, 5, 10]
    #range_of_bin_list = [0, 1, 2, 10]
    range_of_bin_list = [0, 1, 2]

    parameters = {'lambda_selected': None, 'h': None, 'window_size': None, 'kernel': None,
                  'kde_type': None, 'range_of_bin': None}
    fall_back_kde_model = IncrKDEModel(y=y, model=HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42),
                             parameters=parameters, fall_back=True)
    kde_models.append(fall_back_kde_model)

    for lambda_selected in lambda_list:
        for h in h_list:
            for window_size in window_size_list:
                for kernel in kernels:
                    for kde_type in ['labeled', 'binned']:
                        for range_of_bin in range_of_bin_list:
                            if kde_type == 'labeled':
                                range_of_bin = 0
                            parameters = {'lambda_selected':lambda_selected, 'h':h, 'window_size':window_size, 'kernel':kernel, 'kde_type':kde_type, 'range_of_bin':range_of_bin}
                            kde_model = IncrKDEModel(y=y, model=HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42), parameters=parameters, fall_back=False)
                            kde_models.append(kde_model)


    return kde_models

def tune_and_train_on_hierarchical_shrinkage(lambda_list, X, y, tune_metric_selection='mse', tuning_times=4):



    ht = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
    ht_predictions = []
    ht_hs_predictions = []

    training_labels = []
    #counter_examples = 0

    best_lambda = None

    tuning_window_size = min(int(len(X) / (2 * tuning_times)), 3000)  # for example, use 50% of each tuning segment
    print(f"tuning_window_size: {tuning_window_size}")
    tuning_start_indices = [int(i * len(X) / tuning_times) for i in range(tuning_times)]


    if isinstance(X, pd.Series):
        X = X.values
        X = np.array(X)
    if isinstance(y, pd.Series):
        y = y.values
        y = np.array(y)

    i = 0
    while i < len(X):
        # Perform tuning in specific portions of the data stream
        if i in tuning_start_indices:
            round_num = tuning_start_indices.index(i) + 1
            tuning_start_time = time.time()
            print(f'tuning_start_indices: {tuning_start_indices}')
            print(f"Tuning [{round_num}/{tuning_times}] at example {i}...")

            if tune_metric_selection == 'mse':
                print(f"Tuning on {tune_metric_selection}...")
                prev_mse = np.inf
                print(f" models to tune : {len(lambda_list)}")
                for lambda_sel in lambda_list:
                    j = 0
                    hs_preds = []
                    true_labels = []
                    while j < tuning_window_size:
                        if str(type(X)) == '<class \'pandas.core.frame.DataFrame\'>':
                            x_instance =  X.iloc[j:j+1].values
                        elif str(type(X)) == '<class \'numpy.ndarray\'>':
                            x_instance = X[j].reshape(1, -1)

                        else:
                            raise TypeError('X type must either be pandas.core.frame.DataFrame or numpy.ndarray')

                        if isinstance(x_instance, pd.Series) or isinstance(x_instance, pd.DataFrame):
                            x_instance = x_instance.values
                        if isinstance(x_instance, np.ndarray):
                            if x_instance.ndim == 1:
                                x_instance = x_instance.reshape(1, -1)

                        true_labels.append(y[j])
                        hs_ht_pred = ht.predictHSNew(X=x_instance, lambda_selected=lambda_sel)[0]
                        hs_preds.append(hs_ht_pred)
                        j+=1

                    # print(f'size of hs model predictions : {len(hs_preds)}')
                    # print(f'size of y_true to compare with : {len(true_labels)}')
                    model_mse = mean_squared_error(y_true=true_labels, y_pred=hs_preds)
                    print(f"lambda: {lambda_sel}, mse : {model_mse}")
                    if prev_mse >= model_mse:
                        prev_mse = model_mse
                        best_lambda = lambda_sel

                print(f'best model mse : {prev_mse}')
                tuning_duration = time.time() - tuning_start_time
                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                print(f"Best lambda after tuning: {best_lambda}\n")

            elif tune_metric_selection == 'mae':
                print(f"Tuning on {tune_metric_selection}...")
                prev_mae = np.inf
                print(f" models to tune : {len(len(lambda_list))}")
                for lambda_sel in lambda_list:
                    j = 0
                    hs_preds = []
                    true_labels = []
                    while j < tuning_window_size:
                        x_instance = X[j]
                        true_labels.append(y[j])
                        if isinstance(x_instance, pd.Series) or isinstance(x_instance, pd.DataFrame):
                            x_instance = x_instance.values
                        if isinstance(x_instance, np.ndarray):
                            if x_instance.ndim == 1:
                                x_instance = x_instance.reshape(1, -1)

                        hs_ht_pred = ht.predictHSNew(X=x_instance, lambda_selected=lambda_sel)[0]
                        hs_preds.append(hs_ht_pred)
                        j+=1
                    model_mae = mean_absolute_error(y_true=true_labels, y_pred=hs_preds)
                    if prev_mae >= model_mae:
                        prev_mae = model_mae
                        best_lambda = lambda_sel

                tuning_duration = time.time() - tuning_start_time
                print(f'best model mae : {prev_mae}')

                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                print(f"Best lambda after tuning: {best_lambda}\n")

            elif tune_metric_selection == 'r2':
                print(f"Tuning on {tune_metric_selection}...")
                prev_r2 = -1
                print(f" models to tune : {len(len(lambda_list))}")
                for lambda_sel in lambda_list:
                    j = 0
                    hs_preds = []
                    true_labels = []
                    while j < tuning_window_size:
                        x_instance = X[j]
                        true_labels.append(y[j])
                        if isinstance(x_instance, pd.Series) or isinstance(x_instance, pd.DataFrame):
                            x_instance = x_instance.values
                        if isinstance(x_instance, np.ndarray):
                            if x_instance.ndim == 1:
                                x_instance = x_instance.reshape(1, -1)

                        hs_ht_pred = ht.predictHSNew(X=x_instance, lambda_selected=lambda_sel)[0]
                        hs_preds.append(hs_ht_pred)
                        j+=1
                    model_r2 = r2_score(y_true=true_labels, y_pred=hs_preds)
                    if prev_r2 <= model_r2:
                        prev_r2 = model_r2
                        best_lambda = lambda_sel

                tuning_duration = time.time() - tuning_start_time
                print(f'best model r2 : {prev_r2}')

                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                print(f"Best lambda after tuning: {best_lambda}\n")

            else:
                raise TypeError('Tuning is only available for mse, mae and r2 scores.')

            print(f'before i+=tuning_window_size, i is : {i}')
            i += tuning_window_size
            print(f'after i+=tuning_window_size, i is : {i}')


        else:
            #print(f"best_parameters before training: {kde_model.parameters}")
            #print(f" i is : {i}")
            if str(type(X)) == '<class \'pandas.core.frame.DataFrame\'>':
                instance_features = X.iloc[i:i + 1].values
            elif str(type(X)) == '<class \'numpy.ndarray\'>':
                instance_features = X[i].reshape(1, -1)
            else:
                raise TypeError('X type must either be pandas.core.frame.DataFrame or numpy.ndarray')

            actual_value = y[i]
            training_labels.append(actual_value)

            if not best_lambda:
                best_parameters = 0

            # Predict with HoeffdingTree
            ht_pred = ht.predict(instance_features)[0]
            ht_predictions.append(ht_pred)

            # Predict with HoeffdingTree and Hierarchical Shrinkage
            ht_hs_pred = ht.predictHSNew(instance_features, lambda_selected=best_lambda)[0]
            ht_hs_predictions.append(ht_hs_pred)


            ht.partial_fit(instance_features, [actual_value])


            i += 1

    # Return predictions
    print(f"Size of ht predictions: {len(ht_predictions)}")
    print(f"Size of actual labels: {len(training_labels)}")

    return ht_predictions, ht_hs_predictions, training_labels

def tune_and_train(parameters, X, y, tune_metric_selection='mse', tuning_times=4):


    # Initialize a KDE model with some dummy/default parameters before using the best_parameters returned from the tuning process
    dummy_parameters = {
        'lambda_selected': 0.0,
        'h': 1,
        'window_size': 1,
        'kernel': CustomGaussianKernel(sigma=1.0),
        'kde_type': 'labeled',
        'range_of_bin': 0
    }

    kde_model = IncrKDEModel(y=y, model=HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42), parameters=dummy_parameters)
    #kde_model = IncrKDEModel(y=y, model=iSOUPTreeRegressor(), parameters=dummy_parameters)

    ht = HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42)
    ht_predictions = []
    ht_hs_predictions = []

    lambda_list = parameters['lambda_list']
    h_list = parameters['h_list']
    window_size_list = parameters['window_size_list']

    # Initialize Incremental KDE Models
    kde_models = create_models(lambda_list=lambda_list, h_list=h_list, window_size_list=window_size_list, y=y)

    count_window = 0
    train_x = []
    train_y = []

    training_labels = []
    #counter_examples = 0

    best_parameters = {}

    tuning_window_size = min(int(len(X) / (2 * tuning_times)), 3000)  # for example, use 50% of each tuning segment
    print(f"tuning_window_size: {tuning_window_size}")
    tuning_start_indices = [int(i * len(X) / tuning_times) for i in range(tuning_times)]
    tuning_start_time = None
    current_start = None
    round_num = None
    if isinstance(X, pd.Series):
        X = X.values
        X = np.array(X)
    if isinstance(y, pd.Series):
        y = y.values
        y = np.array(y)

    i = 0
    while i < len(X):
        # Perform tuning in specific portions of the data stream
        if i in tuning_start_indices:
            round_num = tuning_start_indices.index(i) + 1
            tuning_start_time = time.time()
            print(f'tuning_start_indices: {tuning_start_indices}')
            print(f"Tuning [{round_num}/{tuning_times}] at example {i}...")

            if tune_metric_selection == 'mse':
                print(f"Tuning on {tune_metric_selection}...")
                prev_mse = np.inf
                print(f" models to tune : {len(kde_models)}")
                counter = 1
                start_idx = 0
                end_idx = 0
                fall_back_mse = np.inf
                for model in kde_models:
                    if counter%50==0:
                        print(f"tuning on model : {counter}")

                    if not model.fall_back:

                        model_ws = model.parameters['window_size']
                        assert model_ws <= tuning_window_size, f"Model window size {model_ws} exceeds tuning window size {tuning_window_size}"
                        model.predictions = []  # Clear old predictions
                        #original_start_index = copy.deepcopy(i)
                        j = 0
                        index = copy.deepcopy(i)
                        flag = False
                        #end_idx = copy.deepcopy(model_ws)
                        end_idx = copy.deepcopy(index)
                        while j < tuning_window_size:
                            start_idx = copy.deepcopy(index)
                            if j + model_ws >= tuning_window_size:
                                #print("if j + model_ws >= tuning_window_size:")
                                end_idx = copy.deepcopy(i + tuning_window_size)
                                index = index + (tuning_window_size - j)
                                flag = True
                            else:
                                j += copy.deepcopy(model_ws)
                                index += copy.deepcopy(model_ws)

                            if not flag:
                                end_idx += model_ws

                            if isinstance(X, pd.DataFrame):
                                X_tune_batch = X.iloc[start_idx:end_idx].values
                            else:  # numpy array
                                X_tune_batch = X[start_idx:end_idx].reshape(-1, X.shape[1])

                            #print(f'X_tune_batch size : {X_tune_batch.shape}')

                            model.predict_then_fit(X_tune_batch, y[start_idx:end_idx])
                            #print(f' j is : {j}')
                            #print(f'start_idx: {start_idx}, end_idx: {end_idx}')
                            #print(f'model.predictions size : {len(model.predictions)}')
                            if flag:
                                counter += 1
                                #print(f'counter: {counter}')
                                j += model_ws
                                break

                    if model.fall_back:
                        model.predictions = []
                        index = copy.deepcopy(i)
                        start_idx = copy.deepcopy(index)
                        end_idx = min(index + tuning_window_size, len(X))
                        if isinstance(X, pd.DataFrame):
                            X_tune_batch = X.iloc[start_idx:end_idx].values
                        else:  # numpy array
                            X_tune_batch = X[start_idx:end_idx].reshape(-1, X.shape[1])

                        model.predict_then_fit(X_tune_batch, y[start_idx:end_idx])
                        model_mse = mean_squared_error(y_true=y[index:end_idx], y_pred=model.predictions)
                        print(f'len(y_true) : {len(y[index:end_idx])}')
                        print(f'len(model.predictions) : {len(model.predictions)}')
                        print(f'fallback model mse : {model_mse}')
                        fall_back_mse = copy.deepcopy(model_mse)
                        counter += 1

                    # print(f'start_idx: {start_idx}, end_idx: {end_idx}')
                    # print(f'i is : {i}')
                    # print(f'size of model predictions : {len(model.predictions)}')
                    # print(f'size of y_true to compare with : {len(y[i:end_idx])}')
                    model_mse = mean_squared_error(y_true=y[i:end_idx], y_pred=model.predictions)
                    if prev_mse >= model_mse:
                        prev_mse = model_mse
                        best_parameters = copy.deepcopy(model.parameters)

                #counter_examples += tuning_window_size
                print(f'best model mse : {prev_mse}')
                tuning_duration = time.time() - tuning_start_time
                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                if fall_back_mse != prev_mse:
                    kde_model.set_parameters(best_parameters)
                print(f"Best parameters after tuning: {best_parameters}\n")

            elif tune_metric_selection == 'mae':
                print(f"Tuning on {tune_metric_selection}...")
                prev_mae = np.inf
                for model in kde_models:
                    model_ws = model.parameters['window_size']
                    assert model_ws <= tuning_window_size, f"Model window size {model_ws} exceeds tuning window size {tuning_window_size}"
                    #start_idx = counter_examples
                    #end_idx = min(counter_examples + model_ws, len(X))
                    model.predictions = []  # Clear old predictions

                    original_start_index = copy.deepcopy(i)

                    for j in range(tuning_window_size):
                        if j + model_ws >= tuning_window_size:
                            j = tuning_window_size - j
                        else:
                            j += model_ws
                        start_idx = i
                        end_idx = min(i + j, len(X))
                        i += model_ws


                        if isinstance(X, pd.DataFrame):
                            X_tune_batch = X.iloc[start_idx:end_idx].values
                        else:  # numpy array
                            X_tune_batch = X[start_idx:end_idx].reshape(-1, X.shape[1])
                        model.predict_then_fit(X_tune_batch, y[start_idx:end_idx])
                    model_mae = mean_absolute_error(y_true=y[original_start_index:end_idx], y_pred=model.predictions)
                    if prev_mae >= model_mae:
                        prev_mae = model_mae
                        best_parameters = copy.deepcopy(model.parameters)

                #counter_examples += tuning_window_size

                tuning_duration = time.time() - tuning_start_time
                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                kde_model.set_parameters(best_parameters)
                print(f"Best parameters after tuning: {best_parameters}\n")

            elif tune_metric_selection == 'r2':
                print(f"Tuning on {tune_metric_selection}...")
                prev_r2 = -1
                for model in kde_models:
                    model_ws = model.parameters['window_size']
                    assert model_ws <= tuning_window_size, f"Model window size {model_ws} exceeds tuning window size {tuning_window_size}"
                    #start_idx = counter_examples
                    #end_idx = min(counter_examples + model_ws, len(X))
                    model.predictions = []  # Clear old predictions
                    original_start_index = copy.deepcopy(i)

                    for j in range(tuning_window_size):
                        if j + model_ws >= tuning_window_size:
                            j = tuning_window_size - j
                        else:
                            j += model_ws
                        start_idx = i
                        end_idx = min(i + j, len(X))
                        i += model_ws


                        if isinstance(X, pd.DataFrame):
                            X_tune_batch = X.iloc[start_idx:end_idx].values
                        else:  # numpy array
                            X_tune_batch = X[start_idx:end_idx].reshape(-1, X.shape[1])

                        model.predict_then_fit(X_tune_batch, y[start_idx:end_idx])
                    model_r2 = r2_score(y_true=y[original_start_index:end_idx], y_pred=model.predictions)
                    if prev_r2 <= model_r2:
                        prev_r2 = model_r2
                        best_parameters = copy.deepcopy(model.parameters)


                #counter_examples += tuning_window_size

                tuning_duration = time.time() - tuning_start_time
                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                kde_model.set_parameters(best_parameters)
                print(f"Best parameters after tuning: {best_parameters}\n")

            else:
                raise TypeError('Tuning is only available for mse, mae and r2 scores.')

            print(f'before i+=tuning_window_size, i is : {i}')
            i += tuning_window_size
            print(f'after i+=tuning_window_size, i is : {i}')


        else:
            #print(f"best_parameters before training: {kde_model.parameters}")
            #print(f" i is : {i}")
            if str(type(X)) == '<class \'pandas.core.frame.DataFrame\'>':
                instance_features = X.iloc[i:i + 1].values
            elif str(type(X)) == '<class \'numpy.ndarray\'>':
                instance_features = X[i].reshape(1, -1)
            else:
                raise TypeError('X type must either be pandas.core.frame.DataFrame or numpy.ndarray')

            actual_value = y[i]
            training_labels.append(actual_value)

            if not best_parameters:
                best_parameters = {'lambda_selected':0.0, 'h':1, 'window_size':1, 'kernel':CustomGaussianKernel, 'kde_type':'labeled', 'range_of_bin':0}


            # Predict with HoeffdingTree
            ht_pred = ht.predict(instance_features)[0]
            ht_predictions.append(ht_pred)

            # Predict with HoeffdingTree and Hierarchical Shrinkage
            ht_hs_pred = ht.predictHSNew(instance_features, lambda_selected=best_parameters['lambda_selected'])[0]
            ht_hs_predictions.append(ht_hs_pred)

            train_x.append(instance_features)
            train_y.append(actual_value)

            count_window += 1

            if count_window == best_parameters['window_size'] or count_window >= len(X) - i:
                # Predict and then Train the Incremental KDE Model
                #print("Predicting and training Incremental KDE Model...")
                kde_model.predict_then_fit(X=train_x, y=train_y)
                train_x = []
                train_y = []

                count_window = 0


            ht.partial_fit(instance_features, [actual_value])


            i += 1

    # Return predictions
    print(f"Size of ht predictions: {len(ht_predictions)}")
    print(f"Size of kde predictions: {len(kde_model.predictions)}")
    print(f"Size of actual labels: {len(training_labels)}")
    assert len(ht_predictions) == len(kde_model.predictions) == len(training_labels), "Mismatch in prediction lengths between models and actual data!"

    return ht_predictions, ht_hs_predictions, kde_model.predictions, training_labels



def california_experiment(metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }):
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import MinMaxScaler

    # Load data
    data = fetch_california_housing(as_frame=True)
    features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
    data = clean_data(data, features_of_interest)
    X, y = data.data, data.target  # small batch for quick test

    # Preprocessing
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Run tuning & training
    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train(
        parameters=parameters,
        X=X_scaled,
        y=y,
        tune_metric_selection=metric,
        tuning_times=4
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/california/ht_preds.txt', ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/california/ht_hs_preds.txt', ht_hs_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/california/kde_preds.txt', kde_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/california/y_true.txt', y_true)

    # Basic checks
    assert len(ht_preds) == len(y_true)
    assert len(ht_hs_preds) == len(y_true)
    assert len(kde_preds) == len(y_true)
    print("Test passed: All prediction lists are of correct length!")

    # Final MSE
    from sklearn.metrics import mean_squared_error
    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)
    mse_kde = mean_squared_error(y_true, kde_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"HAT MSE:      {mse_ht:.4f}")
    print(f"HAT+HS MSE:   {mse_ht_hs:.4f}")
    print(f"HAT with KDE MSE:     {mse_kde:.4f}")



def california_experiment_hs(metric='mse', lambda_list=[0, 1, 10, 50, 100], tuning_times=4):
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import MinMaxScaler

    # Load data
    data = fetch_california_housing(as_frame=True)
    features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
    data = clean_data(data, features_of_interest)
    X, y = data.data, data.target  # small batch for quick test

    # Preprocessing
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Run tuning & training
    ht_preds, ht_hs_preds, y_true = tune_and_train_on_hierarchical_shrinkage(
        lambda_list=lambda_list,
        X=X_scaled,
        y=y,
        tune_metric_selection=metric,
        tuning_times=tuning_times
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/hs/california/ht_preds.txt', ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/hs/california/ht_hs_preds.txt', ht_hs_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/hs/california/y_true.txt', y_true)

    # Basic checks
    assert len(ht_preds) == len(y_true)
    assert len(ht_hs_preds) == len(y_true)
    print("Test passed: All prediction lists are of correct length!")

    # Final MSE
    from sklearn.metrics import mean_squared_error
    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"HT MSE:      {mse_ht:.4f}")
    print(f"HT+HS MSE:   {mse_ht_hs:.4f}")


def NYTaxi_experiment_hs(metric='mse', lambda_list=[0, 1, 10, 50, 100], tuning_times=4):
    # ---------- helpers ----------
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0088  # km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return (2 * R * np.arcsin(np.sqrt(a))).astype(np.float32)

    def add_time_feats(df, col):
        dt = pd.to_datetime(df[col], errors="coerce")
        df = df.loc[dt.notna()].copy()
        dt = dt.loc[dt.notna()]
        df["pu_hour"] = dt.dt.hour.astype(np.int16)
        df["pu_dow"] = dt.dt.dayofweek.astype(np.int8)  # 0=Mon
        df["pu_month"] = dt.dt.month.astype(np.int8)
        df["is_weekend"] = (df["pu_dow"] >= 5).astype(np.int8)
        df["is_rush"] = ((df["pu_hour"].between(7, 10)) | (df["pu_hour"].between(16, 19))).astype(np.int8)
        return df

    def sanitize_coords(df, plat, plon, dlat, dlon):
        lat_ok = df[plat].between(40.4, 41.1) & df[dlat].between(40.4, 41.1)
        lon_ok = df[plon].between(-74.3, -73.5) & df[dlon].between(-74.3, -73.5)
        return df[lat_ok & lon_ok].copy()

    df = pd.read_csv(
        "/Users/pantia-marinaalchirch/Documents/OU/Research/Streams/data/nyc_taxi_kaggle/train.csv", nrows=10000)
    # target
    y = df["trip_duration"].astype(np.float32)

    # (recommended) clip and log1p stabilize the heavy tail
    y = y.clip(lower=60, upper=2 * 60 * 60)
    y = np.log1p(y).astype(np.float32)

    # ---------- build X ----------
    # keep only useful columns (adjust names if needed)
    needed = [
        "vendor_id",
        "pickup_datetime", "dropoff_datetime",
        "passenger_count",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        # "store_and_fwd_flag",  # include if present
    ]
    X = df[[c for c in needed if c in df.columns]].copy()

    # time features from pickup (most predictive & cheap)
    if "pickup_datetime" in X.columns:
        X = add_time_feats(X, "pickup_datetime")

    # sanitize coords (drop obvious outliers & NaNs)
    X = sanitize_coords(
        X,
        plat="pickup_latitude", plon="pickup_longitude",
        dlat="dropoff_latitude", dlon="dropoff_longitude"
    )

    # fast distance/bearing features (drop raw timestamps afterwards)
    X["haversine_km"] = haversine(
        X["pickup_latitude"].values, X["pickup_longitude"].values,
        X["dropoff_latitude"].values, X["dropoff_longitude"].values
    )
    X["manhattan_km"] = (
                                (X["pickup_latitude"] - X["dropoff_latitude"]).abs() +
                                (X["pickup_longitude"] - X["dropoff_longitude"]).abs()
                        ) * 111.0
    X["bearing_deg"] = np.degrees(np.arctan2(
        np.sin(np.radians(X["dropoff_longitude"] - X["pickup_longitude"])) *
        np.cos(np.radians(X["dropoff_latitude"])),
        np.cos(np.radians(X["pickup_latitude"])) *
        np.sin(np.radians(X["dropoff_latitude"])) -
        np.sin(np.radians(X["pickup_latitude"])) *
        np.cos(np.radians(X["dropoff_latitude"])) *
        np.cos(np.radians(X["dropoff_longitude"] - X["pickup_longitude"]))
    )).astype(np.float32)
    X["bearing_deg"] = ((X["bearing_deg"] + 360) % 360).astype(np.float32)

    # light cleanup / downcast
    if "passenger_count" in X.columns:
        X["passenger_count"] = pd.to_numeric(X["passenger_count"], errors="coerce").fillna(1).clip(1, 6).astype(np.int8)
    if "vendor_id" in X.columns:
        X["vendor_id"] = pd.to_numeric(X["vendor_id"], errors="coerce").fillna(1).astype(np.int8)
    if "store_and_fwd_flag" in X.columns:
        X["store_and_fwd_flag"] = X["store_and_fwd_flag"].astype(str).str.upper().map({"Y": 1, "N": 0}).fillna(
            0).astype(np.int8)

    # drop raw timestamps; (optionally) drop raw lat/lon to make trees much faster
    for c in ["pickup_datetime", "dropoff_datetime"]:
        if c in X.columns: X.drop(columns=c, inplace=True)

    # comment the next line if you want to keep raw coords:
    X.drop(columns=["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"], inplace=True)

    # final numeric cast + fill
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for c in X.columns:
        if X[c].dtype == "float64": X[c] = X[c].astype(np.float32)

    y = y.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)

    n_samples, n_features = X.shape
    print(f'samples : {n_samples}, number of features : {n_features}')
    print(f'y shape : {y.shape}')

    print("Processing dataset NYC TAXI")

    # Run tuning & training
    ht_preds, ht_hs_preds, y_true = tune_and_train_on_hierarchical_shrinkage(
        lambda_list=lambda_list,
        X=X,
        y=y,
        tune_metric_selection=metric,
        tuning_times=tuning_times
    )

    np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/NYTaxi/ht_preds.txt',
        ht_preds)
    np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/NYTaxi/ht_hs_preds.txt',
        ht_hs_preds)
    np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/NYTaxi/y_true.txt',
        y_true)

    # Basic checks
    assert len(ht_preds) == len(y_true)
    assert len(ht_hs_preds) == len(y_true)
    print("Test passed: All prediction lists are of correct length!")

    # Final MSE
    from sklearn.metrics import mean_squared_error
    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"HT MSE:      {mse_ht:.4f}")
    print(f"HT+HS MSE:   {mse_ht_hs:.4f}")





def semiconductors_experiment_sum_of_layers(metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }):
    data = pd.read_csv(
        "/Users/pantia-marinaalchirch/Documents/OU/Research/Streams/data/semiconductors_korean/train.csv", nrows=20000)


    data['sum_of_layers'] = data[['layer_1', 'layer_2', 'layer_3', 'layer_4']].sum(axis=1)
    data = data.drop(columns={"layer_1", "layer_2", "layer_3", "layer_4"})

    y = data['sum_of_layers']
    X = data.drop(columns={"sum_of_layers"})

    X = X.astype(np.float32)



    # scale X
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    pca = PCA(n_components=100)
    X = pca.fit_transform(X_scaled)

    print(f'After PCA, X.shape is {X.shape}')

    # Run tuning & training
    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train(
        parameters=parameters,
        X=X,
        y=y,
        tune_metric_selection=metric,
        tuning_times=4
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/Film_Thickness/sum_of_layers/ht_preds.txt',
               ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/Film_Thickness/sum_of_layers/ht_hs_preds.txt',
               ht_hs_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/Film_Thickness/sum_of_layers/kde_preds.txt',
               kde_preds)
    np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/Film_Thickness/sum_of_layers/y_true.txt',
        y_true)

    # Basic checks
    assert len(ht_preds) == len(y_true)
    assert len(ht_hs_preds) == len(y_true)
    assert len(kde_preds) == len(y_true)
    print("Test passed: All prediction lists are of correct length!")

    # Final MSE
    from sklearn.metrics import mean_squared_error
    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)
    mse_kde = mean_squared_error(y_true, kde_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"HT MSE:      {mse_ht:.4f}")
    print(f"HT+HS MSE:   {mse_ht_hs:.4f}")
    print(f"KDE MSE:     {mse_kde:.4f}")



def NYTaxi_experiment(metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }, tuning_times=4):

    # ---------- helpers ----------
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0088  # km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return (2 * R * np.arcsin(np.sqrt(a))).astype(np.float32)

    def add_time_feats(df, col):
        dt = pd.to_datetime(df[col], errors="coerce")
        df = df.loc[dt.notna()].copy()
        dt = dt.loc[dt.notna()]
        df["pu_hour"] = dt.dt.hour.astype(np.int16)
        df["pu_dow"] = dt.dt.dayofweek.astype(np.int8)  # 0=Mon
        df["pu_month"] = dt.dt.month.astype(np.int8)
        df["is_weekend"] = (df["pu_dow"] >= 5).astype(np.int8)
        df["is_rush"] = ((df["pu_hour"].between(7, 10)) | (df["pu_hour"].between(16, 19))).astype(np.int8)
        return df

    def sanitize_coords(df, plat, plon, dlat, dlon):
        lat_ok = df[plat].between(40.4, 41.1) & df[dlat].between(40.4, 41.1)
        lon_ok = df[plon].between(-74.3, -73.5) & df[dlon].between(-74.3, -73.5)
        return df[lat_ok & lon_ok].copy()

    df = pd.read_csv(
        "/Users/pantia-marinaalchirch/Documents/OU/Research/Streams/data/nyc_taxi_kaggle/train.csv", nrows=10000)
    # target
    y = df["trip_duration"].astype(np.float32)

    # (recommended) clip and log1p stabilize the heavy tail
    y = y.clip(lower=60, upper=2 * 60 * 60)
    y = np.log1p(y).astype(np.float32)

    # ---------- build X ----------
    # keep only useful columns (adjust names if needed)
    needed = [
        "vendor_id",
        "pickup_datetime", "dropoff_datetime",
        "passenger_count",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        # "store_and_fwd_flag",  # include if present
    ]
    X = df[[c for c in needed if c in df.columns]].copy()

    # time features from pickup (most predictive & cheap)
    if "pickup_datetime" in X.columns:
        X = add_time_feats(X, "pickup_datetime")

    # sanitize coords (drop obvious outliers & NaNs)
    X = sanitize_coords(
        X,
        plat="pickup_latitude", plon="pickup_longitude",
        dlat="dropoff_latitude", dlon="dropoff_longitude"
    )

    # fast distance/bearing features (drop raw timestamps afterwards)
    X["haversine_km"] = haversine(
        X["pickup_latitude"].values, X["pickup_longitude"].values,
        X["dropoff_latitude"].values, X["dropoff_longitude"].values
    )
    X["manhattan_km"] = (
                                (X["pickup_latitude"] - X["dropoff_latitude"]).abs() +
                                (X["pickup_longitude"] - X["dropoff_longitude"]).abs()
                        ) * 111.0
    X["bearing_deg"] = np.degrees(np.arctan2(
        np.sin(np.radians(X["dropoff_longitude"] - X["pickup_longitude"])) *
        np.cos(np.radians(X["dropoff_latitude"])),
        np.cos(np.radians(X["pickup_latitude"])) *
        np.sin(np.radians(X["dropoff_latitude"])) -
        np.sin(np.radians(X["pickup_latitude"])) *
        np.cos(np.radians(X["dropoff_latitude"])) *
        np.cos(np.radians(X["dropoff_longitude"] - X["pickup_longitude"]))
    )).astype(np.float32)
    X["bearing_deg"] = ((X["bearing_deg"] + 360) % 360).astype(np.float32)

    # light cleanup / downcast
    if "passenger_count" in X.columns:
        X["passenger_count"] = pd.to_numeric(X["passenger_count"], errors="coerce").fillna(1).clip(1, 6).astype(np.int8)
    if "vendor_id" in X.columns:
        X["vendor_id"] = pd.to_numeric(X["vendor_id"], errors="coerce").fillna(1).astype(np.int8)
    if "store_and_fwd_flag" in X.columns:
        X["store_and_fwd_flag"] = X["store_and_fwd_flag"].astype(str).str.upper().map({"Y": 1, "N": 0}).fillna(
            0).astype(np.int8)

    # drop raw timestamps; (optionally) drop raw lat/lon to make trees much faster
    for c in ["pickup_datetime", "dropoff_datetime"]:
        if c in X.columns: X.drop(columns=c, inplace=True)

    # comment the next line if you want to keep raw coords:
    X.drop(columns=["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"], inplace=True)

    # final numeric cast + fill
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for c in X.columns:
        if X[c].dtype == "float64": X[c] = X[c].astype(np.float32)

    y = y.loc[X.index].reset_index(drop=True)
    X = X.reset_index(drop=True)


    n_samples, n_features = X.shape
    print(f'samples : {n_samples}, number of features : {n_features}')
    print(f'y shape : {y.shape}')

    print("Processing dataset NYC TAXI")

    # Run tuning & training
    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train(
        parameters=parameters,
        X=X,
        y=y,
        tune_metric_selection=metric,
        tuning_times=tuning_times
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/NYTaxi/ht_preds.txt', ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/NYTaxi/ht_hs_preds.txt', ht_hs_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/NYTaxi/kde_preds.txt', kde_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/NYTaxi/y_true.txt', y_true)

    # Basic checks
    assert len(ht_preds) == len(y_true)
    assert len(ht_hs_preds) == len(y_true)
    assert len(kde_preds) == len(y_true)
    print("Test passed: All prediction lists are of correct length!")

    # Final MSE
    from sklearn.metrics import mean_squared_error
    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)
    mse_kde = mean_squared_error(y_true, kde_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"HAT MSE:      {mse_ht:.4f}")
    print(f"HAT+HS MSE:   {mse_ht_hs:.4f}")
    print(f"HAT with KDE MSE:     {mse_kde:.4f}")


def electric_power_comsumption_experiment(metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }, tuning_times=4):

    path = "/Users/pantia-marinaalchirch/Documents/OU/Research/Streams/data/household_power_consumption.txt"

    # read first 20k; treat '?' as NaN
    cols = ["Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df = pd.read_csv(path, sep=";", nrows=100000, usecols=cols,
                     na_values="?", low_memory=False)

    # parse datetime (day-first in this dataset)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df["hour"] = df["datetime"].dt.hour.astype("int8")
    df["dow"] = df["datetime"].dt.dayofweek.astype("int8")
    # df["epoch_s"] = df["datetime"].astype("int64") / 1e9  # alternative single numeric time feature

    y = pd.to_numeric(df["Global_active_power"], errors="coerce")

    X = df[[
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "hour", "dow"  # or "epoch_s"
    ]].copy()

    mask = df["datetime"].notna() & y.notna() & (~X.isna().any(axis=1))
    X = X.loc[mask].astype(np.float32).reset_index(drop=True)
    y = y.loc[mask].astype(np.float32).reset_index(drop=True)

    print(X.shape, y.shape)

    # Run tuning & training
    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train(
        parameters=parameters,
        X=X,
        y=y,
        tune_metric_selection=metric,
        tuning_times=tuning_times
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/EPowerConsumption/ht_preds.txt',
               ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/EPowerConsumption/ht_hs_preds.txt',
               ht_hs_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/EPowerConsumption/kde_preds.txt',
               kde_preds)
    np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hat/EPowerConsumption/y_true.txt',
        y_true)

    # Basic checks
    assert len(ht_preds) == len(y_true)
    assert len(ht_hs_preds) == len(y_true)
    assert len(kde_preds) == len(y_true)
    print("Test passed: All prediction lists are of correct length!")

    # Final MSE
    from sklearn.metrics import mean_squared_error
    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)
    mse_kde = mean_squared_error(y_true, kde_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"HAT MSE:      {mse_ht:.4f}")
    print(f"HAT+HS MSE:   {mse_ht_hs:.4f}")
    print(f"HT with KDE MSE:     {mse_kde:.4f}")



def electric_power_comsumption_experiment_hs(metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }, tuning_times=4):

    path = "/Users/pantia-marinaalchirch/Documents/OU/Research/Streams/data/household_power_consumption.txt"

    # read first 20k; treat '?' as NaN
    cols = ["Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    df = pd.read_csv(path, sep=";", nrows=100000, usecols=cols,
                     na_values="?", low_memory=False)

    # parse datetime (day-first in this dataset)
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce")
    df["hour"] = df["datetime"].dt.hour.astype("int8")
    df["dow"] = df["datetime"].dt.dayofweek.astype("int8")
    # df["epoch_s"] = df["datetime"].astype("int64") / 1e9  # alternative single numeric time feature

    y = pd.to_numeric(df["Global_active_power"], errors="coerce")

    X = df[[
        "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
        "hour", "dow"  # or "epoch_s"
    ]].copy()

    mask = df["datetime"].notna() & y.notna() & (~X.isna().any(axis=1))
    X = X.loc[mask].astype(np.float32).reset_index(drop=True)
    y = y.loc[mask].astype(np.float32).reset_index(drop=True)

    print(X.shape, y.shape)

    # Run tuning & training
    ht_preds, ht_hs_preds, y_true = tune_and_train_on_hierarchical_shrinkage(
        parameters=parameters,
        X=X,
        y=y,
        tune_metric_selection=metric,
        tuning_times=tuning_times
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/EPowerConsumption/ht_preds.txt',
               ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/EPowerConsumption/ht_hs_preds.txt',
               ht_hs_preds)
    np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/EPowerConsumption/y_true.txt',
        y_true)

    # Basic checks
    assert len(ht_preds) == len(y_true)
    assert len(ht_hs_preds) == len(y_true)
    print("Test passed: All prediction lists are of correct length!")

    # Final MSE
    from sklearn.metrics import mean_squared_error
    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"HT MSE:      {mse_ht:.4f}")
    print(f"HT+HS MSE:   {mse_ht_hs:.4f}")


"""Extra code for wandb chatgpt generated. Starts form below here"""

# pip install wandb
import time, copy, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import wandb

def semiconductors_experiment_sum_of_layers_wandb(
    metric='mse',
    parameters={
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    },
    nrows=20_000,
    #pca_components=100,
    log_every=200,           # <--- how often to log progressive metrics
    project="semiconductors-ht-wandb",
    run_name="sum_of_layers_ht_stream"
):
    """
    Runs your experiment and streams live metrics to Weights & Biases.
    """

    # --- W&B init ---
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "task": "regression",
            "dataset": "semiconductors_korean/train.csv",
            "nrows": nrows,
            "target": "sum_of_layers",
            #"pca_components": pca_components,
            "scaler": "StandardScaler",
            "metric_for_tuning": metric,
            "log_every": log_every,
            # grid used to create KDE models
            "lambda_list": parameters['lambda_list'],
            "h_list": parameters['h_list'],
            "window_size_list": parameters['window_size_list'],
        }
    )
    cfg = wandb.config

    # optional: pretty x-axes for tuning
    wandb.define_metric("tuning/candidate_step")
    wandb.define_metric("tuning/mse", step_metric="tuning/candidate_step")

    # nice x-axes
    wandb.define_metric("seen_examples")
    for m in ["ht_mse", "ht_hs_mse", "kde_mse"]:
        wandb.define_metric(m, step_metric="seen_examples")

    t0 = time.time()

    # --- Data load & prep ---
    data = pd.read_csv(
        "/Users/pantia-marinaalchirch/Documents/OU/Research/Streams/data/semiconductors_korean/train.csv",
        nrows=nrows
    )

    data['sum_of_layers'] = data[['layer_1', 'layer_2', 'layer_3', 'layer_4']].sum(axis=1)
    data = data.drop(columns={"layer_1", "layer_2", "layer_3", "layer_4"})

    y = data['sum_of_layers'].to_numpy()
    X = data.drop(columns={"sum_of_layers"}).astype(np.float32)

    #scaler_X = StandardScaler()
    #X_scaled = scaler_X.fit_transform(X)

    #pca = PCA(n_components=pca_components)
    #X = pca.fit_transform(X_scaled)

    wandb.log({"prep/num_features_after_pca": X.shape[1]})
    print(f'After PCA, X.shape is {X.shape}')

    # --- Run tuning & training (instrumented) ---
    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train_wandb(
        parameters=parameters,
        X=X,
        y=y,
        tune_metric_selection=metric,
        tuning_times=4,
        log_every=log_every
    )

    # Save outputs to disk (as before)
    out_dir = "/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/Film_Thickness/sum_of_layers"
    os.makedirs(out_dir, exist_ok=True)

    np.savetxt(f'{out_dir}/ht_preds.txt', ht_preds)
    np.savetxt(f'{out_dir}/ht_hs_preds.txt', ht_hs_preds)
    np.savetxt(f'{out_dir}/kde_preds.txt', kde_preds)
    np.savetxt(f'{out_dir}/y_true.txt', y_true)

    # Log files to W&B as an artifact (optional but handy for later comparison)
    art = wandb.Artifact("predictions_sum_of_layers", type="predictions")
    art.add_file(f'{out_dir}/ht_preds.txt')
    art.add_file(f'{out_dir}/ht_hs_preds.txt')
    art.add_file(f'{out_dir}/kde_preds.txt')
    art.add_file(f'{out_dir}/y_true.txt')
    wandb.log_artifact(art)

    # Basic checks
    assert len(ht_preds) == len(y_true)
    assert len(ht_hs_preds) == len(y_true)
    assert len(kde_preds) == len(y_true)
    print("Test passed: All prediction lists are of correct length!")

    # Final MSE
    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)
    mse_kde = mean_squared_error(y_true, kde_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"HT MSE:      {mse_ht:.4f}")
    print(f"HT+HS MSE:   {mse_ht_hs:.4f}")
    print(f"KDE MSE:     {mse_kde:.4f}")

    wandb.log({
        "final/ht_mse": mse_ht,
        "final/ht_hs_mse": mse_ht_hs,
        "final/kde_mse": mse_kde,
        "runtime_seconds": time.time() - t0
    })

    wandb.finish()


def tune_and_train_wandb(parameters, X, y, tune_metric_selection='mse', tuning_times=4, log_every=200):
    """
    Your original tune_and_train with W&B live logging of progressive MSE.
    Only minimal edits: we compute + log running MSEs when data are aligned.
    """
    # --- originals ---
    dummy_parameters = {
        'lambda_selected': 0.0,
        'h': 1,
        'window_size': 1,
        'kernel': CustomGaussianKernel(sigma=1.0),
        'kde_type': 'labeled',
        'range_of_bin': 0
    }

    kde_model = IncrKDEModel(y=y, model=HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42), parameters=dummy_parameters)
    ht = HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42)
    ht_predictions = []
    ht_hs_predictions = []

    lambda_list = parameters['lambda_list']
    h_list = parameters['h_list']
    window_size_list = parameters['window_size_list']

    kde_models = create_models_semiconductors_labeled(lambda_list=lambda_list, h_list=h_list, window_size_list=window_size_list, y=y)

    count_window = 0
    train_x = []
    train_y = []

    training_labels = []
    best_parameters = {}

    tuning_window_size = int(len(X) / (2 * tuning_times))
    tuning_start_indices = [int(i * len(X) / tuning_times) for i in range(tuning_times)]
    tuning_start_time = None
    round_num = None

    if isinstance(X, pd.Series): X = X.values
    if isinstance(y, pd.Series): y = y.values
    X = np.asarray(X)
    y = np.asarray(y)

    # For progressive MSE curves
    def log_progress_if_ready(i):
        """
        Log progressive MSE when all three sequences (ht, ht_hs, kde) align
        and at the desired cadence.
        """
        if len(ht_predictions) and len(training_labels) and len(kde_model.predictions):
            n = min(len(ht_predictions), len(ht_hs_predictions), len(kde_model.predictions), len(training_labels))
            if n > 3 and ((i + 1) % log_every == 0):
                wandb.log({
                    "seen_examples": n,
                    "ht_mse": mean_squared_error(training_labels[:n], ht_predictions[:n]),
                    "ht_hs_mse": mean_squared_error(training_labels[:n], ht_hs_predictions[:n]),
                    "kde_mse": mean_squared_error(training_labels[:n], kde_model.predictions[:n])
                })

    i = 0
    while i < len(X):
        # --- TUNING PHASES ---
        if i in tuning_start_indices:
            print("models to tune : " + str(len(kde_models)))
            round_num = tuning_start_indices.index(i) + 1
            tuning_start_time = time.time()
            print(f'tuning_start_indices: {tuning_start_indices}')
            print(f"Tuning [{round_num}/{tuning_times}] at example {i}...")

            # Track which metric we're tuning on
            wandb.log({"tuning/round": round_num, "tuning/start_at": i, "tuning/metric": tune_metric_selection})

            if tune_metric_selection == 'mse':
                prev_mse = np.inf
                candidate_step = 0  # global-ish counter across rounds for smooth line chart

                # Table to collect this round's candidates (optional but useful)
                round_table = wandb.Table(columns=[
                    "round", "candidate_id", "window_size", "lambda_selected", "h", "mse"
                ])
                count_models = 1
                for cand_id, model in enumerate(kde_models):
                    print(f'Tuning model : {count_models}')
                    model_ws = model.parameters['window_size']
                    assert model_ws <= tuning_window_size, \
                        f"Model window size {model_ws} exceeds tuning window size {tuning_window_size}"

                    model.predictions = []
                    j = 0
                    index = copy.deepcopy(i)
                    flag = False
                    end_idx = copy.deepcopy(index)

                    while j < tuning_window_size:
                        #if j % 100 == 0 and j!=0:
                        #    print(f"Tuning at example j : {j}...")
                        start_idx = copy.deepcopy(index)
                        if j + model_ws >= tuning_window_size:
                            end_idx = copy.deepcopy(i + tuning_window_size)
                            index = index + (tuning_window_size - j)
                            flag = True
                        else:
                            j += copy.deepcopy(model_ws)
                            index += copy.deepcopy(model_ws)

                        if not flag:
                            end_idx += model_ws

                        # slice features
                        if isinstance(X, pd.DataFrame):
                            X_tune_batch = X.iloc[start_idx:end_idx].values
                        else:
                            X_tune_batch = X[start_idx:end_idx].reshape(-1, X.shape[1])

                        model.predict_then_fit(X_tune_batch, y[start_idx:end_idx])
                        if flag:
                            j += model_ws
                            break

                    # === candidate metric ===
                    model_mse = mean_squared_error(y_true=y[i:end_idx], y_pred=model.predictions)

                    # --- (A) per-candidate scalar log for live line chart ---
                    candidate_step += 1
                    wandb.log({
                        "tuning/round": round_num,
                        "tuning/candidate_step": candidate_step,
                        "tuning/candidate_id": cand_id,
                        "tuning/window_size": model.parameters.get("window_size"),
                        "tuning/lambda_selected": model.parameters.get("lambda_selected"),
                        "tuning/h": model.parameters.get("h"),
                        "tuning/mse": model_mse,
                    })

                    # --- (B) also add to a round table (see below) ---
                    round_table.add_data(
                        round_num, cand_id,
                        model.parameters.get("window_size"),
                        model.parameters.get("lambda_selected"),
                        model.parameters.get("h"),
                        model_mse
                    )

                    # track best
                    if prev_mse >= model_mse:
                        prev_mse = model_mse
                        best_parameters = copy.deepcopy(model.parameters)
                    count_models += 1

                # Finish round: set best params + log the table for this round
                tuning_duration = time.time() - tuning_start_time
                kde_model.set_parameters(best_parameters)
                wandb.log({
                    "tuning/round": round_num,
                    "tuning/duration_sec": tuning_duration,
                    "tuning/best_params.window_size": best_parameters.get("window_size"),
                    "tuning/best_params.lambda_selected": best_parameters.get("lambda_selected"),
                    "tuning/best_params.h": best_parameters.get("h"),
                    f"tuning/round_{round_num}_candidates": round_table,  # <--- table in UI
                })
                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                print(f"Best parameters after tuning: {best_parameters}\n")


            elif tune_metric_selection == 'mae':
                prev_mae = np.inf
                for model in kde_models:
                    model_ws = model.parameters['window_size']
                    assert model_ws <= tuning_window_size, f"Model window size {model_ws} exceeds tuning window size {tuning_window_size}"
                    model.predictions = []
                    original_start_index = copy.deepcopy(i)

                    for j in range(tuning_window_size):
                        if j + model_ws >= tuning_window_size:
                            j = tuning_window_size - j
                        else:
                            j += model_ws
                        start_idx = i
                        end_idx = min(i + j, len(X))

                        if isinstance(X, pd.DataFrame):
                            X_tune_batch = X.iloc[start_idx:end_idx].values
                        else:
                            X_tune_batch = X[start_idx:end_idx].reshape(-1, X.shape[1])

                        model.predict_then_fit(X_tune_batch, y[start_idx:end_idx])

                    model_mae = mean_absolute_error(y_true=y[original_start_index:end_idx], y_pred=model.predictions)
                    if prev_mae >= model_mae:
                        prev_mae = model_mae
                        best_parameters = copy.deepcopy(model.parameters)

                tuning_duration = time.time() - tuning_start_time
                kde_model.set_parameters(best_parameters)
                wandb.log({
                    "tuning/round": round_num,
                    "tuning/duration_sec": tuning_duration,
                    "tuning/best_params.window_size": best_parameters.get("window_size"),
                    "tuning/best_params.lambda_selected": best_parameters.get("lambda_selected"),
                    "tuning/best_params.h": best_parameters.get("h"),
                })
                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                print(f"Best parameters after tuning: {best_parameters}\n")

            elif tune_metric_selection == 'r2':
                prev_r2 = -1
                for model in kde_models:
                    model_ws = model.parameters['window_size']
                    assert model_ws <= tuning_window_size, f"Model window size {model_ws} exceeds tuning window size {tuning_window_size}"
                    model.predictions = []
                    original_start_index = copy.deepcopy(i)

                    for j in range(tuning_window_size):
                        if j + model_ws >= tuning_window_size:
                            j = tuning_window_size - j
                        else:
                            j += model_ws
                        start_idx = i
                        end_idx = min(i + j, len(X))

                        if isinstance(X, pd.DataFrame):
                            X_tune_batch = X.iloc[start_idx:end_idx].values
                        else:
                            X_tune_batch = X[start_idx:end_idx].reshape(-1, X.shape[1])

                        model.predict_then_fit(X_tune_batch, y[start_idx:end_idx])

                    model_r2 = r2_score(y_true=y[original_start_index:end_idx], y_pred=model.predictions)
                    if prev_r2 <= model_r2:
                        prev_r2 = model_r2
                        best_parameters = copy.deepcopy(model.parameters)

                tuning_duration = time.time() - tuning_start_time
                kde_model.set_parameters(best_parameters)
                wandb.log({
                    "tuning/round": round_num,
                    "tuning/duration_sec": tuning_duration,
                    "tuning/best_params.window_size": best_parameters.get("window_size"),
                    "tuning/best_params.lambda_selected": best_parameters.get("lambda_selected"),
                    "tuning/best_params.h": best_parameters.get("h"),
                })
                print(f"Finished Tuning [{round_num}/{tuning_times}] in {tuning_duration:.2f} seconds.\n")
                print(f"Best parameters after tuning: {best_parameters}\n")

            else:
                raise TypeError('Tuning is only available for mse, mae and r2 scores.')

            i += tuning_window_size
            print(f"best_parameters before training: {kde_model.parameters}")

        # --- TRAINING / STREAMING PHASE ---
        else:
            print("inside training procedure")
            print(f"best_parameters inside else: {kde_model.parameters}")
            if str(type(X)) == '<class \'pandas.core.frame.DataFrame\'>':
                instance_features = X.iloc[i:i + 1].values
            elif str(type(X)) == '<class \'numpy.ndarray\'>':
                instance_features = X[i].reshape(1, -1)
            else:
                raise TypeError('X type must either be pandas.core.frame.DataFrame or numpy.ndarray')

            actual_value = y[i]
            training_labels.append(actual_value)

            if not best_parameters:
                best_parameters = {
                    'lambda_selected': 0.0, 'h': 1, 'window_size': 1,
                    'kernel': CustomGaussianKernel, 'kde_type': 'labeled', 'range_of_bin': 0
                }

            # Predict with HT
            ht_pred = ht.predict(instance_features)[0]
            ht_predictions.append(ht_pred)

            # Predict with HT + HS
            ht_hs_pred = ht.predictHSNew(instance_features, lambda_selected=best_parameters['lambda_selected'])[0]
            ht_hs_predictions.append(ht_hs_pred)

            # Batch buffer for KDE model
            train_x.append(instance_features)
            train_y.append(actual_value)
            count_window += 1

            # When window full (or end), predict-then-fit KDE model
            if count_window == best_parameters['window_size'] or count_window >= len(X) - i:
                kde_model.predict_then_fit(X=train_x, y=train_y)
                train_x, train_y = [], []
                count_window = 0

            # Train HT incrementally
            ht.partial_fit(instance_features, [actual_value])

            # Live logging (only when lengths align)
            log_progress_if_ready(i)

            i += 1

    # Return predictions
    print(f"Size of ht predictions: {len(ht_predictions)}")
    print(f"Size of kde predictions: {len(kde_model.predictions)}")
    print(f"Size of actual labels: {len(training_labels)}")
    assert len(ht_predictions) == len(kde_model.predictions) == len(training_labels), \
        "Mismatch in prediction lengths between models and actual data!"

    # one last log at the end
    n = len(training_labels)
    wandb.log({
        "seen_examples": n,
        "ht_mse": mean_squared_error(training_labels, ht_predictions),
        "ht_hs_mse": mean_squared_error(training_labels, ht_hs_predictions),
        "kde_mse": mean_squared_error(training_labels, kde_model.predictions),
    })

    return ht_predictions, ht_hs_predictions, kde_model.predictions, training_labels
