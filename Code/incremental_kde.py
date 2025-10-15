import copy
import time

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from Code.california_housing_dataset_cleaning import clean_data
from Code.kernels import CustomGaussianKernel, CustomEpanechnikovKernel
from src.trees import HoeffdingTreeRegressor, HoeffdingAdaptiveTreeRegressor
from Code.model import IncrKDEModel
from Code.preprocess_NYTaxi_data import load_NYTaxi_data
from Code.preprocess_EPowerConsumption import load_EPowerConsumption



# Models to get tuned for different parameters used in kde
def create_models(model_name, lambda_list, h_list, window_size_list, y):
    kde_models = []

    lambda_list = lambda_list if lambda_list else [0]
    h_list = h_list if h_list else [1]
    window_size_list = window_size_list if window_size_list else [1]
    kernels=[CustomGaussianKernel(sigma=1), CustomEpanechnikovKernel()]
    range_of_bin_list = []
    if model_name == "ht":
        range_of_bin_list = [0, 1, 2, 5, 10]
    else:
        range_of_bin_list = [0, 1, 2]
    #range_of_bin_list = [0, 1, 2, 10]
    #range_of_bin_list = [0, 1, 2]
    #range_of_bin_list = [0, 1, 2, 5]

    parameters = {'lambda_selected': None, 'h': None, 'window_size': None, 'kernel': None,
                  'kde_type': None, 'range_of_bin': None}

    model_used = None
    if model_name == 'hat':
        print("model name is hat")
        model_used = HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42)
    if model_name == 'ht':
        print("model name is ht")
        model_used = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
    fall_back_kde_model = IncrKDEModel(y=y, model=model_used, parameters=parameters, fall_back=True)
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
                            kde_model = None
                            if model_name == 'hat':
                                kde_model = IncrKDEModel(y=y, model=HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42),parameters=parameters,
                                                         fall_back=False)
                            if model_name == 'ht':
                                kde_model = IncrKDEModel(y=y, model=HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42), parameters=parameters,
                                                         fall_back=False)
                            kde_models.append(kde_model)


    return kde_models

def tune_and_train_on_hierarchical_shrinkage(lambda_list, X, y, tune_metric_selection='mse', tuning_times=4):

    ht = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
    ht_predictions = []
    ht_hs_predictions = []

    training_labels = []
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

def tune_and_train(parameters, X, y, model_name='ht', tune_metric_selection='mse', tuning_times=4, no_lambda=True):


    # Initialize a KDE model with some dummy/default parameters before using the best_parameters returned from the tuning process
    dummy_parameters = {
        'lambda_selected': 0.0,
        'h': 1,
        'window_size': 1,
        'kernel': CustomGaussianKernel(sigma=1.0),
        'kde_type': 'labeled',
        'range_of_bin': 0
    }

    kde_model = None
    if model_name == 'hat':
        kde_model = IncrKDEModel(y=y,model=HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42), parameters=dummy_parameters)
    if model_name == 'ht':
        kde_model = IncrKDEModel(y=y, model=HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42), parameters=dummy_parameters)


    ht = None
    if model_name=='hat':
        print("initializing a hat")
        ht = HoeffdingAdaptiveTreeRegressor(leaf_prediction="perceptron", random_state=42)
    if model_name=='ht':
        print("initializing an ht")
        ht = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
    ht_predictions = []
    ht_hs_predictions = []

    lambda_list = parameters['lambda_list']
    h_list = parameters['h_list']
    window_size_list = parameters['window_size_list']

    if no_lambda:
        print("no lambda tuning in KDE")
        lambda_list = [0.0]
    # Create and initialize the KDE models used later for tuning
    kde_models = create_models(model_name=model_name, lambda_list=lambda_list, h_list=h_list, window_size_list=window_size_list, y=y)
    print(f'models to tune : {len(kde_models)}')
    count_window = 0
    train_x = []
    train_y = []

    training_labels = []

    best_parameters = {}
    best_lambda = 0

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

                #Perform tuning on lambda alone
                print(f"Tuning on lambda")
                prev_lambda_mse = np.inf
                print(f" models to tune : {len(lambda_list)}")
                for lambda_sel in parameters['lambda_list']:
                    j = 0
                    hs_preds = []
                    true_labels = []
                    while j < tuning_window_size:
                        if str(type(X)) == '<class \'pandas.core.frame.DataFrame\'>':
                            x_instance = X.iloc[j:j + 1].values
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
                        j += 1

                    # print(f'size of hs model predictions : {len(hs_preds)}')
                    # print(f'size of y_true to compare with : {len(true_labels)}')
                    model_mse = mean_squared_error(y_true=true_labels, y_pred=hs_preds)
                    print(f"lambda: {lambda_sel}, mse : {model_mse}")
                    if prev_lambda_mse >= model_mse:
                        prev_lambda_mse = model_mse
                        best_lambda = lambda_sel


                print(f"\nFinished Tuning on lambda")
                print(f'best model mse : {prev_lambda_mse}')
                print(f"Best lambda after tuning: {best_lambda}\n")


                print("Tune on KDE...")
                print(f" models to tune : {len(kde_models)}")
                prev_mse = np.inf
                counter = 1
                start_idx = 0
                end_idx = 0
                fall_back_mse = np.inf
                for model in kde_models:
                    if counter % 50 == 0:
                        print(f"tuning on model : {counter}")
                    model.predictions = []  # Clear old predictions
                    index = copy.deepcopy(i)
                    if not model.fall_back:

                        model_ws = model.parameters['window_size']
                        assert model_ws <= tuning_window_size, f"Model window size {model_ws} exceeds tuning window size {tuning_window_size}"

                        # original_start_index = copy.deepcopy(i)
                        k = 0
                        flag = False
                        # end_idx = copy.deepcopy(model_ws)
                        end_idx = copy.deepcopy(index)
                        while k < tuning_window_size:
                            start_idx = copy.deepcopy(index)
                            if k + model_ws >= tuning_window_size:
                                # print("if j + model_ws >= tuning_window_size:")
                                end_idx = copy.deepcopy(i + tuning_window_size)
                                index = index + (tuning_window_size - k)
                                flag = True
                            else:
                                k += copy.deepcopy(model_ws)
                                index += copy.deepcopy(model_ws)

                            if not flag:
                                end_idx += model_ws

                            if isinstance(X, pd.DataFrame):
                                X_tune_batch = X.iloc[start_idx:end_idx].values
                            else:  # numpy array
                                X_tune_batch = X[start_idx:end_idx].reshape(-1, X.shape[1])

                            # print(f'X_tune_batch size : {X_tune_batch.shape}')

                            model.predict_then_fit(X_tune_batch, y[start_idx:end_idx])
                            # print(f' j is : {j}')
                            # print(f'start_idx: {start_idx}, end_idx: {end_idx}')
                            # print(f'model.predictions size : {len(model.predictions)}')
                            if flag:
                                counter += 1
                                # print(f'counter: {counter}')
                                k += model_ws
                                break

                    else:
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

                # counter_examples += tuning_window_size
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
                    model.predictions = []

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
            ht_hs_pred = ht.predictHSNew(instance_features, lambda_selected=best_lambda)[0]
            ht_hs_predictions.append(ht_hs_pred)

            train_x.append(instance_features)
            train_y.append(actual_value)

            count_window += 1

            if count_window == best_parameters['window_size'] or count_window >= len(X) - i:
                # Predict and then Train the Incremental KDE Model
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

    return ht_predictions, ht_hs_predictions, kde_model.predictions, training_labels



def california_experiment(model_name='ht', no_lambda=True, n_rows=None, metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }):


    data = fetch_california_housing(as_frame=True)
    features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
    data = clean_data(data, features_of_interest)
    X, y = data.data, data.target

    if n_rows is not None:
        X = X[:n_rows]
        y = y[:n_rows]


    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train(
        parameters=parameters,
        X=X_scaled,
        y=y,
        tune_metric_selection=metric,
        tuning_times=4,
        no_lambda=no_lambda,
        model_name=model_name
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+model_name+'/california/ht_preds.txt', ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+model_name+'/california/ht_hs_preds.txt', ht_hs_preds)
    if no_lambda:
        np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+model_name+'/california/kde_no_hs_preds.txt', kde_preds)
    else:
        np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/' + model_name + '/california/kde_preds.txt', kde_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+model_name+'/california/y_true.txt', y_true)

    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)
    mse_kde = mean_squared_error(y_true, kde_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"{model_name} MSE:      {mse_ht:.4f}")
    print(f"{model_name}+HS MSE:   {mse_ht_hs:.4f}")
    print(f"{model_name}+KDE MSE:   {mse_kde:.4f}")





def NYTaxi_experiment(path, model_name='ht', no_lambda=True,  n_rows=1000, metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }, tuning_times=4):

    print("Processing dataset NYC TAXI")

    X, y = load_NYTaxi_data(path=path, n_rows=n_rows)

    n_samples, n_features = X.shape
    print(f'samples : {n_samples}, number of features : {n_features}')
    print(f'y shape : {y.shape}')

    # run tuning and training
    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train(
        parameters=parameters,
        X=X,
        y=y,
        model_name=model_name,
        tune_metric_selection=metric,
        tuning_times=tuning_times,
        no_lambda=no_lambda
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+ model_name +'/NYTaxi/ht_preds.txt', ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+ model_name +'/NYTaxi/ht_hs_preds.txt', ht_hs_preds)
    if no_lambda:
        np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+ model_name +'/NYTaxi/kde_no_hs_preds.txt', kde_preds)
    else:
        np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+ model_name +'/NYTaxi/kde_preds.txt', kde_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+ model_name +'/NYTaxi/y_true.txt', y_true)

    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)
    mse_kde = mean_squared_error(y_true, kde_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"{model_name} MSE:      {mse_ht:.4f}")
    print(f"{model_name}+HS MSE:   {mse_ht_hs:.4f}")
    print(f"{model_name}+KDE MSE:     {mse_kde:.4f}")


def electric_power_comsumption_experiment(path, model_name='ht', no_lambda=True, n_rows=1000, metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }, tuning_times=4):

    X, y = load_EPowerConsumption(path, n_rows)

    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train(
        parameters=parameters,
        X=X,
        y=y,
        tune_metric_selection=metric,
        tuning_times=tuning_times,
        no_lambda=no_lambda,
        model_name=model_name
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+model_name+'/EPowerConsumption/ht_preds.txt',
               ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+model_name+'/EPowerConsumption/ht_hs_preds.txt',
               ht_hs_preds)
    if no_lambda:
        np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+model_name+'/EPowerConsumption/kde_no_hs_preds.txt',
               kde_preds)
    else:
        np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/' + model_name + '/EPowerConsumption/kde_preds.txt',
        kde_preds)
    np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/'+model_name+'/EPowerConsumption/y_true.txt',
        y_true)

    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)
    mse_kde = mean_squared_error(y_true, kde_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"{model_name} MSE:      {mse_ht:.4f}")
    print(f"{model_name}+HS MSE:   {mse_ht_hs:.4f}")
    print(f"{model_name}+KDE MSE:     {mse_kde:.4f}")



def electric_power_comsumption_experiment_hs(path, model_name='ht', n_rows=1000, metric='mse', parameters = {
        'lambda_list': [1, 5, 10],
        'h_list': [0.1, 1, 10],
        'window_size_list': [50, 100, 200]
    }, tuning_times=4):


    X, y = load_EPowerConsumption(path, n_rows)


    ht_preds, ht_hs_preds, kde_preds, y_true = tune_and_train(
        parameters=parameters,
        X=X,
        y=y,
        model_name=model_name,
        tune_metric_selection=metric,
        tuning_times=tuning_times
    )

    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/'+model_name+'/EPowerConsumption/ht_preds.txt',
               ht_preds)
    np.savetxt('/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/'+model_name+'/EPowerConsumption/ht_hs_preds.txt',
               ht_hs_preds)
    np.savetxt(
        '/Users/pantia-marinaalchirch/PycharmProjects/ht_library_my_version/results/hs/'+model_name+'/EPowerConsumption/y_true.txt',
        y_true)

    mse_ht = mean_squared_error(y_true, ht_preds)
    mse_ht_hs = mean_squared_error(y_true, ht_hs_preds)

    print("\nFinal Mean Squared Errors:")
    print(f"{model_name} MSE:      {mse_ht:.4f}")
    print(f"{model_name}+HS MSE:   {mse_ht_hs:.4f}")

