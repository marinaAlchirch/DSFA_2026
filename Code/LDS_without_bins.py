import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.trees import HoeffdingTreeRegressor  # Standard model
from Code.weight_update import weight_update





def run_model(lambda_selected, h, window_size, x, target, kernel_choice='gaussian'):
    standard_tree = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
    standard_tree_lds = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)

    # Initialize list predictions for all models
    ht_predictions = []
    ht_hs_predictions = []
    ht_lds_predictions = []
    ht_lds_hs_predictions = []

    # Initialize lists of MAE for all models performances
    ht_mae_list = []
    ht_hs_mae_list = []
    ht_lds_mae_list = []
    ht_lds_hs_mae_list = []

    # Initialize lists of MSE for all models performances
    ht_mse_list = []
    ht_hs_mse_list = []
    ht_lds_mse_list = []
    ht_lds_hs_mse_list = []

    # Initialize lists of R^2 for all models performances
    ht_r2_list = []
    ht_hs_r2_list = []
    ht_lds_r2_list = []
    ht_lds_hs_r2_list = []

    actual_labels = []
    actual_lds_labels = []

    unique_labels = sorted(set(target))
    weights = {label: 0 for label in unique_labels}


    count_window = 0
    train_x = []
    train_y = []

    for i in range(len(x)):

        if str(type(x)) == '<class \'pandas.core.frame.DataFrame\'>':
            instance_features = x.iloc[i:i + 1].values
        elif str(type(x)) == '<class \'numpy.ndarray\'>':
            instance_features = x[i].reshape(1, -1)
        else:
            raise TypeError('x type must either be pandas.core.frame.DataFrame or numpy.ndarray')

        actual_value = target[i]
        actual_labels.append(actual_value)

        # Predict Ht and HT + HS
        ht_pred = standard_tree.predict(instance_features)[0]
        ht_predictions.append(ht_pred)

        # Calculate MAE, MSE and R^2 for HT
        # MAE
        ht_mae = mean_absolute_error(actual_labels, ht_predictions)
        ht_mae_list.append(ht_mae)
        # MSE
        ht_mse = mean_squared_error(actual_labels, ht_predictions)
        ht_mse_list.append(ht_mse)
        # R^2 : Calculate only when len(predictions)>= 2 (R^2 nees variance so, it needs at least 2 samples to be correctly computed
        if len(ht_predictions)>=2:
            ht_r2 = r2_score(actual_labels, ht_predictions)
            ht_r2_list.append(ht_r2)

        # Calculate MAE, MSE and R^2 for HT + HS
        # MAE
        ht_hs_mae = mean_absolute_error(actual_labels, ht_hs_predictions)
        ht_hs_mae_list.append(ht_hs_mae)
        # MSE
        ht_hs_mse = mean_squared_error(actual_labels, ht_hs_predictions)
        ht_hs_mse_list.append(ht_hs_mse)
        # R^2 : Calculate only when len(predictions)>= 2 (R^2 nees variance so, it needs at least 2 samples to be correctly computed
        if len(ht_hs_predictions) >= 2:
            ht_hs_r2 = r2_score(actual_labels, ht_hs_predictions)
            ht_hs_r2_list.append(ht_hs_r2)


        ht_hs_pred = standard_tree.predictHSNew(instance_features, lambda_selected=lambda_selected)[0]
        ht_hs_predictions.append(ht_hs_pred)

        count_window += 1
        train_x.append(instance_features)

        train_y.append(actual_value)

        instances_seen = 0

        if count_window == window_size or count_window >= len(x) - i:

            sigma = np.std(train_y)
            for label_idx in range(len(train_y)):

                for label_key in weights.keys():
                    weights[label_key] = weight_update(instances_seen, weights[label_key], label_key,
                                                       train_y[label_idx], h, kernel_choice, sigma)

                instances_seen += 1  # Increment counter

            normalized_incremental_weights = weights.copy()
            for label in normalized_incremental_weights.keys():
                normalized_incremental_weights[label] = 1 / (normalized_incremental_weights[label] + 1e-6)



            for instance_idx in range(len(train_x)):

                actual_lds_labels.append(train_y[instance_idx])

                # Predict
                standard_pred_lds = standard_tree_lds.predict(train_x[instance_idx])[0]
                ht_lds_predictions.append(standard_pred_lds)

                hs_dir_pred = standard_tree_lds.predictHSNew(train_x[instance_idx], lambda_selected=lambda_selected)[0]
                ht_lds_hs_predictions.append(hs_dir_pred)

                # Calculate MAE, MSE and R^2 for HT + LDS
                # MAE
                ht_lds_mae = mean_absolute_error(actual_lds_labels, ht_lds_predictions)
                ht_lds_mae_list.append(ht_lds_mae)
                # MSE
                ht_lds_mse = mean_squared_error(actual_lds_labels, ht_lds_predictions)
                ht_lds_mse_list.append(ht_lds_mse)
                # R^2 : Calculate only when len(predictions)>= 2 (R^2 nees variance so, it needs at least 2 samples to be correctly computed
                if len(ht_lds_predictions) >= 2:
                    ht_lds_r2 = r2_score(actual_lds_labels, ht_lds_predictions)
                    ht_lds_r2_list.append(ht_lds_r2)

                # Calculate MAE, MSE and R^2 for HT + LDS + HS
                # MAE
                ht_lds_hs_mae = mean_absolute_error(actual_lds_labels, ht_lds_hs_predictions)
                ht_lds_hs_mae_list.append(ht_lds_hs_mae)
                # MSE
                ht_lds_hs_mse = mean_squared_error(actual_lds_labels, ht_lds_hs_predictions)
                ht_lds_hs_mse_list.append(ht_lds_hs_mse)
                # R^2 : Calculate only when len(predictions)>= 2 (R^2 nees variance so, it needs at least 2 samples to be correctly computed
                if len(ht_lds_hs_predictions) >= 2:
                    ht_lds_hs_r2 = r2_score(actual_lds_labels, ht_lds_hs_predictions)
                    ht_lds_hs_r2_list.append(ht_lds_hs_r2)

                # Train
                standard_tree_lds.partial_fit(train_x[instance_idx], [train_y[instance_idx]],
                                              sample_weight=[normalized_incremental_weights[train_y[instance_idx]]])

            train_x = []
            train_y = []

            count_window = 0

        standard_tree.partial_fit(instance_features, [actual_value])

    # Compute MAE, MSE and R^2 of all models at the end of training

    # For HT:
    ht_mae = mean_absolute_error(actual_labels, ht_predictions)
    ht_mse = mean_squared_error(actual_labels, ht_predictions)
    ht_r2 = r2_score(actual_labels, ht_predictions)

    # For HT + HS:
    ht_hs_mae = mean_absolute_error(actual_labels, ht_hs_predictions)
    ht_hs_mse = mean_squared_error(actual_labels, ht_hs_predictions)
    ht_hs_r2 = r2_score(actual_labels, ht_hs_predictions)

    # For HT + LDS:
    ht_lds_mae = mean_absolute_error(actual_labels, ht_lds_predictions)
    ht_lds_mse = mean_squared_error(actual_labels, ht_lds_predictions)
    ht_lds_r2 = r2_score(actual_labels, ht_lds_predictions)

    # For HT + LDS + HS:
    ht_lds_hs_mae = mean_absolute_error(actual_labels, ht_lds_hs_predictions)
    ht_lds_hs_mse = mean_squared_error(actual_labels, ht_lds_hs_predictions)
    ht_lds_hs_r2 = r2_score(actual_labels, ht_lds_hs_predictions)

    # Printout MAE, MSE and R^2 of all models at the end of training
    print(f"HT: MAE={ht_mae:.4f}, MSE={ht_mse:.4f}, R2={ht_r2:.4f}")
    print(f"HT + HS : MAE={ht_hs_mae:.4f}, MSE={ht_hs_mse:.4f}, R2={ht_hs_r2:.4f}")

    print(f"HT + LDS with window size :{window_size}, MAE={ht_lds_mae:.4f}, MSE={ht_lds_mse:.4f}, R2={ht_lds_r2:.4f}")
    print(
        f"HT + LDS + HS with window size :{window_size}, MAE={ht_lds_hs_mae:.4f}, MSE={ht_lds_hs_mse:.4f}, R2={ht_lds_hs_r2:.4f}")

    model_score_list = {}  # Key : str-> model_scorename, Value: list : model_score_name
    model_score_list['ht_mae_list'] = ht_mae_list
    model_score_list['ht_mse_list'] = ht_mse_list
    model_score_list['ht_r2_list'] = ht_r2_list

    model_score_list['ht_hs_mae_list'] = ht_hs_mae_list
    model_score_list['ht_hs_mse_list'] = ht_hs_mse_list
    model_score_list['ht_hs_r2_list'] = ht_hs_r2_list

    model_score_list['ht_lds_mae_list'] = ht_lds_mae_list
    model_score_list['ht_lds_mse_list'] = ht_lds_mse_list
    model_score_list['ht_lds_r2_list'] = ht_lds_r2_list

    model_score_list['ht_lds_hs_mae_list'] = ht_lds_hs_mae_list
    model_score_list['ht_lds_hs_mse_list'] = ht_lds_hs_mse_list
    model_score_list['ht_lds_hs_r2_list'] = ht_lds_hs_r2_list

    return model_score_list


def tune_model_on_mae(lambda_list, h_list, window_size_list, x, target, kernel_choice='gaussian'):
    ht_mae_for_best_ht_lds_hs = np.inf
    ht_hs_mae_for_best_ht_lds_hs = np.inf
    ht_lds_mae_for_best_ht_lds_hs = np.inf
    best_ht_lds_hs_mae = np.inf

    best_lambda = lambda_list[0]
    best_h = h_list[0]
    best_window_size = window_size_list[0]

    for lambda_selected in lambda_list:
        print(f"For lambda  : {lambda_selected}")
        for h in h_list:
            for window_size in window_size_list:
                standard_tree = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
                standard_tree_lds = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)

                # Initialize list predictions for all models
                ht_predictions = []
                ht_hs_predictions = []
                ht_lds_predictions = []
                ht_lds_hs_predictions = []


                actual_labels = []
                actual_lds_labels = []

                unique_labels = sorted(set(target))
                weights = {label: 0 for label in unique_labels}

                count_window = 0
                train_x = []
                train_y = []

                for i in range(len(x)):

                    if str(type(x)) == '<class \'pandas.core.frame.DataFrame\'>':
                        instance_features = x.iloc[i:i + 1].values
                    elif str(type(x)) == '<class \'numpy.ndarray\'>':
                        instance_features = x[i].reshape(1, -1)
                    else:
                        raise TypeError('x type must either be pandas.core.frame.DataFrame or numpy.ndarray')

                    actual_value = target[i]
                    actual_labels.append(actual_value)

                    # Predict HT
                    ht_pred = standard_tree.predict(instance_features)[0]
                    ht_predictions.append(ht_pred)

                    # Predict HT + HS
                    ht_hs_pred = standard_tree.predictHSNew(instance_features, lambda_selected=lambda_selected)[0]
                    ht_hs_predictions.append(ht_hs_pred)

                    count_window += 1
                    train_x.append(instance_features)

                    train_y.append(actual_value)

                    instances_seen = 0

                    if count_window == window_size or count_window >= len(x) - i:

                        sigma = np.std(train_y)
                        for label_idx in range(len(train_y)):

                            for label_key in weights.keys():
                                weights[label_key] = weight_update(instances_seen, weights[label_key], label_key,
                                                                   train_y[label_idx], h, kernel_choice, sigma)

                            instances_seen += 1  # Increment counter

                        normalized_incremental_weights = weights.copy()
                        for label in normalized_incremental_weights.keys():
                            normalized_incremental_weights[label] = 1 / (normalized_incremental_weights[label] + 1e-6)

                        for instance_idx in range(len(train_x)):

                            actual_lds_labels.append(train_y[instance_idx])

                            # Predict HT + LDS and HT + LDS + HS
                            standard_pred_lds = standard_tree_lds.predict(train_x[instance_idx])[0]
                            ht_lds_predictions.append(standard_pred_lds)

                            hs_dir_pred = standard_tree_lds.predictHSNew(train_x[instance_idx], lambda_selected=lambda_selected)[0]
                            ht_lds_hs_predictions.append(hs_dir_pred)

                            # Train
                            standard_tree_lds.partial_fit(train_x[instance_idx], [train_y[instance_idx]], sample_weight=[normalized_incremental_weights[train_y[instance_idx]]])

                        train_x = []
                        train_y = []

                        count_window = 0

                    standard_tree.partial_fit(instance_features, [actual_value])

                # Compute MAE of all models at the end of training

                # For HT:
                ht_mae = mean_absolute_error(actual_labels, ht_predictions)

                # For HT + HS:
                ht_hs_mae = mean_absolute_error(actual_labels, ht_hs_predictions)

                # For HT + LDS:
                ht_lds_mae = mean_absolute_error(actual_labels, ht_lds_predictions)

                # For HT + LDS + HS:
                ht_lds_hs_mae = mean_absolute_error(actual_labels, ht_lds_hs_predictions)

                if best_ht_lds_hs_mae >= ht_lds_hs_mae and ht_lds_hs_mae <= ht_mae:
                    best_lambda = lambda_selected
                    best_h = h
                    best_window_size = window_size
                    ht_mae_for_best_ht_lds_hs = ht_mae
                    ht_hs_mae_for_best_ht_lds_hs = ht_hs_mae
                    ht_lds_mae_for_best_ht_lds_hs = ht_lds_mae
                    best_ht_lds_hs_mae = ht_lds_hs_mae

    # Printout best hyperparameters
    print("Best hyperparameters are :")
    print(f"Lambda : {best_lambda}")
    print(f"Bandwidth : {best_h}")
    print(f"Window size : {best_window_size}")

    hyperpameters = {}
    hyperpameters['best_lambda'] = best_lambda
    hyperpameters['best_h'] = best_h
    hyperpameters['best_window_size'] = best_window_size

    # Printout Best MAE value of all models at the end of tuning
    print(f"HT: MAE={ht_mae_for_best_ht_lds_hs:.4f}")
    print(f"HT + HS : MAE={ht_hs_mae_for_best_ht_lds_hs:.4f}")

    print(f"HT + LDS : MAE={ht_lds_mae_for_best_ht_lds_hs:.4f}")
    print(f"HT + LDS + HS : MAE={best_ht_lds_hs_mae:.4f}")

    # Return dictionary that contains the best hyperparameters found from tuning
    return hyperpameters


def tune_model_on_mse(lambda_list, h_list, window_size_list, x, target, kernel_choice='gaussian'):
    ht_mse_for_best_ht_lds_hs = np.inf
    ht_hs_mse_for_best_ht_lds_hs = np.inf
    ht_lds_mse_for_best_ht_lds_hs = np.inf
    best_ht_lds_hs_mse = np.inf

    best_lambda = lambda_list[0]
    best_h = h_list[0]
    best_window_size = window_size_list[0]

    for lambda_selected in lambda_list:
        print(f"For lambda  : {lambda_selected}")
        for h in h_list:
            for window_size in window_size_list:
                standard_tree = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
                standard_tree_lds = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)

                # Initialize list predictions for all models
                ht_predictions = []
                ht_hs_predictions = []
                ht_lds_predictions = []
                ht_lds_hs_predictions = []

                actual_labels = []
                actual_lds_labels = []

                unique_labels = sorted(set(target))
                weights = {label: 0 for label in unique_labels}

                count_window = 0
                train_x = []
                train_y = []

                for i in range(len(x)):

                    if str(type(x)) == '<class \'pandas.core.frame.DataFrame\'>':
                        instance_features = x.iloc[i:i + 1].values
                    elif str(type(x)) == '<class \'numpy.ndarray\'>':
                        instance_features = x[i].reshape(1, -1)
                    else:
                        raise TypeError('x type must either be pandas.core.frame.DataFrame or numpy.ndarray')

                    actual_value = target[i]
                    actual_labels.append(actual_value)

                    # Predict HT
                    ht_pred = standard_tree.predict(instance_features)[0]
                    ht_predictions.append(ht_pred)

                    # Predict HT + HS
                    ht_hs_pred = standard_tree.predictHSNew(instance_features, lambda_selected=lambda_selected)[0]
                    ht_hs_predictions.append(ht_hs_pred)

                    count_window += 1
                    train_x.append(instance_features)

                    train_y.append(actual_value)

                    instances_seen = 0

                    if count_window == window_size or count_window >= len(x) - i:

                        sigma = np.std(train_y)
                        for label_idx in range(len(train_y)):

                            for label_key in weights.keys():
                                weights[label_key] = weight_update(instances_seen, weights[label_key], label_key,
                                                                   train_y[label_idx], h, kernel_choice, sigma)

                            instances_seen += 1  # Increment counter

                        normalized_incremental_weights = weights.copy()
                        for label in normalized_incremental_weights.keys():
                            normalized_incremental_weights[label] = 1 / (normalized_incremental_weights[label] + 1e-6)

                        for instance_idx in range(len(train_x)):
                            actual_lds_labels.append(train_y[instance_idx])

                            # Predict HT + LDS and HT + LDS + HS
                            standard_pred_lds = standard_tree_lds.predict(train_x[instance_idx])[0]
                            ht_lds_predictions.append(standard_pred_lds)

                            hs_dir_pred = \
                            standard_tree_lds.predictHSNew(train_x[instance_idx], lambda_selected=lambda_selected)[0]
                            ht_lds_hs_predictions.append(hs_dir_pred)

                            # Train
                            standard_tree_lds.partial_fit(train_x[instance_idx], [train_y[instance_idx]], sample_weight=[normalized_incremental_weights[train_y[instance_idx]]])

                        train_x = []
                        train_y = []

                        count_window = 0

                    standard_tree.partial_fit(instance_features, [actual_value])

                # Compute MSE of all models at the end of training
                # For HT:
                ht_mse = mean_squared_error(actual_labels, ht_predictions)

                # For HT + HS:
                ht_hs_mse = mean_squared_error(actual_labels, ht_hs_predictions)

                # For HT + LDS:
                ht_lds_mse = mean_squared_error(actual_labels, ht_lds_predictions)

                # For HT + LDS + HS:
                ht_lds_hs_mse = mean_squared_error(actual_labels, ht_lds_hs_predictions)

                if best_ht_lds_hs_mse >= ht_lds_hs_mse and ht_lds_hs_mse <= ht_mse:
                    best_lambda = lambda_selected
                    best_h = h
                    best_window_size = window_size
                    ht_mse_for_best_ht_lds_hs = ht_mse
                    ht_hs_mse_for_best_ht_lds_hs = ht_hs_mse
                    ht_lds_mse_for_best_ht_lds_hs = ht_lds_mse
                    best_ht_lds_hs_mse = ht_lds_hs_mse

    # Printout best hyperparameters
    print("Best hyperparameters are :")
    print(f"Lambda : {best_lambda}")
    print(f"Bandwidth : {best_h}")
    print(f"Window size : {best_window_size}")

    hyperpameters = {}
    hyperpameters['best_lambda'] = best_lambda
    hyperpameters['best_h'] = best_h
    hyperpameters['best_window_size'] = best_window_size

    # Printout Best MAE value of all models at the end of tuning
    print(f"HT: MSE={ht_mse_for_best_ht_lds_hs:.4f}")
    print(f"HT + HS : MSE={ht_hs_mse_for_best_ht_lds_hs:.4f}")

    print(f"HT + LDS : MSE={ht_lds_mse_for_best_ht_lds_hs:.4f}")
    print(f"HT + LDS + HS : MSE={best_ht_lds_hs_mse:.4f}")

    # Return dictionary that contains the best hyperparameters found from tuning
    return hyperpameters

def tune_model_on_r2(lambda_list, h_list, window_size_list, x, target, kernel_choice='gaussian'):
    ht_r2_for_best_ht_lds_hs = 0
    ht_hs_r2_for_best_ht_lds_hs = 0
    ht_lds_r2_for_best_ht_lds_hs = 0
    best_ht_lds_hs_r2 = 0

    best_lambda = lambda_list[0]
    best_h = h_list[0]
    best_window_size = window_size_list[0]

    for lambda_selected in lambda_list:
        print(f"For lambda  : {lambda_selected}")
        for h in h_list:
            for window_size in window_size_list:
                standard_tree = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
                standard_tree_lds = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)

                # Initialize list predictions for all models
                ht_predictions = []
                ht_hs_predictions = []
                ht_lds_predictions = []
                ht_lds_hs_predictions = []

                actual_labels = []
                actual_lds_labels = []

                unique_labels = sorted(set(target))
                weights = {label: 0 for label in unique_labels}

                count_window = 0
                train_x = []
                train_y = []

                for i in range(len(x)):

                    if str(type(x)) == '<class \'pandas.core.frame.DataFrame\'>':
                        instance_features = x.iloc[i:i + 1].values
                    elif str(type(x)) == '<class \'numpy.ndarray\'>':
                        instance_features = x[i].reshape(1, -1)
                    else:
                        raise TypeError('x type must either be pandas.core.frame.DataFrame or numpy.ndarray')

                    actual_value = target[i]
                    actual_labels.append(actual_value)

                    # Predict HT
                    ht_pred = standard_tree.predict(instance_features)[0]
                    ht_predictions.append(ht_pred)

                    # Predict HT + HS
                    ht_hs_pred = standard_tree.predictHSNew(instance_features, lambda_selected=lambda_selected)[0]
                    ht_hs_predictions.append(ht_hs_pred)

                    count_window += 1
                    train_x.append(instance_features)

                    train_y.append(actual_value)

                    instances_seen = 0

                    if count_window == window_size or count_window >= len(x) - i:
                        sigma = np.std(train_y)
                        for label_idx in range(len(train_y)):

                            for label_key in weights.keys():
                                weights[label_key] = weight_update(instances_seen, weights[label_key], label_key,
                                                                   train_y[label_idx], h, kernel_choice, sigma)

                            instances_seen += 1  # Increment counter

                        normalized_incremental_weights = weights.copy()
                        for label in normalized_incremental_weights.keys():
                            normalized_incremental_weights[label] = 1 / (normalized_incremental_weights[label] + 1e-6)

                        for instance_idx in range(len(train_x)):
                            actual_lds_labels.append(train_y[instance_idx])

                            # Predict HT + LDS and HT + LDS + HS
                            standard_pred_lds = standard_tree_lds.predict(train_x[instance_idx])[0]
                            ht_lds_predictions.append(standard_pred_lds)

                            hs_dir_pred = \
                            standard_tree_lds.predictHSNew(train_x[instance_idx], lambda_selected=lambda_selected)[0]
                            ht_lds_hs_predictions.append(hs_dir_pred)

                            # Train
                            standard_tree_lds.partial_fit(train_x[instance_idx], [train_y[instance_idx]], sample_weight=[normalized_incremental_weights[train_y[instance_idx]]])

                        train_x = []
                        train_y = []

                        count_window = 0

                    standard_tree.partial_fit(instance_features, [actual_value])

                # Compute MAE of all models at the end of training

                # For HT:
                ht_r2 = r2_score(actual_labels, ht_predictions)

                # For HT + HS:
                ht_hs_r2 = r2_score(actual_labels, ht_hs_predictions)

                # For HT + LDS:
                ht_lds_r2 = r2_score(actual_labels, ht_lds_predictions)

                # For HT + LDS + HS:
                ht_lds_hs_r2 = r2_score(actual_labels, ht_lds_hs_predictions)

                if best_ht_lds_hs_r2 <= ht_lds_hs_r2 and ht_lds_hs_r2 >= ht_r2:
                    best_lambda = lambda_selected
                    best_h = h
                    best_window_size = window_size
                    ht_r2_for_best_ht_lds_hs = ht_r2
                    ht_hs_r2_for_best_ht_lds_hs = ht_hs_r2
                    ht_lds_r2_for_best_ht_lds_hs = ht_lds_r2
                    best_ht_lds_hs_r2 = ht_lds_hs_r2

    # Printout best hyperparameters
    print("Best hyperparameters are :")
    print(f"Lambda : {best_lambda}")
    print(f"Bandwidth : {best_h}")
    print(f"Window size : {best_window_size}")

    hyperpameters = {}
    hyperpameters['best_lambda'] = best_lambda
    hyperpameters['best_h'] = best_h
    hyperpameters['best_window_size'] = best_window_size

    # Printout Best R^2 value of all models at the end of tuning
    print(f"HT: R^2={ht_r2_for_best_ht_lds_hs:.4f}")
    print(f"HT + HS : R^2={ht_hs_r2_for_best_ht_lds_hs:.4f}")

    print(f"HT + LDS : R^2={ht_lds_r2_for_best_ht_lds_hs:.4f}")
    print(f"HT + LDS + HS : R^2={best_ht_lds_hs_r2:.4f}")

    # Return dictionary that contains the best hyperparameters found from tuning
    return hyperpameters


# This method is only for film thickness in semiconductors data
def run_tune_model(x, target, kernel_choice, tune_metric_selection='mse'):

    best_parameters = {}

    ht = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
    ht_lds = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)

    #Initialize list predictions for a layer/target
    ht_predictions = []
    ht_hs_predictions = []
    ht_lds_predictions = []
    ht_lds_hs_predictions = []

    actual_labels = []

    unique_labels = sorted(set(target))
    weights = {label: 0 for label in unique_labels}


    # Track the number of instances seen

    count_window = 0
    train_x = []

    train_y = []

    actual_lds_labels = []

    ht_mae_list = []
    ht_mse_list = []
    ht_r2_list = []

    ht_hs_mae_list = []
    ht_hs_mse_list = []
    ht_hs_r2_list = []

    ht_dir_mae_list = []
    ht_dir_mse_list = []
    ht_dir_r2_list = []

    ht_hs_dir_mae_list = []
    ht_hs_dir_mse_list = []
    ht_hs_dir_r2_list = []

    counter_examples = 0
    print(str(type(x)))

    for i in range(len(x)):

        # Perform tuning on bandwidth and window size for x[0:1K], x[5K:6K], x[10K:11K], x[15K:16K]
        if (counter_examples % 1000 == 0) and (counter_examples < 1000 or (counter_examples > 5000 and counter_examples <= 6000) or (counter_examples > 10000 and counter_examples <= 11000) or (counter_examples > 15000 and counter_examples <= 16000)):
            if tune_metric_selection == 'mse':
                best_parameters = tune_model_on_mse(lambda_list=[1], h_list=[1, 10, 50, 100, 500, 1000], window_size_list=[10, 50, 100, 200], x=x[counter_examples:counter_examples+1000], target=target[counter_examples:counter_examples+1000], kernel_choice=kernel_choice)
            elif tune_metric_selection == 'mae':
                best_parameters = tune_model_on_mae(lambda_list=[1], h_list=[1, 10, 50, 100, 500, 1000], window_size_list=[10, 50, 100, 200], x=x[counter_examples:counter_examples+1000], target=target[counter_examples:counter_examples+1000], kernel_choice=kernel_choice)
            elif tune_metric_selection == 'r2':
                best_parameters = tune_model_on_r2(lambda_list=[1], h_list=[1, 10, 50, 100, 500, 1000], window_size_list=[10, 50, 100, 200], x=x[counter_examples:counter_examples+1000], target=target[counter_examples:counter_examples+1000], kernel_choice=kernel_choice)
            else:
                raise TypeError('Tuning is only available for mse, mae and r2 scores.')

        else:

            if str(type(x)) == '<class \'pandas.core.frame.DataFrame\'>':
                instance_features = x.iloc[i:i + 1].values
            elif str(type(x)) == '<class \'numpy.ndarray\'>':
                instance_features = x[i].reshape(1, -1)
            else:
                raise TypeError('x type must either be pandas.core.frame.DataFrame or numpy.ndarray')
            actual_value = target[i]
            # Predict
            ht_pred = ht.predict(instance_features)[0]
            ht_predictions.append(ht_pred)


            ht_hs_pred = ht.predictHSNew(instance_features, lambda_selected=best_parameters['best_lambda'])[0]
            ht_hs_predictions.append(ht_hs_pred)


            actual_labels.append(actual_value)


            #Hoeffding Tree : MAE, MSE and R^2
            mae_ht = mean_absolute_error(actual_labels, ht_predictions)
            ht_mae_list.append(mae_ht)

            mse_ht = mean_squared_error(actual_labels, ht_predictions)
            ht_mse_list.append(mse_ht)

            r2_ht = r2_score(actual_labels, ht_predictions)
            ht_r2_list.append(r2_ht)

            #HT with HS : : MAE, MSE and R^2
            mae_ht_hs = mean_absolute_error(actual_labels, ht_hs_predictions)
            ht_hs_mae_list.append(mae_ht_hs)

            mse_ht_hs = mean_squared_error(actual_labels, ht_hs_predictions)
            ht_hs_mse_list.append(mse_ht_hs)

            r2_ht_hs = r2_score(actual_labels, ht_hs_predictions)
            ht_hs_r2_list.append(r2_ht_hs)


            count_window += 1

            train_x.append(instance_features)


            train_y.append(actual_value)

            instances_seen = 0


            if count_window == best_parameters['best_window_size'] or count_window >= len(x) - i:

                sigma = np.std(train_y)
                for label_idx in range(len(train_y)):

                    for label_key in weights.keys():
                        weights[label_key] = weight_update(instances_seen, weights[label_key], label_key,
                                                           train_y[label_idx], h, kernel_choice, sigma)

                    instances_seen += 1  # Increment counter

                normalized_incremental_weights = weights.copy()
                for label in normalized_incremental_weights.keys():
                    normalized_incremental_weights[label] = 1 / (normalized_incremental_weights[label] + 1e-6)




                for instance_idx in range(len(train_x)):

                    actual_lds_labels.append(train_y[instance_idx])

                    #Predict
                    ht_lds_pred = ht_lds.predict(train_x[instance_idx])[0]
                    ht_lds_predictions.append(ht_lds_pred)
                    ht_lds_hs_pred = ht_lds.predictHSNew(train_x[instance_idx], lambda_selected=best_parameters['best_lambda'])[0]
                    ht_lds_hs_predictions.append(ht_lds_hs_pred)

                    #Train
                    ht_lds.partial_fit(train_x[instance_idx], [train_y[instance_idx]], sample_weight=[normalized_incremental_weights[train_y[instance_idx]]])

                    #HT with DIR : MAE, MSE and R^2
                    mae_ht_dir = mean_absolute_error(actual_lds_labels, ht_lds_predictions)
                    ht_dir_mae_list.append(mae_ht_dir)

                    mse_ht_dir1 = mean_squared_error(actual_lds_labels, ht_lds_predictions)
                    ht_dir_mse_list.append(mse_ht_dir1)

                    r2_ht_dir = r2_score(actual_lds_labels, ht_lds_predictions)
                    ht_dir_r2_list.append(r2_ht_dir)


                    #HT with DIR and HS : MAE, MSE and R^2
                    mae_ht_hs_dir = mean_absolute_error(actual_lds_labels, ht_lds_hs_predictions)
                    ht_hs_dir_mae_list.append(mae_ht_hs_dir)

                    mse_ht_hs_dir1 = mean_squared_error(actual_lds_labels, ht_lds_hs_predictions)
                    ht_hs_dir_mse_list.append(mse_ht_hs_dir1)

                    r2_ht_hs_dir = r2_score(actual_lds_labels, ht_lds_hs_predictions)
                    ht_hs_dir_r2_list.append(r2_ht_hs_dir)



                train_x = []
                train_y = []


                count_window = 0


            ht.partial_fit(instance_features, [actual_value])
        counter_examples += 1

    # Evaluate all the models

    #For layer 1
    standard_mae = mean_absolute_error(actual_labels, ht_predictions)
    standard_mse = mean_squared_error(actual_labels, ht_predictions)
    standard_r2 = r2_score(actual_labels, ht_predictions)

    hs_mae = mean_absolute_error(actual_labels, ht_hs_predictions)
    hs_mse = mean_squared_error(actual_labels, ht_hs_predictions)
    hs_r2 = r2_score(actual_labels, ht_hs_predictions)

    standard_dir_mae = mean_absolute_error(actual_labels, ht_lds_predictions)
    standard_dir_mse = mean_squared_error(actual_labels, ht_lds_predictions)
    standard_dir_r2 = r2_score(actual_labels, ht_lds_predictions)

    hs_dir_mae = mean_absolute_error(actual_labels, ht_lds_hs_predictions)
    hs_dir_mse = mean_squared_error(actual_labels, ht_lds_hs_predictions)
    hs_dir_r2 = r2_score(actual_labels, ht_lds_hs_predictions)



    #Print MAE results for all models
    print(f"Standard Hoeffding Tree: MAE ={standard_mae:.4f}, MSE ={standard_mse:.4f}, R^2 ={standard_r2:.4f}")
    print(f"Hierarchical Shrinkage Tree: MAE ={hs_mae:.4f}, MSE ={hs_mse:.4f}, R^2 ={hs_r2:.4f}")
    print(f"Hoeffding Tree with LDS: MAE ={standard_dir_mae:.4f}, MSE ={standard_dir_mse:.4f}, R^2 ={standard_dir_r2:.4f}")
    print(f"Hierarchical Shrinkage Hoeffding Tree with LDS: MAE ={hs_dir_mae:.4f}, MSE ={hs_dir_mse:.4f}, R^2 ={hs_dir_r2:.4f}")

    return ht_hs_mae_list, ht_mse_list, ht_r2_list, ht_hs_mae_list, ht_hs_mse_list, ht_hs_r2_list, ht_dir_mae_list, ht_dir_mse_list, ht_dir_r2_list, ht_hs_dir_mae_list, ht_hs_dir_mse_list, ht_hs_dir_r2_list