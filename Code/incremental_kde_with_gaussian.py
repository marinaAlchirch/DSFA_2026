from src.trees import HoeffdingTreeRegressor
from src.lazy import KNNRegressor
from Code.model import KDEModel


def tune_model(models, x,  tune_metric_selection='mse'):

    for model in models.items():
        model.predict()



# Models to get tuned for different parameters used in kde
def create_models(lambda_list, h_list, window_size_list):
    models = []
    lambda_list = lambda_list if lambda_list else [0]
    h_list = h_list if h_list else [1]
    window_size_list = window_size_list if window_size_list else [1]

    for lambda_selected in lambda_list:
        for h in h_list:
            for window_size in window_size_list:
                parameters = {lambda_selected:lambda_selected, h:h, window_size:window_size}
                models.append(model=HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42), parameters=parameters, predictions=[])


    return models


# TODO : fix the below
def tune_and_train(parameters, tune_metric_selection='mse'):



    ht = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)

    lambda_list = parameters['lambda_list']
    h_list = parameters['h_list']
    window_size_list = parameters['window_size_list']

    models = create_models(lambda_list=lambda_list, h_list=h_list, window_size_list=window_size_list)



    #Initialize list predictions for a layer/target
    ht_predictions = []


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