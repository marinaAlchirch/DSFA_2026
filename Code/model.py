import copy

import numpy as np
import pandas as pd

from Code.bin_structure import CollectionOfBins
from Code.kernels import CustomGaussianKernel


class IncrKDEModel:


    def __init__(self, y, model, parameters, fall_back=False):
        self.model = model
        self.y = y
        self.fall_back = fall_back

        if parameters is None:
            self.parameters = {
                'lambda_selected': 0.0,
                'h': 1,
                'window_size': 1,
                'kernel': CustomGaussianKernel(sigma=1.0),
                'kde_type': 'labeled',
                'range_of_bin': 0
            }
        else:
            self.parameters = copy.deepcopy(parameters)

        # Store parameters separately as immutable configuration
        self.lambda_selected = self.parameters['lambda_selected']
        self.h = self.parameters['h']
        self.window_size = self.parameters['window_size']
        self.kernel = self.parameters['kernel']
        self.kde_type = self.parameters['kde_type']
        self.range_of_bin = self.parameters['range_of_bin']

        self.predictions = []

        if self.kde_type == 'labeled':
            self.unique_labels = sorted(set(y))
            self.weights = {label: 0 for label in self.unique_labels}

        elif self.kde_type == 'binned':
            # Do not mutate parameters, even if range_of_bin is 0
            self.range_of_bin = self.parameters['range_of_bin'] if self.parameters['range_of_bin'] > 0 else 1
            self.bins_structure = CollectionOfBins(range_of_bin=self.range_of_bin)
            self.bins = self.bins_structure.organiseLabelsIntoBins(y)

        else:
            if not self.fall_back:
                raise TypeError("KDE type must be either 'labeled' or 'binned'")

    def set_parameters(self, new_parameters):

        self.parameters = copy.deepcopy(new_parameters)

        # Update core fields
        self.lambda_selected = new_parameters['lambda_selected']
        self.h = new_parameters['h']
        self.window_size = new_parameters['window_size']
        self.kernel = new_parameters['kernel']
        self.kde_type = new_parameters['kde_type']
        self.range_of_bin = new_parameters['range_of_bin']

        # Reset weights/bins depending on type
        if self.kde_type == 'labeled':
            self.unique_labels = sorted(set(self.y))
            self.weights = {label: 0 for label in self.unique_labels}
        elif self.kde_type == 'binned':
            if self.range_of_bin == 0:
                self.range_of_bin = 1
            self.bins_structure = CollectionOfBins(range_of_bin=self.range_of_bin)
            self.bins = self.bins_structure.organiseLabelsIntoBins(self.y)
        else:
            if not self.fall_back:
                raise TypeError("KDE type must be either 'labeled' or 'binned'")

    def predict_then_fit(self, X, y):
        instances_seen = 0
        if not self.fall_back:
            if self.get_kde_type() == 'binned':
                if not hasattr(self, "bins_structure") or self.bins_structure is None:
                    range_of_bin = self.range_of_bin if self.range_of_bin > 0 else 1
                    self.bins_structure = CollectionOfBins(range_of_bin=range_of_bin)
                    self.bins = self.bins_structure.organiseLabelsIntoBins(self.y)

                for label in y:
                    bin_of_last_label = self.bins_structure.findBinOflabel(label, self.bins)
                    for bin in self.bins:
                        bin.weight = self.weight_update(instances_seen, bin.weight, bin.lower_bound,
                                                           bin_of_last_label.lower_bound, self.parameters['kernel'], self.parameters['h'])

                    instances_seen += 1  # Increment counter

                # Normalize weights
                normalized_incremental_bins = self.bins.copy()
                for bin in normalized_incremental_bins:
                    bin.weight = 1 / (bin.weight + 1e-6)

                for instance_idx in range(len(X)):

                    x_instance = X[instance_idx]
                    if isinstance(x_instance, pd.Series) or isinstance(x_instance, pd.DataFrame):
                        x_instance = x_instance.values
                    if isinstance(x_instance, np.ndarray):
                        if x_instance.ndim == 1:
                            x_instance = x_instance.reshape(1, -1)

                    # Predict
                    if self.parameters['lambda_selected'] == 0.0:
                        prediction = self.model.predict(x_instance)[0]

                    else :
                        lam = self.parameters['lambda_selected']
                        prediction = self.model.predictHSNew(x_instance, lambda_selected=lam)[0]

                    self.predictions.append(prediction)

                    # Partial fit
                    target_value = y.iloc[instance_idx] if isinstance(y, pd.Series) else y[instance_idx]
                    bin_of_label = self.bins_structure.findBinOflabel(target_value,
                                                                      normalized_incremental_bins)


                    self.model.partial_fit(X=x_instance, y=[target_value],
                                                  sample_weight=[bin_of_label.weight])


            else:
                for label_idx in range(len(y)):
                    label = y.iloc[label_idx] if isinstance(y, pd.Series) else y[label_idx]

                    for label_key in self.weights.keys():
                        self.weights[label_key] = self.weight_update(instances_seen, self.weights[label_key], label_key,
                                                           label, self.parameters['kernel'], self.parameters['h'])

                    instances_seen += 1  # Increment counter

                normalized_incremental_weights = self.weights.copy()
                for label in normalized_incremental_weights.keys():
                    normalized_incremental_weights[label] = 1 / (normalized_incremental_weights[label] + 1e-6)

                for instance_idx in range(len(X)):

                    x_instance = X[instance_idx]
                    if isinstance(x_instance, pd.Series) or isinstance(x_instance, pd.DataFrame):
                        x_instance = x_instance.values
                    if isinstance(x_instance, np.ndarray):
                        if x_instance.ndim == 1:
                            x_instance = x_instance.reshape(1, -1)

                    # Predict
                    if self.parameters['lambda_selected'] == 0.0:
                        prediction = self.model.predict(x_instance)[0]

                    else:
                        lam = self.parameters['lambda_selected']
                        prediction = self.model.predictHSNew(x_instance, lambda_selected=lam)[0]

                    self.predictions.append(prediction)

                    # Train
                    target_value = y.iloc[instance_idx] if isinstance(y, pd.Series) else y[instance_idx]
                    self.model.partial_fit(X=x_instance, y=[target_value], sample_weight=[normalized_incremental_weights[target_value]])
        else:
            # ht = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
            # ht_no_weight = HoeffdingTreeRegressor(leaf_prediction="perceptron", random_state=42)
            # ht_preds = []
            # ht_no_weight_preds = []
            for instance_idx in range(len(X)):

                x_instance = X[instance_idx]
                if isinstance(x_instance, pd.Series) or isinstance(x_instance, pd.DataFrame):
                    x_instance = x_instance.values
                if isinstance(x_instance, np.ndarray):
                    if x_instance.ndim == 1:
                        x_instance = x_instance.reshape(1, -1)

                # Predict
                prediction = self.model.predict(x_instance)[0]
                self.predictions.append(prediction)

                # ht_pred = ht.predict(x_instance)[0]
                # ht_preds.append(ht_pred)
                #
                # ht_no_weight_pred = ht_no_weight.predict(x_instance)[0]
                # ht_no_weight_preds.append(ht_no_weight_pred)

                # Train
                target_value = y.iloc[instance_idx] if isinstance(y, pd.Series) else y[instance_idx]
                self.model.partial_fit(X=x_instance, y=[target_value])
            #     ht.partial_fit(X=x_instance, y=[target_value], sample_weight=[1 / len(sorted(set(self.y)))])
            #     ht_no_weight.partial_fit(X=x_instance, y=[target_value])
            #
            # for pred in ht_preds:
            #     if pred not in self.predictions:
            #         print("HT and KDE model with no KDE just HT have a different prediction ")
            #         break
            # for pred in ht_no_weight_preds:
            #     if pred not in self.predictions:
            #         print("HT with no weight assigned and KDE model with no KDE just HT have a different prediction ")
            #         break


    def get_model_parameters(self):
        return self.parameters

    def get_kde_type(self):
        return self.parameters['kde_type']

    def get_model_predictions(self):
        return self.predictions

    def weight_update(self, num_instances, prev_weight, z_query, z, kernel, h=1.0):
        kernel_value = kernel.get_kernel_value((z_query - z) / h)
        new_weight = prev_weight + (1 / (num_instances + 1)) * ((kernel_value / h) - prev_weight)
        return new_weight
