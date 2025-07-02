from Code.bin_structure import CollectionOfBins


class KDEModel:

    def __init__(self, X, y, model, parameters, kde_type='labeled', range_of_bin=1, predictions=[]):
        self.model = model
        self.X = X
        self.y = y
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters
        if kde_type == 'labeled':
            self.unique_labels = sorted(set(y))
            self.weights = {label: 0 for label in self.unique_labels}
        elif kde_type == 'binned':
            self.bins_structure = CollectionOfBins(range_of_bin=range_of_bin)
            self.bins = self.bins_structure.organiseLabelsIntoBins(y)
        else:
            raise TypeError("KDE can either be 'labeled' or 'binned'")
        self.predictions = predictions

    def get_model_parameters(self):
        return self.parameters

    def get_kde_type(self):
        return self.kde_type

    def get_model_predictions(self):
        return self.predictions
