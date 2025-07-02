import copy
import itertools
import numpy as np
from operator import attrgetter, itemgetter

from sklearn.tree import DecisionTreeClassifier

from src.utils import get_dimensions, normalize_values_in_dict, calculate_object_size
from src.core import BaseSKMObject, ClassifierMixin
from src.rules.base_rule import Rule

from ._split_criterion import GiniSplitCriterion
from ._split_criterion import InfoGainSplitCriterion
from ._split_criterion import HellingerDistanceCriterion
from ._attribute_test import NominalAttributeMultiwayTest
from ._nodes import Node
from ._nodes import FoundNode
from ._nodes import SplitNode
from ._nodes import ActiveLeaf, InactiveLeaf
from ._nodes import LearningNode
from ._nodes import ActiveLearningNodeMC
from ._nodes import ActiveLearningNodeNB
from ._nodes import ActiveLearningNodeNBA
from ._nodes import InactiveLearningNodeMC
from copy import deepcopy
from sklearn.base import BaseEstimator
from imodels.util import checks


import warnings


def HoeffdingTree(max_byte_size=33554432, memory_estimate_period=1000000, grace_period=200,
                  split_criterion='info_gain', split_confidence=0.0000001, tie_threshold=0.05,
                  binary_split=False, stop_mem_management=False, remove_poor_atts=False,
                  no_preprune=False, leaf_prediction='nba', nb_threshold=0,
                  nominal_attributes=None):     # pragma: no cover
    warnings.warn("'HoeffdingTree' has been renamed to 'HoeffdingTreeClassifier' in v0.5.0.\n"
                  "The old name will be removed in v0.7.0", category=FutureWarning)
    return HoeffdingTreeClassifierHS(max_byte_size=max_byte_size,
                                   memory_estimate_period=memory_estimate_period,
                                   grace_period=grace_period,
                                   split_criterion=split_criterion,
                                   split_confidence=split_confidence,
                                   tie_threshold=tie_threshold,
                                   binary_split=binary_split,
                                   stop_mem_management=stop_mem_management,
                                   remove_poor_atts=remove_poor_atts,
                                   no_preprune=no_preprune,
                                   leaf_prediction=leaf_prediction,
                                   nb_threshold=nb_threshold,
                                   nominal_attributes=nominal_attributes)


class HoeffdingTreeClassifierHS(BaseSKMObject, ClassifierMixin):
    """ Hoeffding Tree or Very Fast Decision Tree classifier.

    Parameters
    ----------
    max_byte_size: int (default=33554432)
        Maximum memory consumed by the tree.
    memory_estimate_period: int (default=1000000)
        Number of instances between memory consumption checks.
    grace_period: int (default=200)
        Number of instances a leaf should observe between split attempts.
    split_criterion: string (default='info_gain')
        | Split criterion to use.
        | 'gini' - Gini
        | 'info_gain' - Information Gain
        | 'hellinger' - Helinger Distance
    split_confidence: float (default=0.0000001)
        Allowed error in split decision, a value closer to 0 takes longer to decide.
    tie_threshold: float (default=0.05)
        Threshold below which a split will be forced to break ties.
    binary_split: boolean (default=False)
        If True, only allow binary splits.
    stop_mem_management: boolean (default=False)
        If True, stop growing as soon as memory limit is hit.
    remove_poor_atts: boolean (default=False)
        If True, disable poor attributes.
    no_preprune: boolean (default=False)
        If True, disable pre-pruning.
    leaf_prediction: string (default='nba')
        | Prediction mechanism used at leafs.
        | 'mc' - Majority Class
        | 'nb' - Naive Bayes
        | 'nba' - Naive Bayes Adaptive
    nb_threshold: int (default=0)
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    Notes
    -----
    A Hoeffding Tree [1]_ is an incremental, anytime decision tree induction algorithm that is
    capable of learning from massive data streams, assuming that the distribution generating
    examples does not change over time. Hoeffding trees exploit the fact that a small sample can
    often be enough to choose an optimal splitting attribute. This idea is supported mathematically
    by the Hoeffding bound, which quantifies the number of observations (in our case, examples)
    needed to estimate some statistics within a prescribed precision (in our case, the goodness of
    an attribute).

    A theoretically appealing feature of Hoeffding Trees not shared by other incremental decision
    tree learners is that it has sound guarantees of performance. Using the Hoeffding bound one
    can show that its output is asymptotically nearly identical to that of a non-incremental
    learner using infinitely many examples.

    Implementation based on MOA [2]_.

    References
    ----------
    .. [1] G. Hulten, L. Spencer, and P. Domingos. Mining time-changing data streams.
       In KDD’01, pages 97–106, San Francisco, CA, 2001. ACM Press.

    .. [2] Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer.
       MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604, 2010.

    Examples
    --------
    >>> # Imports
    >>> from src.data import SEAGenerator
    >>> from src.trees import HoeffdingTreeClassifier
    >>>
    >>> # Setting up a data stream
    >>> stream = SEAGenerator(random_state=1)
    >>>
    >>> # Setup Hoeffding Tree estimator
    >>> ht = HoeffdingTreeClassifier()
    >>>
    >>> # Setup variables to control loop and track performance
    >>> n_samples = 0
    >>> correct_cnt = 0
    >>> max_samples = 200
    >>>
    >>> # Train the estimator with the samples provided by the data stream
    >>> while n_samples < max_samples and stream.has_more_samples():
    >>>     X, y = stream.next_sample()
    >>>     y_pred = ht.predict(X)
    >>>     if y[0] == y_pred[0]:
    >>>         correct_cnt += 1
    >>>     ht = ht.partial_fit(X, y)
    >>>     n_samples += 1
    >>>
    >>> # Display results
    >>> print('{} samples analyzed.'.format(n_samples))
    >>> print('Hoeffding Tree accuracy: {}'.format(correct_cnt / n_samples))
    """

    _GINI_SPLIT = 'gini'
    _INFO_GAIN_SPLIT = 'info_gain'
    _HELLINGER = 'hellinger'
    _MAJORITY_CLASS = 'mc'
    _NAIVE_BAYES = 'nb'
    _NAIVE_BAYES_ADAPTIVE = 'nba'

    # ====================================
    # == Hoeffding Tree implementation ===
    # ====================================
    def __init__(self,
                 max_byte_size=33554432,
                 memory_estimate_period=1000000,
                 grace_period=200,
                 split_criterion='info_gain',
                 split_confidence=0.0000001,
                 tie_threshold=0.05,
                 binary_split=False,
                 stop_mem_management=False,
                 remove_poor_atts=False,
                 no_preprune=False,
                 leaf_prediction='nba',
                 nb_threshold=0,
                 nominal_attributes=None,
                 #HS code Marina
                 estimator_: BaseEstimator = DecisionTreeClassifier(max_leaf_nodes=20),
                 reg_param: float = 1,
                 shrinkage_scheme_: str = "node_based",
                 max_leaf_nodes: int = None,
                 random_state: int = None  #end of HS code Marina
     ):
        """ HoeffdingTreeClassifierHS class constructor."""
        super().__init__()
        self.max_byte_size = max_byte_size
        self.memory_estimate_period = memory_estimate_period
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.stop_mem_management = stop_mem_management
        self.remove_poor_atts = remove_poor_atts
        self.no_preprune = no_preprune
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0
        self.classes = None
        #HS code Marina
        self.reg_param = reg_param
        self.estimator_ = estimator_
        self.shrinkage_scheme_ = shrinkage_scheme_
        self.random_state = random_state
        if checks.check_is_fitted(self.estimator_):
            self._shrink()
        if max_leaf_nodes is not None:
            self.estimator_.max_leaf_nodes = max_leaf_nodes
            self.estimator_.random_state = random_state
        # end of HS code Marina


    @property
    def split_criterion(self):
        return self._split_criterion

    @split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion not in [self._GINI_SPLIT,
                                   self._INFO_GAIN_SPLIT,
                                   self._HELLINGER]:
            print("Invalid split_criterion option {}', will use default '{}'".
                  format(split_criterion, self._INFO_GAIN_SPLIT))
            self._split_criterion = self._INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @property
    def leaf_prediction(self):
        return self._leaf_prediction

    @leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in [self._MAJORITY_CLASS,
                                   self._NAIVE_BAYES,
                                   self._NAIVE_BAYES_ADAPTIVE]:
            print("Invalid leaf_prediction option {}', will use default '{}'".
                  format(leaf_prediction, self._NAIVE_BAYES_ADAPTIVE))
            self._leaf_prediction = self._NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    def measure_byte_size(self):
        """ Calculate the size of the tree.

        Returns
        -------
        int
            Size of the tree in bytes.

        """
        return calculate_object_size(self)

    def reset(self):
        """ Reset the Hoeffding Tree to default values."""
        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._inactive_leaf_byte_size_estimate = 0.0
        self._active_leaf_byte_size_estimate = 0.0
        self._byte_size_estimate_overhead_fraction = 1.0
        self._growth_allowed = True
        self._train_weight_seen_by_model = 0.0

        return self

    # insert hS code from imodels (Marina)
    def get_params(self, deep=True):
        d = {
            "reg_param": self.reg_param,
            "estimator_": self.estimator_,
            "shrinkage_scheme_": self.shrinkage_scheme_,
            "max_leaf_nodes": self.estimator_.max_leaf_nodes,
        }
        if deep:
            return deepcopy(d)
        return d


    def _shrink(self):
        print("inside _shrink()")

        self._shrink_tree(tree=self, reg_param=self.reg_param)



    def _shrink_tree(
        self, tree, reg_param, i=0, parent_val=None, parent_num=None, cum_sum=0
    ):
        """Shrink the tree"""
        # !!!! Start of addition in initial code : Skip shrinkage and return the tree as is for case lambda=0 !!!!
       # if reg_param == 0.0:
        #    return tree
        # !!!! End of addition in initial code : Skip shrinkage and return the tree as is for case lambda=0 !!!!

        print("reg_param prin : " + str(reg_param))
        if reg_param is None:
            reg_param = 1.0
        #if reg_param != 0.0:
        print("reg_param meta : " + str(reg_param))
        left = tree.children_left[i]
        right = tree.children_right[i]
        is_leaf = left == right
        n_samples = tree.weighted_n_node_samples[i]


        val = deepcopy(tree.value[i, :, :]) # changed based on regression imodels code logic

        # Step 1: Update cum_sum
        # if root
        if parent_val is None and parent_num is None:
            cum_sum = val

        # if has parent
        else:
            if self.shrinkage_scheme_ == "node_based":
                print("node_based shrink")
                val_new = (val - parent_val) / (1 + reg_param / parent_num)
            elif self.shrinkage_scheme_ == "constant":
                print("constant shrink")
                val_new = (val - parent_val) / (1 + reg_param)
            else:  # leaf_based
                val_new = 0
                print("leaf_based shrink")
            cum_sum += val_new

        # Step 2: Update node values
        if (
            self.shrinkage_scheme_ == "node_based"
            or self.shrinkage_scheme_ == "constant"
        ):
            tree.value[i, :, :] = cum_sum
            #print("NOT LBS")
        else:  # leaf_based
            #print("LBS")
            if is_leaf:  # update node values if leaf_based
                root_val = tree.value[0, :, :]
                tree.value[i, :, :] = root_val + (val - root_val) / (
                    1 + reg_param / n_samples
                )
            else:
                tree.value[i, :, :] = val

                # Step 3: Recurse if not leaf
        if not is_leaf:
            #print("NOT LBS 2")
            self._shrink_tree(
                tree,
                reg_param,
                left,
                parent_val=val,
                parent_num=n_samples,
                cum_sum=copy.deepcopy(cum_sum),
            )
            self._shrink_tree(
                tree,
                reg_param,
                right,
                parent_val=val,
                parent_num=n_samples,
                cum_sum=copy.deepcopy(cum_sum),
            )

            # edit the non-leaf nodes for later visualization (doesn't effect predictions)

        return tree
    # end of HS code (Marina)


    def partial_fit(self, X, y, classes=None, sample_weight=None):
        print("partial fit for HS")
        """ Incrementally trains the model. Train samples (instances) are
        composed of X attributes and their corresponding targets y.

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Instance attributes.
        y: array_like
            Classes (targets) for all samples in X.
        classes: numpy.array
            Contains the class values in the stream. If defined, will be used
            to define the length of the arrays returned by `predict_proba`
        sample_weight: float or array-like
            Samples weight. If not provided, uniform weights are assumed.

        Returns
        -------
            self

        Notes
        -----
        Tasks performed before training:

        * Verify instance weight. if not provided, uniform weights (1.0) are assumed.
        * If more than one instance is passed, loop through X and pass instances one at a time.
        * Update weight seen by model.

        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.

        """
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.
                                 format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self._partial_fit(X[i], y[i], sample_weight[i])

            print("before calling _shrink()")
            self._shrink()

        return self



    def _partial_fit(self, X, y, sample_weight):
        """ Trains the model on samples X and corresponding targets y.

        Private function where actual training is carried on.

        Parameters
        ----------
        X: numpy.ndarray of shape (1, n_features)
            Instance attributes.
        y: int
            Class label for sample X.
        sample_weight: float
            Sample weight.

        """
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
        leaf_node = found_node.node
        if leaf_node is None:
            leaf_node = self._new_learning_node()
            found_node.parent.set_child(found_node.parent_branch, leaf_node)
            self._active_leaf_node_cnt += 1
        if isinstance(leaf_node, LearningNode):
            learning_node = leaf_node
            learning_node.learn_one(X, y, weight=sample_weight, tree=self)
            if self._growth_allowed and isinstance(learning_node, ActiveLeaf):
                active_learning_node = learning_node
                weight_seen = active_learning_node.total_weight
                weight_diff = weight_seen - active_learning_node.last_split_attempt_at
                if weight_diff >= self.grace_period:
                    self._attempt_to_split(active_learning_node, found_node.parent,
                                           found_node.parent_branch)
                    active_learning_node.last_split_attempt_at = weight_seen
        # Split node encountered a previously unseen categorical value
        # (in a multi-way test)
        elif isinstance(leaf_node, SplitNode) and \
                isinstance(leaf_node.split_test, NominalAttributeMultiwayTest):
            # Creates a new branch to the new categorical value
            current = found_node.node
            leaf_node = self._new_learning_node()
            branch_id = current.split_test.add_new_branch(
                X[current.split_test.get_atts_test_depends_on()[0]]
            )
            current.set_child(branch_id, leaf_node)
            self._active_leaf_node_cnt += 1
            leaf_node.learn_one(X, y, weight=sample_weight, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_byte_size()



    def _get_votes_for_instance(self, X):
        """ Get class votes for a single instance.

        Parameters
        ----------
        X: numpy.ndarray of length equal to the number of features.
            Instance attributes.

        Returns
        -------
        dict (class_value, weight)

        """
        if self._tree_root is not None:
            found_node = self._tree_root.filter_instance_to_leaf(X, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                leaf_node = found_node.parent
            return leaf_node.predict_one(X, tree=self) if not isinstance(leaf_node, SplitNode) \
                else leaf_node.stats
        else:
            return {}

    def predict(self, X):
        """ Predicts the label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        y_proba = self.predict_proba(X)
        for i in range(r):
            index = np.argmax(y_proba[i])
            predictions.append(index)
        return np.array(predictions)


    def predict_proba(self, X):
        """ Predicts probabilities of all label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted the probabilities of all the labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = copy.deepcopy(self._get_votes_for_instance(X[i]))
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append([0])
            else:
                if sum(votes.values()) != 0:
                    votes = normalize_values_in_dict(votes, inplace=False)
                if self.classes is not None:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(votes.keys())) + 1)
                for key, value in votes.items():
                    y_proba[int(key)] = value
                predictions.append(y_proba)
        # Set result as np.array
        if self.classes is not None:
            predictions = np.asarray(predictions)
        else:
            # Fill missing values related to unobserved classes to ensure we get a 2D array
            predictions = np.asarray(list(itertools.zip_longest(*predictions, fillvalue=0.0))).T
        return predictions

    ########## Marina code starts here ###########

    # Predict proba : sort the example in a leaf. Then take the probabilities of each class in this leaf for this example
    # returns an array of probabilities, where each index corresponds to a different class
    # predict : select the class (index) that has maximum probability from predict proba (max element --> argmax).

    #Predict Proba with HS : instead of the elements being probabilties they will foloww the HS formula.
    #That is : root mean response + sum(difference between mean responses (in order) of each node in the path (except root)/(1 + lambda/samples in parent node)

    # def _get_votes_for_instanceHS(self, X): # We will return the votes for each class in the node along with path from node to root
    #     """ Get class votes for a single instance.
    #
    #     Parameters
    #     ----------
    #     X: numpy.ndarray of length equal to the number of features.
    #         Instance attributes.
    #
    #     Returns
    #     -------
    #     dict (class_value, weight)
    #
    #     """
    #     path  = []
    #     if self._tree_root is not None:
    #         found_node, path = self._tree_root.filter_instance_to_leafHS(X, None, -1)
    #         leaf_node = found_node.node
    #         if leaf_node is None:
    #             leaf_node = found_node.parent
    #         return leaf_node.predict_one(X, tree=self), path if not isinstance(leaf_node, SplitNode) \
    #             else leaf_node.stats, path
    #     else:
    #         return {}, path

    def get_votes_for_each_node_in_path(self, X, lambda_selected):


        #print("in")
        sum_of_hs = {}
        path = []
        if self._tree_root is not None:
            #print("self._tree_root is not None")
            found_node, path = self._tree_root.filter_instance_to_leafHS(X, None, -1)
            leaf_node = found_node.node
            if leaf_node is None:
                #print("leaf_node is None")
                path.remove(leaf_node)
                leaf_node = found_node.parent

            #print("len(path) : " + str(len(path)))
            if len(path) == 1 :
                return leaf_node.predict_one(X, tree=self) if not isinstance(leaf_node, SplitNode) \
                    else leaf_node.stats

            path_leaf_to_root = list(reversed(path))


            # TODO : see if we can do normalization in the votes dictionary
            # if sum(stats_of_current_node.values()) != 0:
            #   votes = normalize_values_in_dict(votes, inplace=False)

            for i in range(len(path_leaf_to_root)-2):
                print("iterate the path")
                current_node = path_leaf_to_root[i]
                stats_of_current_node  = current_node._stats # class votes
                number_of_samples_in_current_node = sum(stats_of_current_node.values())
                parent_node = path_leaf_to_root[i+1]
                stats_of_parent_node = parent_node._stats  # class votes
                number_of_samples_in_parent_node = sum(stats_of_parent_node.values())
                for key, value in stats_of_current_node.items():
                    for key2, value2 in stats_of_parent_node.items():
                        if int(key) == int(key2):
                            paronomastis = 1 + lambda_selected / number_of_samples_in_parent_node
                            current_node_root_mean_response = value/number_of_samples_in_current_node
                            parent_node_mean_response = value2/number_of_samples_in_parent_node
                            sum_of_hs[int(key)] += (current_node_root_mean_response - parent_node_mean_response)/paronomastis
            for key3, value3 in sum_of_hs.items():
                root_mean_response = self._tree_root.stats[key3] / sum(self._tree_root.stats.values())
                sum_of_hs[int(key3)] = root_mean_response + value3

            # Set result as np.array
        #print("sum_of_hs : " + str(sum_of_hs))
        return sum_of_hs

    def predictHS(self, X, lambda_selected=0.5):
        """ Predicts the label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        y_proba_hs = self.predict_probaHS(X, lambda_selected)
       # print("y_proba_hs : " + str(y_proba_hs))
        for i in range(r):
            index = np.argmax(y_proba_hs[i])
            predictions.append(index)
        return np.array(predictions)

    def predict_probaHS(self, X, lambda_selected):
        """ Predicts probabilities of all label of the X instance(s)

        Parameters
        ----------
        X: numpy.ndarray of shape (n_samples, n_features)
            Samples for which we want to predict the labels.

        Returns
        -------
        numpy.array
            Predicted the probabilities of all the labels for all instances in X.

        """
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            pred_example = copy.deepcopy(self.get_votes_for_each_node_in_path(X[i], lambda_selected))
            print("pred example : " + str(pred_example))
            if pred_example == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append([0])
            else:
                if sum(pred_example.values()) != 0:
                    pred_example = normalize_values_in_dict(pred_example, inplace=False)
                if self.classes is not None:
                     y_proba = np.zeros(int(max(self.classes)) + 1)
                else:
                     y_proba = np.zeros(int(max(pred_example.keys())) + 1)
                for key, value in pred_example.items():
                     y_proba[int(key)] = value
                predictions.append(y_proba)
        # Set result as np.array
        if self.classes is not None:
            predictions = np.asarray(predictions)
        else:
            # Fill missing values related to unobserved classes to ensure we get a 2D array
            predictions = np.asarray(list(itertools.zip_longest(*predictions, fillvalue=0.0))).T
        return predictions


    # def predict_probaHS(self, X, lambda_selected):
    #     """ Predicts probabilities of all label of the X instance(s)
    #
    #     Parameters
    #     ----------
    #     X: numpy.ndarray of shape (n_samples, n_features)
    #         Samples for which we want to predict the labels.
    #
    #     Returns
    #     -------
    #     numpy.array
    #         Predicted the probabilities of all the labels for all instances in X.
    #
    #     """
    #     r, _ = get_dimensions(X)
    #     predictions = []
    #     for i in range(r):
    #         #vote : key class , values how many samples in this class
    #         # path + node's path from it till the root
    #         votes, path = copy.deepcopy(self._get_votes_for_instanceHS(X[i]))
    #         for votes in votes_list:
    #             if votes == {}:
    #                 # Tree is empty, all classes equal, default to zero
    #                 predictions.append([0])
    #             else:
    #                 if sum(votes.values()) != 0:
    #                     votes = normalize_values_in_dict(votes, inplace=False)
    #                 if self.classes is not None:
    #                     y_proba = np.zeros(int(max(self.classes)) + 1)
    #                 else:
    #                     y_proba = np.zeros(int(max(votes.keys())) + 1)
    #                 # for key, value in votes.items():
    #                 #     y_proba[int(key)] = value
    #
    #                 #TODO : take the class votes of each node in path and calculate the mean response
    #
    #                 for key, value in votes.items():
    #
    #                     y_proba[int(key)] = value
    #                 predictions.append(y_proba)
    #             # Set result as np.array
    #             if self.classes is not None:
    #                 predictions = np.asarray(predictions)
    #             else:
    #                 # Fill missing values related to unobserved classes to ensure we get a 2D array
    #                 predictions = np.asarray(list(itertools.zip_longest(*predictions, fillvalue=0.0))).T
    #             return predictions
    #                 # path_leaf_to_root = list(reversed(path))
    #                 # sum_of_hs_equation = 0
    #                 # root_mean_response = 0
    #                 # for k in range(len(path_leaf_to_root) - 2):  # we leave the root out of the loop (sum)
    #                 #
    #                 #     current_node = path_leaf_to_root[k]
    #                 #     parent_node = path_leaf_to_root[k + 1]
    #                 #     # if counter < len(path_leaf_to_root) -1 :
    #                 #     #    print("current node : " + str(current_node) + " " + "current node mean response : " + str(parent_node.stats[1] / parent_node.stats[0]))
    #                 #     #    print("parent node : " + str(parent_node) + " parent node mean response : " + str(parent_node.stats[1] / parent_node.stats[0]))
    #                 #
    #                 #     current_node_class_votes = current_node.get_class_votes(X, self)
    #                 #     total_weight_current_node = sum(current_node_class_votes.values())  # Sum of all weights
    #                 #     print("total weight current node : " + str(total_weight_current_node))
    #                 #     if total_weight_current_node > 0:
    #                 #         current_node_mean_response = sum(key * value for key, value in
    #                 #                                          current_node_class_votes.items()) / total_weight_current_node
    #                 #     else:
    #                 #         current_node_mean_response = 0  # Handle case when there are no votes (avoid division by zero)
    #                 #
    #                 #     parent_class_votes = parent_node.get_class_votes(X, self)
    #                 #     total_weight_parent_node = sum(parent_class_votes.values())  # Sum of all weights
    #                 #     if total_weight_parent_node > 0:
    #                 #         parent_node_mean_response = sum(key * value for key, value in
    #                 #                                         parent_class_votes.items()) / total_weight_parent_node
    #                 #     else:
    #                 #         parent_node_mean_response = 0  # Handle case when there are no votes (avoid division by zero)
    #                 #
    #                 #     samples_in_parent = total_weight_parent_node
    #                 #     paronomastis = 1 + lambda_selected / samples_in_parent
    #                 #     sum_of_hs_equation += (current_node_mean_response - parent_node_mean_response) / paronomastis
    #                 #
    #                 #     root_class_votes = self._tree_root.get_class_votes(X, self)
    #                 #     total_weight_root_node = sum(current_node_class_votes.values())
    #                 #
    #                 #     if total_weight_root_node > 0:
    #                 #         root_mean_response = sum(key * value for key, value in
    #                 #                                         root_class_votes.items()) / total_weight_root_node
    #                 #     else:
    #                 #         root_mean_response = 0
    #                 #     for key, value in votes.items():
    #                 #         y_proba[int(key)] = value
    #                 # predictions.append(root_mean_response + sum_of_hs_equation)
    #

    ########## Marina code ends here ###########

    @property
    def model_measurements(self):
        """ Collect metrics corresponding to the current status of the tree.

        Returns
        -------
        string
            A string buffer containing the measurements of the tree.
        """
        measurements = {'Tree size (_nodes)': self._decision_node_cnt
                        + self._active_leaf_node_cnt + self._inactive_leaf_node_cnt,
                        'Tree size (leaves)': self._active_leaf_node_cnt
                        + self._inactive_leaf_node_cnt,
                        'Active learning _nodes': self._active_leaf_node_cnt,
                        'Tree depth': self._measure_tree_depth(),
                        'Active leaf byte size estimate': self._active_leaf_byte_size_estimate,
                        'Inactive leaf byte size estimate': self._inactive_leaf_byte_size_estimate,
                        'Byte size estimate overhead': self._byte_size_estimate_overhead_fraction
                        }
        return measurements

    def _measure_tree_depth(self):
        """ Calculate the depth of the tree.

        Returns
        -------
        int
            Depth of the tree.
        """
        if isinstance(self._tree_root, Node):
            return self._tree_root.subtree_depth()
        return 0

    def _new_learning_node(self, initial_class_observations=None, is_active=True):
        """ Create a new learning node.

        The type of learning node depends on the tree configuration.
        """
        if initial_class_observations is None:
            initial_class_observations = {}
        if is_active:
            if self._leaf_prediction == self._MAJORITY_CLASS:
                return ActiveLearningNodeMC(initial_class_observations)
            elif self._leaf_prediction == self._NAIVE_BAYES:
                return ActiveLearningNodeNB(initial_class_observations)
            else:  # NAIVE BAYES ADAPTIVE (default)
                return ActiveLearningNodeNBA(initial_class_observations)
        else:
            return InactiveLearningNodeMC(initial_class_observations)

    def get_model_description(self):
        """ Walk the tree and return its structure in a buffer.

        Returns
        -------
        string
            The description of the model.

        """
        if self._tree_root is not None:
            buffer = ['']
            description = ''
            self._tree_root.describe_subtree(self, buffer, 0)
            for line in range(len(buffer)):
                description += buffer[line]
            return description

    @staticmethod
    def _hoeffding_bound(range_val, confidence, n):
        r""" Compute the Hoeffding bound, used to decide how many samples are necessary at each
        node.

        Notes
        -----
        The Hoeffding bound is defined as:

        .. math::

           \epsilon = \sqrt{\frac{R^2\ln(1/\delta))}{2n}}

        where:

        :math:`\epsilon`: Hoeffding bound.

        :math:`R`: Range of a random variable. For a probability the range is 1, and for an
        information gain the range is log *c*, where *c* is the number of classes.

        :math:`\delta`: Confidence. 1 minus the desired probability of choosing the correct
        attribute at any given node.

        :math:`n`: Number of samples.

        Parameters
        ----------
        range_val: float
            Range value.
        confidence: float
            Confidence of choosing the correct attribute.
        n: int or float
            Number of samples.

        Returns
        -------
        float
            The Hoeffding bound.

        """
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))

    def _new_split_node(self, split_test, class_observations):
        """ Create a new split node."""
        return SplitNode(split_test, class_observations)

    def _attempt_to_split(self, node, parent: SplitNode, parent_idx: int):
        """ Attempt to split a node.

        If the samples seen so far are not from the same class then:

        1. Find split candidates and select the top 2.
        2. Compute the Hoeffding bound.
        3. If the difference between the top 2 split candidates is larger than the Hoeffding bound:
           3.1 Replace the leaf node by a split node.
           3.2 Add a new leaf node on each branch of the new split node.
           3.3 Update tree's metrics

        Optional: Disable poor attribute. Depends on the tree's configuration.

        Parameters
        ----------
        node:
            The node to evaluate.
        parent: SplitNode
            The node's parent.
        parent_idx: int
            Parent node's branch index.

        """
        if not node.observed_class_distribution_is_pure():
            if self._split_criterion == self._GINI_SPLIT:
                split_criterion = GiniSplitCriterion()
            elif self._split_criterion == self._INFO_GAIN_SPLIT:
                split_criterion = InfoGainSplitCriterion()
            elif self._split_criterion == self._HELLINGER:
                split_criterion = HellingerDistanceCriterion()
            else:
                split_criterion = InfoGainSplitCriterion()
            best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort(key=attrgetter('merit'))
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(split_criterion.get_range_of_merit(
                    node.stats), self.split_confidence, node.total_weight)
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):
                    should_split = True
                if self.remove_poor_atts:
                    poor_atts = set()
                    # Add any poor attribute to set
                    for i in range(len(best_split_suggestions)):
                        if best_split_suggestions[i] is not None:
                            split_atts = best_split_suggestions[i].split_test.\
                                get_atts_test_depends_on()
                            if len(split_atts) == 1:
                                if (best_suggestion.merit - best_split_suggestions[i].merit
                                        > hoeffding_bound):
                                    poor_atts.add(int(split_atts[0]))
                    for poor_att in poor_atts:
                        node.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.split_test is None:
                    # Preprune - null wins
                    self._deactivate_learning_node(node, parent, parent_idx)
                else:
                    new_split = self._new_split_node(split_decision.split_test, node.stats)

                    for i in range(split_decision.num_splits()):
                        new_child = self._new_learning_node(
                            split_decision.resulting_stats_from_split(i))
                        new_split.set_child(i, new_child)
                    self._active_leaf_node_cnt -= 1
                    self._decision_node_cnt += 1
                    self._active_leaf_node_cnt += split_decision.num_splits()
                    if parent is None:
                        self._tree_root = new_split
                    else:
                        parent.set_child(parent_idx, new_split)
                # Manage memory
                self._enforce_tracker_limit()

    def _sort_learning_nodes(self, learning_nodes):
        """ Define strategy to sort learning _nodes according to their likeliness of being split."""
        learning_nodes.sort(key=lambda n: n.node.calculate_promise())
        return learning_nodes

    def _enforce_tracker_limit(self):
        """ Track the size of the tree and disable/enable _nodes if required."""
        byte_size = (self._active_leaf_byte_size_estimate
                     + self._inactive_leaf_node_cnt * self._inactive_leaf_byte_size_estimate) \
            * self._byte_size_estimate_overhead_fraction
        if self._inactive_leaf_node_cnt > 0 or byte_size > self.max_byte_size:
            if self.stop_mem_management:
                self._growth_allowed = False
                return
        learning_nodes = self._find_learning_nodes()
        learning_nodes = self._sort_learning_nodes(learning_nodes)
        max_active = 0
        while max_active < len(learning_nodes):
            max_active += 1
            if (((max_active * self._active_leaf_byte_size_estimate
                    + (len(learning_nodes) - max_active) * self._inactive_leaf_byte_size_estimate)
                    * self._byte_size_estimate_overhead_fraction) > self.max_byte_size):
                max_active -= 1
                break
        cutoff = len(learning_nodes) - max_active
        for i in range(cutoff):
            if isinstance(learning_nodes[i].node, ActiveLeaf):
                self._deactivate_learning_node(learning_nodes[i].node,
                                               learning_nodes[i].parent,
                                               learning_nodes[i].parent_branch)
        for i in range(cutoff, len(learning_nodes)):
            if isinstance(learning_nodes[i].node, InactiveLeaf):
                self._activate_learning_node(learning_nodes[i].node,
                                             learning_nodes[i].parent,
                                             learning_nodes[i].parent_branch)

    def _estimate_model_byte_size(self):
        """ Calculate the size of the model and trigger tracker function if the actual model size
        exceeds the max size in the configuration."""
        learning_nodes = self._find_learning_nodes()
        total_active_size = 0
        total_inactive_size = 0
        for found_node in learning_nodes:
            if not found_node.node.is_leaf():  # Safety check for non-trivial tree structures
                continue
            if isinstance(found_node.node, ActiveLeaf):
                total_active_size += calculate_object_size(found_node.node)
            else:
                total_inactive_size += calculate_object_size(found_node.node)
        if total_active_size > 0:
            self._active_leaf_byte_size_estimate = total_active_size / self._active_leaf_node_cnt
        if total_inactive_size > 0:
            self._inactive_leaf_byte_size_estimate = total_inactive_size \
                / self._inactive_leaf_node_cnt
        actual_model_size = calculate_object_size(self)
        estimated_model_size = (self._active_leaf_node_cnt * self._active_leaf_byte_size_estimate
                                + self._inactive_leaf_node_cnt
                                * self._inactive_leaf_byte_size_estimate)
        self._byte_size_estimate_overhead_fraction = actual_model_size / estimated_model_size
        if actual_model_size > self.max_byte_size:
            self._enforce_tracker_limit()

    def _deactivate_all_leaves(self):
        """ Deactivate all leaves. """
        learning_nodes = self._find_learning_nodes()
        for cur_node in learning_nodes:
            if isinstance(cur_node, ActiveLeaf):
                self._deactivate_learning_node(cur_node.node,
                                               cur_node.parent,
                                               cur_node.parent_branch)

    def _deactivate_learning_node(self, to_deactivate: ActiveLeaf, parent: SplitNode,
                                  parent_branch: int):
        """ Deactivate a learning node.

        Parameters
        ----------
        to_deactivate: ActiveLearningNode
            The node to deactivate.
        parent: SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self._new_learning_node(
            to_deactivate.stats, is_active=False
        )
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1

    def _activate_learning_node(self, to_activate: InactiveLeaf, parent: SplitNode,
                                parent_branch: int):
        """ Activate a learning node.

        Parameters
        ----------
        to_activate: InactiveLearningNode
            The node to activate.
        parent: SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch index.

        """
        new_leaf = self._new_learning_node(to_activate.stats)
        if parent is None:
            self._tree_root = new_leaf
        else:
            parent.set_child(parent_branch, new_leaf)
        self._active_leaf_node_cnt += 1
        self._inactive_leaf_node_cnt -= 1

    def _find_learning_nodes(self):
        """ Find learning _nodes in the tree.

        Returns
        -------
        list
            List of learning _nodes in the tree.
        """
        found_list = []
        self.__find_learning_nodes(self._tree_root, None, -1, found_list, 0)
        return found_list

    def __find_learning_nodes(self, node, parent, parent_branch, found, depth):
        """ Find learning _nodes in the tree from a given node.

        Parameters
        ----------
        node: skmultiflow.trees._nodes.Node
            The node to start the search.
        parent: LearningNode or SplitNode
            The node's parent.
        parent_branch: int
            Parent node's branch.
        depth: int
            The node's depth.

        Returns
        -------
        list
            List of learning _nodes.
        """
        if node is not None:
            if isinstance(node, LearningNode):
                found.append(FoundNode(node, parent, parent_branch, depth))
            if isinstance(node, SplitNode):
                split_node = node
                for i in range(split_node.n_children):
                    self.__find_learning_nodes(
                        split_node.get_child(i), split_node, i, found, depth + 1
                    )

    def get_model_rules(self):
        """ Returns list of rules describing the tree.

        Returns
        -------
        list (Rule)
            list of the rules describing the tree
        """
        root = self._tree_root
        rules = []

        def recurse(node, cur_rule, ht):
            if isinstance(node, SplitNode):
                for i, child in node._children.items():
                    predicate = node.get_predicate(i)
                    r = copy.deepcopy(cur_rule)
                    r.predicate_set.append(predicate)
                    recurse(child, r, ht)
            else:
                cur_rule.observed_class_distribution = node.stats.copy()
                cur_rule.class_idx = max(node.stats.items(), key=itemgetter(1))[0]
                rules.append(cur_rule)

        rule = Rule()
        recurse(root, rule, self)
        return rules

    def get_rules_description(self):
        """ Prints the description of tree using rules."""
        description = ''
        for rule in self.get_model_rules():
            description += str(rule) + '\n'

        return description
