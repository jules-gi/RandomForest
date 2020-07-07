"""
This module provides an Experiment object to run random forest expriments.

Contributors :
    Jules Girard, jules.girard@outlook.com

Update :  july 7, 2020
"""

import os
from warnings import warn
import copy
import numpy as np
from time import perf_counter

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_random_state
from sklearn.tree import DecisionTreeClassifier

from rflitis.utils import upload
from rflitis.utils.samples import get_class_dictionary


MAX_INT = np.iinfo(np.int32).max


class Experiment:
    """
    Run experiment by building, saving and loading models and results of
    the experiment.

    ----------
    Parameters
    ----------

    X: array-like of shape (n_samples, n_features)
        Dataset to use in the experiment.

    y: array-like of shape (n_samples,)
        Labels of the dataset to use in the experiment.

    n_replications: int (default = 10)
        Number of experiment replications.

    classifier: RandomForestClassifier Object from sklearn,
                or None (default = None)
        Random Forest initialized.

    train_size: float (default = 0.5)
        Between 0.0 and 1.0, represent the proportion of the dataset
        to include in the training set.

    random_state: int or RandomState (default = None)
        Controls the randomness the samples selection
        when building training and test sets.

    path: string, or None (default = None)
        path to save and/or load models and results.

    load: boolean (default = False)
        If True, load the models existing in the given path (path must contains
        models previously built).
    """

    def __init__(self,
                 X, y,
                 # sample_weight=None,
                 n_replications=10,
                 classifier=None,
                 train_size=.50,
                 random_state=None,
                 path=None,
                 load=False):

        self._X = X
        self._y = y
        # self.sample_weight = sample_weight
        self.n_replications = n_replications
        self._classifier = classifier
        self._train_size = train_size
        self._random_state = random_state
        self._path = path

        self.models_ = []
        self._train_test_samples = {"train": {}, "test": {}}
        self._n_instances, self._n_features = X.shape
        self._class_dict, self._n_classes = get_class_dictionary(y)
        self._params = ("n_replications", "_train_size", "_random_state",
                        "_path", "_n_instances", "_n_features")

        if path is not None:
            upload.check_path(path)

            if len(os.listdir(path)) != 0:
                reply = ""
                while True:
                    if load or reply == "yes":
                        # Load previous attributes and models
                        self._load_attributes()
                        self._load_models()

                        break
                    elif reply == "no":
                        raise FileExistsError(f"'{path}' already exist: "
                                              f"Change your saving path.")

                    reply = str(input(f"'{path}' already contains results."
                                      f" Do you want to load it and change"
                                      f" parameters to the previous state "
                                      f"(yes/no)? "))
            else:
                self._make_directory()

        elif load:
            raise ValueError(f"'load' attribute can't be True if 'path' "
                             f"is None: load={load}, path={path}.")

        elif classifier is None:
            raise ValueError(f"'_classifier' is require when 'path' is None: "
                             f"_classifier={classifier}")

    def _load_attributes(self):
        """Load self attributes saved in path."""

        params = upload.load_dict(f"{self._path}/parameters/exp_params.json")

        for param, value in params.items():
            if getattr(self, param) != value:
                self.__setattr__(param, value)
                warn(f"Loaded experiment had different attributes than the "
                     f"initial experiment : '{param}' changed to {value}.")

        if (self._n_instances, self._n_features) != self._X.shape:
            raise ValueError(f"Input data dimensions don't match saved data "
                             f"dimensions. Current dimensions are "
                             f"{self._X.shape}, whereas saved dimensions were "
                             f"{(self._n_instances, self._n_features)}.")

        if self._classifier is not None:
            warn("Previous '_classifier' was loaded.")
        self._classifier = \
            upload.load_model(f"{self._path}/parameters/base_clf.joblib")

    def _load_models(self):
        """Load models saved in path."""

        if os.path.exists(f"{self._path}/dump/models"):
            self.models_ = [
                upload.load_model(f"{self._path}/dump/models/{filename}")
                for filename in os.listdir(f"{self._path}/dump/models")
                if ".joblib" in filename
            ]

            self._train_test_samples = \
                upload.load_dict(f"{self._path}/"
                                 f"parameters/train_test_samples.json")

            if len(self.models_) < self.n_replications:
                warn(f"{len(self.models_)} models are already build. Call "
                     f"'build_models' in order to build the "
                     f"{self.n_replications - len(self.models_)} next models.")
            elif len(self.models_) > self.n_replications:
                raise AttributeError(f"Number of replications less than the "
                                     f"number of models already built:"
                                     f"{len(self.models_)} models built, "
                                     f"{self.n_replications} replications "
                                     f"required.")

    def _make_directory(self):
        """Create directory to save models and results in path."""

        if self._classifier is None:
            raise ValueError(f"'_classifier' is require when 'path' is None: "
                             f"_classifier={self._classifier}")

        os.mkdir(f"{self._path}/parameters")
        os.mkdir(f"{self._path}/dump")
        os.mkdir(f"{self._path}/results")

        self._save_params()
        upload.save_model(model=self._classifier,
                          filename=f"{self._path}/parameters/base_clf.joblib")

    def _save_params(self):
        """Save self atributes in path."""

        params = {key: self.__getattribute__(key)
                  for key in self._params}

        upload.save_dict(dictionary=params,
                         filename=f"{self.path}/parameters/exp_params.json",
                         overwrite=True)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, new_path):
        """Set path attribute if not exist already."""

        if self._path is None:
            upload.check_path(new_path)
            if len(os.listdir(new_path)) == 0:
                self._path = new_path
                self._make_directory()
                # Save models if it exists
                if len(self.models_) > 0:
                    i = 0
                    for model in self.models_:
                        upload.save_model(model=model,
                                          filename=f"{self._path}/dump/"
                                                   f"models/model_{i}.joblib")
            else:
                raise FileExistsError(f"'{new_path}' already exist: "
                                      f"Change your saving path.")
        else:
            raise AttributeError(f"Changing path attribute is forbidden when "
                                 f"it's already set: path='{self._path}'.")

    def _check_is_built(self):
        """Check if self contains models."""

        if len(self.models_) == 0:
            raise AttributeError(f"No models exists: call 'build_models' "
                                 f"method with appropriate parameters before "
                                 f"using this experiment.")

    def _check_save_attribute(self, save):
        """Check if path exist to save models or results."""
        if self._path is None and save:
            raise ValueError(f"Can't save results if 'path' is None. Change "
                             f"'path' attribute if you need to save results.")

    def build_models(self):
        """
        Build models of the experiment. If path is not None, will save the
        models in the given path.
        """

        n_more_models = self.n_replications - len(self.models_)

        if n_more_models < 0:
            raise ValueError(f"'n_replications' attribute must be larger or "
                             f"equal to the number of models built: "
                             f"n_replications={self.n_replications}, n_models"
                             f"={len(self.models_)}")
        elif n_more_models == 0:
            warn(f"Building models without increasing "
                 f"'n_replications' attribute does not build "
                 f"new models: n_replications={self.n_replications}.")
        else:
            if not os.path.exists(f"{self._path}/dump/models"):
                os.mkdir(f"{self._path}/dump/models")

            indices = np.arange(self._n_instances).tolist()
            random_state_split = check_random_state(self._random_state)

            if len(self.models_) > 0 and self._random_state is not None:
                # Get the random state we would have
                # if we didn't use warm_start strategy
                for i in range(len(self.models_)):
                    train_test_split(self._X, self._y, indices,
                                     train_size=self._train_size,
                                     random_state=random_state_split)

            time_fit = {}
            for rep in range(n_more_models):
                i = len(self.models_)
                print(f"Build model {i} / {self.n_replications - 1}")

                X_train, _, y_train, _, id_train, id_test = \
                    train_test_split(self._X, self._y, indices,
                                     train_size=self._train_size,
                                     random_state=random_state_split)

                self._train_test_samples["train"][str(i)] = list(id_train)
                self._train_test_samples["test"][str(i)] = list(id_test)

                model = copy.deepcopy(self._classifier)
                start = perf_counter()
                model.fit(X_train, y_train)
                time_fit[i] = perf_counter() - start

                self.models_.append(model)

                # Save models, train_test_split and time_fit results
                if self._path is not None:
                    upload.save_dict(dictionary=self._train_test_samples,
                                     filename=f"{self._path}/parameters/"
                                              f"train_test_samples.json",
                                     overwrite=True)
                    upload.save_dict(dictionary=time_fit,
                                     filename=f"{self._path}/"
                                              f"results/time_fit.json",
                                     overwrite=True)
                    upload.save_model(model=model,
                                      filename=f"{self._path}/dump/"
                                               f"models/model_{i}.joblib")

    def get_score(self, voting="hard", return_time=False, save=False):
        """
        Compute performance score of the 'n_replications' models.

        voting: {"hard", "soft"}, (default = "hard")
            Use hard-voting or soft-voting to compute models performance score.

        return_time: boolean (default = False)
            If True, return computation time for each models.

        save: boolean (default = False)
            If True and path is not None, save results in the given path.
        """

        self._check_is_built()
        self._check_save_attribute(save)

        score = {}
        time_score = {}
        i = 0
        for model in self.models_:
            test_ids = np.asarray(self._train_test_samples["test"][str(i)])

            start = perf_counter()
            score[i] = model.score(self._X[test_ids],
                                   self._y[test_ids],
                                   voting=voting)
            # TODO : implement score with sample_weight attribute
            time_score[i] = perf_counter() - start

            i += 1

        if save:
            upload.save_dict(dictionary=score,
                             filename=f"{self._path}/results/score.json",
                             overwrite=True)
            upload.save_dict(dictionary=time_score,
                             filename=f"{self._path}/results/time_score.json",
                             overwrite=True)

        if return_time:
            return score, time_score
        else:
            return score

    def get_monitoring(self):
        # TODO
        self._check_is_built()
        pass

    def get_leaves_structure(self, save=False):
        """
        Analyze trees structure for each models.

        save: boolean (default = False)
            If True and path is not None, save results in the given path.
        """
        self._check_is_built()
        self._check_save_attribute(save)

        n_models = len(self.models_)
        n_leaves = {i: [] for i in range(n_models)}
        leaves_depth_mean = {i: [] for i in range(n_models)}
        leaves_depth_std = {i: [] for i in range(n_models)}

        i = 0
        for model in self.models_:
            for estimator in model.estimators_:

                n_nodes = estimator.tree_.node_count
                children_left = estimator.tree_.children_left
                children_right = estimator.tree_.children_right

                depth = np.zeros(n_nodes, dtype=int)
                leaves = np.zeros(n_nodes, dtype=bool)
                stack = [(0, -1)]
                while len(stack) > 0:
                    node_id, parent_depth = stack.pop()
                    depth[node_id] = parent_depth + 1

                    if children_left[node_id] != children_right[node_id]:
                        stack.append((children_left[node_id],
                                      parent_depth + 1))
                        stack.append((children_right[node_id],
                                      parent_depth + 1))
                    else:
                        leaves[node_id] = True

                n_leaves[i].append(int(leaves.sum()))
                leaves_depth_mean[i].append(depth[leaves].mean())
                leaves_depth_std[i].append(depth[leaves].std())
            i += 1

        leaves_structures = {
            "n_leaves": n_leaves,
            "leaves_depth_mean": leaves_depth_mean,
            "leaves_depth_std": leaves_depth_std
        }

        if save:
            upload.save_dict(dictionary=leaves_structures,
                             filename=f"{self._path}/"
                                      f"results/leaves_structures.json",
                             overwrite=True)

    def get_feature_importances(self, save=False):
        """
        Compute feature importances for each models.

        save: boolean (default = False)
            If True and path is not None, save results in the given path.

        See feature_importances_ of RandomForestClassifier (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.feature_importances_)
        """
        self._check_is_built()
        self._check_save_attribute(save)

        features_importance = {}
        i = 0
        for model in self.models_:
            features_importance[i] = model.feature_importances_.tolist()
            i += 1

        if save:
            upload.save_dict(dictionary=features_importance,
                             filename=f"{self._path}/"
                                      f"results/features_importance.json",
                             overwrite=True)

        return features_importance

    def get_feature_impurity_gain(self, save=False):
        """
        Compute features impurity gain for each models.

        save: boolean (default = False)
            If True and path is not None, save results in the given path.
        """

        self._check_save_attribute(save)

        impurity_gain = {}
        for criterion in ("entropy", "gini"):

            impurity_gain[criterion] = []
            for feature_id in range(self._n_features):
                tree = DecisionTreeClassifier(criterion=criterion,
                                              max_features=1,
                                              max_depth=1,
                                              random_state=self._random_state)

                tree.fit(self._X[:, feature_id].reshape(-1, 1), self._y)
                if len(tree.tree_.impurity) == 1:
                    impurity_gain[criterion].append(0.0)
                elif len(tree.tree_.impurity) == 3:
                    par_impurity, left_impurity, right_impurity = \
                        tree.tree_.impurity
                    n_par, n_left, _ = tree.tree_.n_node_samples
                    p = n_left / n_par
                    impurity_gain[criterion].append(par_impurity -
                                                    p * left_impurity -
                                                    (1 - p) * right_impurity)
                else:
                    raise ValueError(f"Length of 'tree.tree_.impurity' should "
                                     f"be 1 or 3, got "
                                     f"{len(tree.tree_.impurity)}.")

        if save:
            upload.save_dict(dictionary=impurity_gain,
                             filename=f"{self._path}/"
                                      f"results/features_impurity_gain.json",
                             overwrite=True)

        return impurity_gain

    def search_max_feature(self,
                           n_estimators=100,
                           criterion="entropy",
                           feature_step=None,
                           train_size=2/3,
                           n_jobs=-1,
                           save=False):
        """
        Search the optimal 'max_features' to get the best performance score.

        n_estimators: int (default = 100)
            The number of trees in the forest.

        criterion: {'entropy', 'gini'}, (default = 'entropy')
            The strategy used to choose the split at each node. Supported
            strategies are "best" to choose the best split and "random" to
            choose the best random split (from sklearn).

        feature_step: int < n_features (default = None)
            If is None and n_features < 100, feature_step = 1, else 5.

        training_size: float (default = 0.667)
            Between 0.0 and 1.0, represent the proportion of the dataset
            to include in the training set.

        n_jobs: int (default = -1)
            The number of jobs to run in parallel
            (See https://scikit-learn.org/stable/glossary.html#term-n-jobs)

        save: boolean (default = False)
            If True and path is not None, save results in the given path.
        """

        self._check_save_attribute(save)
        random_state_estimator = check_random_state(self._random_state)
        random_state_sample = check_random_state(self._random_state)

        if not os.path.exists(f"{self._path}/dump/search_max_feature"):
            os.mkdir(f"{self._path}/dump/search_max_feature")

        result_filename = f"{self._path}/results/search_max_feature.npy"
        if not os.path.isfile(result_filename):
            if feature_step is None:
                feature_step = 1 if self._n_features < 100 else 5

            clf = RandomForestClassifier(n_estimators=n_estimators,
                                         criterion=criterion,
                                         n_jobs=n_jobs,
                                         random_state=random_state_estimator)

            prev_k = 0
            results = {}
            for k in range(1, self._n_features + 1, feature_step):
                print(f"max_features = {k}/{self._n_features}")
                tmp_filename = f"{self._path}/dump/search_max_feature/" \
                               f"max_features_{k}.npy"
                if not os.path.isfile(tmp_filename):
                    if prev_k > 0:
                        random_state_estimator.randint(MAX_INT, size=prev_k)
                        for i in range(prev_k):
                            train_test_split(
                                self._X, self._y,
                                train_size=train_size,
                                random_state=random_state_sample)
                        prev_k = 0

                    clf.max_features = k
                    scores = np.empty(self.n_replications)
                    for i in range(self.n_replications):
                        X_train, X_test,\
                            y_train, y_test = train_test_split(
                                self._X, self._y,
                                train_size=train_size,
                                random_state=random_state_sample)

                        clf.fit(X_train, y_train)
                        scores[i] = (1.0 -
                                     clf.score(X_test, y_test)) * 100.0

                    np.save(tmp_filename, scores)
                else:
                    scores = np.load(tmp_filename)
                    prev_k += 1

                results[str(k)] = scores.tolist()
        else:
            results = upload.load_dict(result_filename)

        if save:
            upload.save_dict(dictionary=results,
                             filename=f"{self._path}/"
                                      f"results/search_max_feature.json",
                             overwrite=True)

        return results


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import fetch_openml
    from rflitis import ForestRKClassifier

    path = "/Users/jules/Desktop/Test"
    X, y = fetch_openml("diabetes", version=1, return_X_y=True)
    clf = ForestRKClassifier(n_estimators=30)

    exp = Experiment(X=X, y=y, classifier=clf, path=path)
    # exp.build_models()
    # exp.get_score(voting="hard", save=True)
    # exp.get_feature_importances(save=True)
    # exp.get_feature_impurity_gain(save=True)
    # exp.get_leaves_structure(save=True)
    exp.search_max_features(n_estimators=100, save=True)

    print("Finished")
