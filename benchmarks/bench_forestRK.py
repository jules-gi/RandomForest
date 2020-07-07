import os
from rflitis import ForestRKClassifier
from rflitis.experiment.experiment import Experiment
from sklearn.datasets import fetch_openml

N_ESTIMATORS = 300
RANDOM_STATE = 123
CRITERION = "entropy"
PATH = f"/Volumes/JulesG/SID/RandomForest/Experiments"

data_ens = [
    (fetch_openml("ecoli", version=2)),
    (fetch_openml("diabetes", version=1)),
    (fetch_openml("vehicle", version=1)),
    (fetch_openml("clean1", version=1)),
    (fetch_openml("arcene", version=1)),
    (fetch_openml("madelon", version=1)),
    (fetch_openml("segment", version=1)),
    (fetch_openml("spambase", version=1)),
    (fetch_openml("sylvine", version=1)),
    (fetch_openml("pendigits", version=1)),
    (fetch_openml("letter", version=1)),
    (fetch_openml("madeline", version=1)),
    (fetch_openml("philippine", version=1)),
    (fetch_openml("volkert", version=1)),
    (fetch_openml("jannis", version=1)),
    (fetch_openml("gina_prior", version=1)),
    (fetch_openml("fabert", version=1)),
    (fetch_openml("USPS", version=2)),
    (fetch_openml("gina", version=1)),
    (fetch_openml("dilbert", version=1)),
    (fetch_openml("SVHN_small", version=1)),
    (fetch_openml("gisette", version=2)),
]

for data in data_ens:
    data_name = data.details["name"]
    print(data_name)
    X, y = data.data, data.target
    if not os.path.exists(f"{PATH}/{data_name}"):
        os.mkdir(f"{PATH}/{data_name}")

    if data_name == "clean1":
        X = X[:, 2:]

    load = os.path.exists(f"{PATH}/{data_name}/search_max_feature")
    if not load:
        os.mkdir(f"{PATH}/{data_name}/search_max_feature")

    clf = ForestRKClassifier(n_estimators=100,
                             criterion="entropy",
                             max_features=None,
                             random_state=RANDOM_STATE)

    exp_max_feature = Experiment(X, y,
                                 n_replications=20,
                                 classifier=clf,
                                 path=f"{PATH}/{data_name}/search_max_feature",
                                 random_state=RANDOM_STATE,
                                 load=load)

    exp_max_feature.search_max_feature(n_estimators=200,
                                       save=True)

    load = os.path.exists(f"{PATH}/{data_name}/features")
    if not load:
        os.mkdir(f"{PATH}/{data_name}/features")

    clf = ForestRKClassifier(n_estimators=100,
                             criterion="entropy",
                             max_features=None,
                             random_state=RANDOM_STATE)

    exp_features = Experiment(X, y,
                              n_replications=1,
                              classifier=clf,
                              path=f"{PATH}/{data_name}/features",
                              load=load)

    exp_features.build_models()
    exp_features.get_feature_importances(save=True)
    exp_features.get_feature_impurity_gain(save=True)

    for criterion in ("gini", "entropy"):
        if not os.path.exists(f"{PATH}/{data_name}/{criterion}"):
            os.mkdir(f"{PATH}/{data_name}/{criterion}")

        rf_random = ForestRKClassifier(n_estimators=N_ESTIMATORS,
                                       criterion=criterion,
                                       max_features="random",
                                       random_state=RANDOM_STATE)

        rf_sqrt = ForestRKClassifier(n_estimators=N_ESTIMATORS,
                                     criterion=criterion,
                                     max_features="sqrt",
                                     random_state=RANDOM_STATE)

        rf_log2 = ForestRKClassifier(n_estimators=N_ESTIMATORS,
                                     criterion=criterion,
                                     max_features="log2",
                                     random_state=RANDOM_STATE)

        rf_1 = ForestRKClassifier(n_estimators=N_ESTIMATORS,
                                  criterion=criterion,
                                  max_features=1,
                                  random_state=RANDOM_STATE)

        for clf, clf_name in ((rf_random, "rf_random"),
                              (rf_sqrt, "rf_sqrt"),
                              (rf_log2, "rf_log2"),
                              (rf_1, "rf_1")):
            print(f"{data_name}, {criterion}, {clf_name}")

            temp_path = f"{PATH}/{data_name}/{criterion}/{clf_name}"

            load = os.path.exists(temp_path)
            exp = Experiment(X, y,
                             classifier=clf,
                             n_replications=10,
                             train_size=2 / 3,
                             random_state=RANDOM_STATE,
                             path=temp_path,
                             load=load)

            exp.build_models()
            if not os.path.exists(f"{temp_path}/results/score.json"):
                exp.get_score(voting="hard", save=True)
            if not os.path.exists(f"{temp_path}/results/feature_importance.json"):
                exp.get_feature_importances(save=True)
            if not os.path.exists(f"{temp_path}/results/leaves_structures.json"):
                exp.get_leaves_structure(save=True)
