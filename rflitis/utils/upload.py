import os
import numpy as np
import json
import joblib as jl

from warnings import warn


def _check_type_object(obj, type_target):
    if not isinstance(obj, type_target):
        raise TypeError(f"'obj' parameter should be of type {type_target}: "
                        f"type(obj)={type(obj)}.")


def _check_file_not_exist(filename):
    if os.path.isfile(filename):
        raise FileExistsError(f"'{filename}' already exist: "
                              f"change your saving path.")


def check_path(path):
    _check_type_object(path, str)

    if os.path.exists(path) is False:
        os.mkdir(f"{path}")
        warn(f"'{path}' folder was created because it did not exist yet.")


def save_dict(dictionary, filename, overwrite=False):
    _check_type_object(dictionary, dict)
    if not overwrite:
        _check_file_not_exist(filename)

    with open(filename, 'w') as file:
        json.dump(obj=dictionary, fp=file)


def save_array(array, filename, overwrite=False):
    _check_type_object(array, np.ndarray)
    if not overwrite:
        _check_file_not_exist(filename)

    np.save(filename, array)


def save_model(model, filename, overwrite=False):
    # TODO : Find a way to check if 'model' can be saved in joblib format
    if not overwrite:
        _check_file_not_exist(filename)

    jl.dump(value=model, filename=filename, compress=True)


def load_dict(filename):
    with open(filename) as file:
        dictionary = json.load(file)

    return dictionary


def load_array(filename):
    return np.load(filename)


def load_model(filename):
    return jl.load(filename)
