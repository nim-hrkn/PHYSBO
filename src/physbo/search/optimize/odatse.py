# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import copy
import os

import numpy as np


import odatse
import odatse.solver.function


def default_alg_dict(
    min_X: np.ndarray, max_X: np.ndarray, algorithm_name: str = "mapper"
):
    """
    Return the default algorithm parameter settings for the given algorithm name.

    Parameters
    ----------
    algorithm_name : str, optional (default: "mapper")
        The name of the algorithm to use.
        "exchange": Exchange Monte Carlo
        "pamc": Population Annealing Monte Carlo
        "minsearch": Nelder-Mead method
        "mapper": Grid search
        "bayes": Bayesian optimization

    Returns
    -------
    dict
        The default algorithm parameter settings for the given algorithm name.
    """

    # plain Python lists: ODAT-SE 4's MeshIterator computes `[1] + num_list`, which is a list
    # concatenation for a list but an element-wise addition for a numpy array (the mesh then
    # collapses onto a diagonal and the mapper evaluates only num_list[0] distinct points)
    min_X = np.array(min_X, dtype=float).tolist()
    max_X = np.array(max_X, dtype=float).tolist()
    dim = len(min_X)
    L_X = np.array(max_X) - np.array(min_X)
    d_X = (L_X / 100).tolist()

    if algorithm_name == "exchange":
        return {
            "name": "exchange",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "step_list": d_X,
            },
            "exchange": {
                "numsteps": 1000,
                "numsteps_exchange": 10,
                "Tmin": 0.1,
                "Tmax": 10.0,
                "Tlogspace": True,
                "nreplica_per_proc": 10,
            },
        }
    elif algorithm_name == "pamc":
        return {
            "name": "pamc",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "step_list": d_X,
            },
            "pamc": {
                "numsteps_annealing": 10,
                "Tnum": 11,
                "Tmax": 10.0,
                "Tmin": 0.1,
                "Tlogspace": True,
                "nreplica_per_proc": 100,
                "resampling_interval": 2,
                "fix_num_replicas": True,
            },
        }
    elif algorithm_name == "minsearch":
        return {
            "name": "minsearch",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "unit_list": d_X,
            },
            "minimize": {"maxiter": 100},
        }
    elif algorithm_name == "mapper":
        return {
            "name": "mapper",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "num_list": [11] * dim,
            },
        }
    elif algorithm_name == "bayes":
        return {
            "name": "bayes",
            "seed": 12345,
            "param": {
                "min_list": min_X,
                "max_list": max_X,
                "num_list": 21 * np.ones(dim, dtype=int),
            },
            "bayes": {
                "random_max_num_probes": 10,
                "bayes_max_num_probes": 40,
                "score": "EI",
                "num_rand_basis": 0,
            },
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


class Optimizer:
    def __init__(self, alg_dict):
        """
        Parameters
        ----------
        alg_dict : dict
            Dictionary specifying the optimization algorithm and its parameters.
        """
        self.alg_dict = copy.deepcopy(alg_dict)

    def __call__(self, fn, mpicomm=None):
        """
        Optimize a function using ODATSE (Open-source Data Assimilation and Topological Structure Exploration)

        Parameters
        ----------
        fn : callable
            The objective function to minimize. Should take a numpy array of shape (dim,) and return a float.
        min_X : numpy.ndarray
            Lower bounds for each dimension
        max_X : numpy.ndarray
            Upper bounds for each dimension
        mpicomm : MPI.Comm, optional
            MPI communicator for parallel optimization. If None, runs in serial.

        Returns
        -------
        numpy.ndarray
            The optimal point found by the optimizer
        """

        dim = len(self.alg_dict["param"]["min_list"])

        base_dict = {
            "dimension": dim,
            "root_dir": ".",
            "output_dir": "odatse_output",
        }

        info_dict = {"base": base_dict, "algorithm": self.alg_dict, "solver": {}}

        info = odatse.Info(info_dict)
        solver = odatse.solver.function.Solver(info)
        solver.set_function(
            lambda x: -fn(x)
        )  # ODATSE minimizes the function, so we need to negate the objective function
        runner = odatse.Runner(solver, info)
        alg_module = odatse.algorithm.choose_algorithm(self.alg_dict["name"])

        try:
            # try development version of odatse
            alg = alg_module.Algorithm(info, runner, mpicomm=mpicomm)
        except TypeError:
            alg = alg_module.Algorithm(info, runner)

        result = alg.main()

        if isinstance(result, dict) and "x" in result:
            # every algorithm of ODAT-SE 4 (mapper included) returns the best point
            X = np.asarray(result["x"], dtype=float)
        elif self.alg_dict.get("name") == "mapper":
            # ODAT-SE 3: mapper returns nothing useful; read the best mesh point from ColorMap.txt
            X, _fx = read_colormap_minimum(
                os.path.join(base_dict["output_dir"], "ColorMap.txt"), dim
            )
        else:
            X = result["x"]

        return X.reshape(1, -1)


def read_colormap_minimum(path, dim):
    """
    Read the point with the smallest function value from a ColorMap.txt written by the mapper algorithm.

    ODAT-SE 4 writes a header line ``#x1 x2 ... fval`` first; ODAT-SE 3 writes none.
    Header and comment lines (starting with ``#``) and empty lines are skipped.

    Parameters
    ----------
    path : str
        Path of ColorMap.txt (each data line: the ``dim`` coordinates followed by the function value).
    dim : int
        Dimension of the search space.

    Returns
    -------
    X : numpy.ndarray
        Coordinates of the minimum, shape (dim,).
    fx : float
        The minimum function value.
    """
    X = np.zeros(dim)
    fx = np.inf
    with open(path) as f:
        for line in f:
            words = line.split()
            if not words or words[0].startswith("#"):
                continue
            value = float(words[-1])
            if value < fx:
                fx = value
                X = np.array([float(x) for x in words[0:-1]])
    if not np.isfinite(fx):
        raise ValueError(f"no data lines in {path}")
    return X, fx
