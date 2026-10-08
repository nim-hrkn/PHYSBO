# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2025- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pytest

physbo = pytest.importorskip("physbo")
pytest.importorskip("odatse")

from physbo.search.optimize.odatse import read_colormap_minimum  # noqa: E402


def test_read_colormap_minimum_skips_header(tmp_path):
    """ODAT-SE 4 writes '#x1 x2 fval' as the first line of ColorMap.txt (ODAT-SE 3 wrote no header)."""
    path = tmp_path / "ColorMap.txt"
    path.write_text(
        "#x1 x2 fval\n"
        "-2.000000 -2.000000 8.000000\n"
        "\n"
        "0.000000 1.000000 1.000000\n"
        "# a comment\n"
        "0.500000 -0.500000 0.500000\n"
        "2.000000 2.000000 8.000000\n"
    )
    X, fx = read_colormap_minimum(str(path), 2)
    assert fx == pytest.approx(0.5)
    assert np.allclose(X, [0.5, -0.5])


def test_read_colormap_minimum_without_header(tmp_path):
    path = tmp_path / "ColorMap.txt"
    path.write_text("0.0 0.0 3.0\n1.0 0.0 -1.0\n")
    X, fx = read_colormap_minimum(str(path), 2)
    assert fx == -1.0 and np.allclose(X, [1.0, 0.0])


def test_read_colormap_minimum_empty(tmp_path):
    path = tmp_path / "ColorMap.txt"
    path.write_text("#x1 fval\n")
    with pytest.raises(ValueError):
        read_colormap_minimum(str(path), 1)


@pytest.mark.parametrize("name", ["exchange", "pamc", "minsearch", "mapper", "bayes"])
def test_default_alg_dict_uses_plain_lists(name):
    """ODAT-SE 4 does `[1] + num_list` (list concatenation); a numpy array would be added element-wise."""
    from physbo.search.optimize.odatse import default_alg_dict

    d = default_alg_dict(np.array([-2.0, -2.0]), np.array([2.0, 2.0]), name)
    param = d["param"]
    for key in ("min_list", "max_list", "num_list", "step_list"):
        if key in param:
            assert isinstance(param[key], list), f"{name}: {key} is {type(param[key]).__name__}"
            assert len(param[key]) == 2
    if name == "mapper":
        assert param["num_list"] == [11, 11]
