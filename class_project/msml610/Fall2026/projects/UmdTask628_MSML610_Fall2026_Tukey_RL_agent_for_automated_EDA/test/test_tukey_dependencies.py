"""
Check that every pinned dependency imports and exposes the API Tukey uses.

This is integration gate G1: a library upgrade that renames a function or a
keyword argument fails here, before any experiment depends on it.

Import as:

import test.test_tukey_dependencies as ttetudep
"""

import inspect
import logging
from typing import Any, List

import CausalDisco.analytics as cdanaly
import CausalDisco.baselines as cdbasel
import causallearn.graph.SHD as cgrshd
import causallearn.search.ConstraintBased.FCI as cscofci
import causallearn.search.ConstraintBased.PC as cscopc
import causallearn.search.ScoreBased.GES as cscoges
import causallearn.utils.DAG2CPDAG as cudcpdag
import causallearn.utils.DAG2PAG as cudpag
import lingam
import pgmpy.models as pgmodel
import sklearn.datasets as skdata
import torch

import helpers.hunit_test as hunitest

_LOG = logging.getLogger(__name__)


def _get_missing_params(func: Any, required: List[str]) -> str:
    """
    List the required parameter names that `func` does not accept.

    :param func: callable to inspect
    :param required: parameter names Tukey passes to `func`
    :return: sorted list of missing names, as a string (e.g., "[]")
    """
    params = set(inspect.signature(func).parameters)
    missing = sorted(set(required) - params)
    return str(missing)


# #############################################################################
# Test_causal_learn_api
# #############################################################################


class Test_causal_learn_api(hunitest.TestCase):
    """
    Test that the causal-learn functions used by Tukey accept our arguments.
    """

    def test1(self) -> None:
        """
        Test `pc()` accepts the arguments the baseline wrapper passes.
        """
        # Prepare inputs.
        required = ["data", "alpha", "indep_test", "show_progress"]
        # Run test.
        actual = _get_missing_params(cscopc.pc, required)
        # Check outputs.
        self.assert_equal(actual, "[]")

    def test2(self) -> None:
        """
        Test `ges()`, `fci()`, `dag2cpdag()`, `dag2pag()` and `SHD`.
        """
        # Prepare inputs.
        checks = [
            (cscoges.ges, ["X", "score_func"]),
            (cscofci.fci, ["dataset", "independence_test_method"]),
            (cudcpdag.dag2cpdag, ["G"]),
            (cudpag.dag2pag, ["dag", "islatent"]),
        ]
        # Run test.
        actual = [_get_missing_params(func, req) for func, req in checks]
        actual.append(str(callable(cgrshd.SHD)))
        # Check outputs.
        self.assert_equal(str(actual), "['[]', '[]', '[]', '[]', 'True']")


# #############################################################################
# Test_lingam_api
# #############################################################################


class Test_lingam_api(hunitest.TestCase):
    """
    Test that `lingam.DirectLiNGAM` exposes the API the baseline uses.
    """

    def test1(self) -> None:
        """
        Test `DirectLiNGAM` has a callable `fit()`.
        """
        # Prepare inputs.
        model = lingam.DirectLiNGAM()
        # Run test.
        actual = str(callable(getattr(model, "fit", None)))
        # Check outputs.
        self.assert_equal(actual, "True")


# #############################################################################
# Test_causal_disco_api
# #############################################################################


class Test_causal_disco_api(hunitest.TestCase):
    """
    Test the released CausalDisco 0.2.4 functions Tukey relies on.
    """

    def test1(self) -> None:
        """
        Test the sortability and sort-regress functions exist.
        """
        # Prepare inputs.
        analytics = ["var_sortability", "r2_sortability"]
        baselines = [
            "var_sort_regress",
            "r2_sort_regress",
            "random_sort_regress",
        ]
        # Run test.
        missing = [name for name in analytics if not hasattr(cdanaly, name)]
        missing += [name for name in baselines if not hasattr(cdbasel, name)]
        actual = str(missing)
        # Check outputs.
        self.assert_equal(actual, "[]")


# #############################################################################
# Test_pgmpy_api
# #############################################################################


class Test_pgmpy_api(hunitest.TestCase):
    """
    Test the pgmpy 1.1.2 keyword arguments used by the `pgmpy_lg` family.
    """

    def test1(self) -> None:
        """
        Test `get_random()` and `simulate()` accept our keyword arguments.
        """
        # Prepare inputs.
        model_cls = pgmodel.LinearGaussianBayesianNetwork
        get_random_args = ["n_nodes", "n_edges", "loc", "scale", "seed"]
        simulate_args = ["n_samples", "seed"]
        # Run test.
        actual = [
            _get_missing_params(model_cls.get_random, get_random_args),
            _get_missing_params(model_cls.simulate, simulate_args),
        ]
        # Check outputs.
        self.assert_equal(str(actual), "['[]', '[]']")


# #############################################################################
# Test_torch_build
# #############################################################################


class Test_torch_build(hunitest.TestCase):
    """
    Test that the image runs the CPU-only torch build.
    """

    def test1(self) -> None:
        """
        Test the torch version is a `+cpu` build and CUDA is unavailable.
        """
        # Prepare inputs.
        version = torch.__version__
        # Run test.
        actual = str([version.endswith("+cpu"), torch.cuda.is_available()])
        # Check outputs.
        self.assert_equal(actual, "[True, False]")


# #############################################################################
# Test_sklearn_api
# #############################################################################


class Test_sklearn_api(hunitest.TestCase):
    """
    Test the scikit-learn generators and datasets Tukey calls.
    """

    def test1(self) -> None:
        """
        Test `make_regression()` and `load_diabetes()` accept our arguments.
        """
        # Prepare inputs.
        make_regression_args = [
            "n_samples",
            "n_features",
            "n_informative",
            "n_targets",
            "noise",
            "shuffle",
            "coef",
            "random_state",
        ]
        # Run test.
        actual = [
            _get_missing_params(skdata.make_regression, make_regression_args),
            _get_missing_params(skdata.load_diabetes, ["scaled"]),
        ]
        # Check outputs.
        self.assert_equal(str(actual), "['[]', '[]']")
