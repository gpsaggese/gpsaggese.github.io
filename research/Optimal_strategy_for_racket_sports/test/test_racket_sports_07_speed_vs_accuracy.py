"""
Run the `racket_sports_07_speed_vs_accuracy` notebook top to bottom.

Import as:

import research.Optimal_strategy_for_racket_sports.test.test_racket_sports_07_speed_vs_accuracy as rosfrsttrs0sva
"""

import logging
import os

import helpers.hserver as hserver
import helpers.hsystem as hsystem
import helpers.hunit_test as hunitest
import pytest

_LOG = logging.getLogger(__name__)


# #############################################################################
# Test_notebook
# #############################################################################


@pytest.mark.skipif(
    not hserver.is_inside_docker(),
    reason=(
        "Run inside the image with ./docker_cmd.sh 'pip install -q pytest &&"
        " python -m pytest test/test_racket_sports_07_speed_vs_accuracy.py'"
    ),
)
class Test_notebook(hunitest.TestCase):
    """
    Run `racket_sports_07_speed_vs_accuracy.ipynb`, the reduced run.
    """

    @pytest.mark.superslow
    def test1(self) -> None:
        """
        Test that the notebook runs top to bottom, in about a minute.
        """
        # Prepare inputs.
        notebook_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "notebooks",
        )
        scratch_dir = self.get_scratch_space()
        cmd = (
            f"cd {notebook_dir} && jupyter nbconvert --to notebook --execute"
            " --ExecutePreprocessor.timeout=900"
            f" --output-dir {scratch_dir} --output run"
            " racket_sports_07_speed_vs_accuracy.ipynb"
        )
        # Run test.
        hsystem.system(cmd)
        # Check outputs.
        self.assertTrue(os.path.exists(os.path.join(scratch_dir, "run.ipynb")))
