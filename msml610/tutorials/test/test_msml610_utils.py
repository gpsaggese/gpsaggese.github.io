import helpers.hunit_test as hunitest
import msml610.tutorials.L07_01_bayesian_coin_utils as mtl0bcout

import numpy as np


# #############################################################################
# Test_loss1
# #############################################################################


class Test_loss1(hunitest.TestCase):
    def test_squared_loss1(self) -> None:
        y = 3
        yhat = 1
        loss = mtl0bcout.squared_loss(y, yhat)
        self.assertEqual(loss, 4)

    def test_squared_loss2(self) -> None:
        y = np.array([3, 4])
        yhat = np.array([1, 2])
        loss = mtl0bcout.squared_loss(y, yhat)
        self.assert_equal(loss, [4, 4])

    def test_squared_loss3(self) -> None:
        y = np.array([3, 4])
        yhat = 1
        loss = mtl0bcout.squared_loss(y, yhat)
        self.assert_equal(loss, [4, 9])
