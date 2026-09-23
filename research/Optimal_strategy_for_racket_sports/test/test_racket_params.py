"""Tests for racket_params module."""

import dataclasses

import numpy as np
import pytest

from research.Optimal_strategy_for_racket_sports import racket_params


class TestCourtGeometry:
    """Tests for CourtGeometry dataclass."""

    def test_valid_court(self):
        court = racket_params.CourtGeometry(
            length_m=23.77,
            width_m=8.23,
            net_height_center_m=0.914,
            net_height_post_m=1.07,
            non_volley_zone_m=0.0,
            service_line_m=6.40,
        )
        assert court.length_m == 23.77
        assert court.width_m == 8.23

    def test_invalid_length(self):
        with pytest.raises(ValueError):
            racket_params.CourtGeometry(
                length_m=-1,
                width_m=8.23,
                net_height_center_m=0.914,
                net_height_post_m=1.07,
            )

    def test_frozen(self):
        court = racket_params.CourtGeometry(
            length_m=23.77,
            width_m=8.23,
            net_height_center_m=0.914,
            net_height_post_m=1.07,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            court.length_m = 24.0


class TestCourtRegion:
    """Tests for CourtRegion dataclass."""

    def test_valid_region(self):
        region = racket_params.CourtRegion(x_min=-1, x_max=1, y_min=0, y_max=2)
        assert region.x_min == -1
        assert region.x_max == 1

    def test_invalid_x_bounds(self):
        with pytest.raises(ValueError):
            racket_params.CourtRegion(x_min=1, x_max=1, y_min=0, y_max=2)

    def test_contains_interior(self):
        region = racket_params.CourtRegion(x_min=-1, x_max=1, y_min=0, y_max=2)
        assert region.contains(np.array([0.0]), np.array([1.0]))[0]

    def test_contains_boundary(self):
        region = racket_params.CourtRegion(x_min=-1, x_max=1, y_min=0, y_max=2)
        assert region.contains(np.array([-1.0]), np.array([0.0]))[0]
        assert region.contains(np.array([1.0]), np.array([2.0]))[0]

    def test_contains_outside(self):
        region = racket_params.CourtRegion(x_min=-1, x_max=1, y_min=0, y_max=2)
        assert not region.contains(np.array([2.0]), np.array([1.0]))[0]
        assert not region.contains(np.array([0.0]), np.array([3.0]))[0]

    def test_contains_vectorized(self):
        region = racket_params.CourtRegion(x_min=-1, x_max=1, y_min=0, y_max=2)
        x = np.array([-1, 0, 1, 2])
        y = np.array([1, 1, 1, 1])
        result = region.contains(x, y)
        assert np.array_equal(result, [True, True, True, False])


class TestGetNetHeight:
    """Tests for get_net_height function."""

    def test_center_height(self):
        court = racket_params.TENNIS.court
        height = racket_params.get_net_height(court, np.array([0.0]))
        assert np.isclose(height[0], court.net_height_center_m)

    def test_post_height(self):
        court = racket_params.TENNIS.court
        x_post = court.width_m / 2
        height = racket_params.get_net_height(court, np.array([x_post]))
        assert np.isclose(height[0], court.net_height_post_m)

    def test_symmetry(self):
        court = racket_params.TENNIS.court
        x = np.array([2.0])
        height_pos = racket_params.get_net_height(court, x)
        height_neg = racket_params.get_net_height(court, -x)
        assert np.isclose(height_pos[0], height_neg[0])

    def test_linear_interpolation(self):
        court = racket_params.TENNIS.court
        x_post = court.width_m / 2
        x_half = x_post / 2
        height_half = racket_params.get_net_height(court, np.array([x_half]))
        expected = court.net_height_center_m + 0.5 * (
            court.net_height_post_m - court.net_height_center_m
        )
        assert np.isclose(height_half[0], expected)

    def test_vectorized(self):
        court = racket_params.TENNIS.court
        x = np.array([0.0, 1.0, 2.0])
        heights = racket_params.get_net_height(court, x)
        assert len(heights) == 3
        assert heights[0] == court.net_height_center_m


class TestGetHalfCourtRegion:
    """Tests for get_half_court_region function."""

    def test_tennis_half_court(self):
        region = racket_params.get_half_court_region(racket_params.TENNIS.court)
        assert region.x_min == -racket_params.TENNIS.court.width_m / 2
        assert region.x_max == racket_params.TENNIS.court.width_m / 2
        assert region.y_min == 0
        assert region.y_max == racket_params.TENNIS.court.length_m / 2

    def test_pickleball_half_court(self):
        region = racket_params.get_half_court_region(
            racket_params.PICKLEBALL.court
        )
        assert region.x_min == -racket_params.PICKLEBALL.court.width_m / 2
        assert region.x_max == racket_params.PICKLEBALL.court.width_m / 2
        assert region.y_min == 0
        assert region.y_max == racket_params.PICKLEBALL.court.length_m / 2


class TestGetServiceBoxRegion:
    """Tests for get_service_box_region function."""

    def test_tennis_deuce(self):
        region = racket_params.get_service_box_region(
            racket_params.TENNIS.court, "deuce"
        )
        assert region.x_min == -racket_params.TENNIS.court.width_m / 2
        assert region.x_max == 0
        assert region.y_min == 0
        assert region.y_max == racket_params.TENNIS.court.service_line_m

    def test_tennis_ad(self):
        region = racket_params.get_service_box_region(
            racket_params.TENNIS.court, "ad"
        )
        assert region.x_min == 0
        assert region.x_max == racket_params.TENNIS.court.width_m / 2
        assert region.y_min == 0
        assert region.y_max == racket_params.TENNIS.court.service_line_m

    def test_pickleball_deuce(self):
        region = racket_params.get_service_box_region(
            racket_params.PICKLEBALL.court, "deuce"
        )
        assert region.x_min == -racket_params.PICKLEBALL.court.width_m / 2
        assert region.x_max == 0
        assert region.y_min == racket_params.PICKLEBALL.court.non_volley_zone_m
        assert region.y_max == racket_params.PICKLEBALL.court.service_line_m

    def test_invalid_serve_side(self):
        with pytest.raises(ValueError):
            racket_params.get_service_box_region(
                racket_params.TENNIS.court, "invalid"
            )


class TestPresets:
    """Tests for preset sport and player parameters."""

    def test_tennis_params(self):
        assert racket_params.TENNIS.name == "Tennis"
        assert racket_params.TENNIS.court.length_m == 23.77
        assert racket_params.TENNIS.court.width_m == 8.23
        assert racket_params.TENNIS.court.service_line_m == 6.40

    def test_pickleball_params(self):
        assert racket_params.PICKLEBALL.name == "Pickleball"
        assert racket_params.PICKLEBALL.court.length_m == 13.41
        assert racket_params.PICKLEBALL.court.non_volley_zone_m == 2.13

    def test_default_player(self):
        assert racket_params.DEFAULT_PLAYER.reaction_time_s == 0.2
        assert racket_params.DEFAULT_PLAYER.move_speed_mps == 1.5
        assert racket_params.DEFAULT_PLAYER.contact_height_m == 1.0

    def test_default_error(self):
        assert np.isclose(
            racket_params.DEFAULT_ERROR.sigma_theta_rad, np.radians(1.5)
        )
