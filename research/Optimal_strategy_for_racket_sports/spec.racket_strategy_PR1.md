# `racket_strategy_PR1` Implementation Spec: Scaffold and Sport Parameters

- `PR1` of `plan.racket_strategy.md`: create the project skeleton and the parameter
  layer that every later PR imports
- Scope: Docker scaffold from `class_project/project_template/`, and
  `racket_params.py` holding the model $\mathcal{M}$ of paper Section VII
- Roadmap position: depends on nothing; `PR2`-`PR7` import `racket_params.py`
- This is a specification only: no code in this document has been implemented

## Design Decisions

- **Court-fixed frame**: one frame in meters for every court-plane quantity
  - Fact: the paper puts the origin at the striker contact point (Section III-A)
  - Origin at the net center on the ground; `x` lateral (0 on the center line);
    `y` along the court, striker side `y < 0`, returner side `y > 0`
  - Why: cells, bounds, opponent, service boxes are all court-fixed; `PR2` keeps
    the paper's local frame ($S$ at origin) for the vertical plane only
- **Split the model tuple by owner**: `SportParams`, `ShotErrorModel`, and
  `PlayerParams` (which holds a `ShotErrorModel`), fields in the interface below
  - Why: asymmetric striker and returner (Section VIII) need no new code
- **Net profile**: `get_net_height()` is linear in $|x|$ from the center height to
  the post height at the singles sideline
  - Why: Table I gives both; a constant center height is optimistic for wide shots
- **Presets separate facts from illustrative values**
  - `TENNIS`, `PICKLEBALL`: Table I values plus the service line (rulebook facts)
    - Tennis: service line 6.40 m from the net, no non-volley zone
    - Pickleball: service court from the non-volley zone (2.13 m) to the baseline
  - `DEFAULT_PLAYER`: $t_r = 0.2$ s, $v_p = 1.5$ m/s (Section VI), $h_0 = 1$ m
  - `DEFAULT_ERROR`: $\sigma_\theta = \sigma_\phi = 1.5$ deg, $\sigma_v = 0.05$,
    mirroring Figure 1; the docstring marks them "not calibrated"

## Trade-off and Alternative Design

- YAML file or plain `dict`: rejected, they add I/O or lose type hints and
  construction-time validation
- Trade-off: frozen dataclasses are verbose but make every experiment in `PR7` an
  explicit, hashable parameter set (`dataclasses.replace()` for sweeps)

## Out of Scope

- Doubles geometry: paper future work (v)
- Direction-dependent movement speed and acceleration (Section VIII)
- Ball radius: the paper models a point-mass ball (Section III-B)

## Current State

- `research/Optimal_strategy_for_racket_sports/` is empty
- `papers/Optimal_strategy_for_racket_sports/figures/make_figures.py:30-37`
  hard-codes $g$, $h_0$, net distance, and net height; `PR8` handles it
- `class_project/project_template/`: Docker scripts; `requirements.txt` lacks `scipy`

## Implementation

- Scaffold (new files):
  - Copy `Dockerfile`, `docker_*.sh`, `bashrc`, `etc_sudoers`, `run_jupyter.sh`,
    `utils.sh`, `version.sh`; set `IMAGE_NAME=umd_racket_strategy`
  - `requirements.txt`: template packages plus `scipy` (used by `PR4`)
  - `README.md`: quick start plus the module table, updated by each PR

### `racket_params.py`

- Interface (all dataclasses frozen, validated in `__post_init__` with `hdbg`):
  ```python
  GRAVITY_MPS2 = 9.81
  class CourtGeometry:  # length_m, width_m, net_height_center_m,
      ...               # net_height_post_m, non_volley_zone_m, service_line_m
  class CourtRegion:    # x_min, x_max, y_min, y_max
      def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...
  class SportParams:    # name, court, max_ball_speed_mps
  class ShotErrorModel: # sigma_theta_rad, sigma_v_frac, sigma_phi_rad
  class PlayerParams:   # reaction_time_s, move_speed_mps, contact_height_m, error
  def get_net_height(court: CourtGeometry, x: np.ndarray) -> np.ndarray: ...
  def get_half_court_region(court: CourtGeometry) -> CourtRegion: ...
  def get_service_box_region(court: CourtGeometry, serve_side: str) -> CourtRegion:
      # "deuce" -> x in [-W/2, 0], "ad" -> x in [0, W/2].
  ```

## Interaction with Existing Code

- None: new directory; the Docker image name does not clash with the template

## Configuration and Secrets

- New Docker image `gpsaggese/umd_racket_strategy`; no secrets

## Unit Test Plan

- `test/test_racket_params.py`:
  - `TestCourtRegion`: `contains()` on interior, boundary, and outside points
  - `Test_get_net_height`: center, sideline, and symmetry in `x`
  - `Test_get_service_box_region`: tennis deuce bounds; pickleball box after NVZ

## Risks and Limitations to Call Out

- Deuce and ad sides are easy to flip: the reviewer checks against a court sketch
- Readers may take the illustrative defaults as calibrated values

## Result (to Fill in Once Implemented)

- TBD
