# Common

- Infrastructure shared by the MVP components: settings, database pool, migrations,
  and demo seed data
- `common/` imports no other Noesis component

## Structure of the Dir

| Dir           | Description                                                    |
| :------------ | :------------------------------------------------------------- |
| `migrations/` | `NNN_name.sql` schema files, applied once each, in order, by `noesis_db.py` |
| `test/`       | Unit tests; `noesis_test_utils.py` has the shared Postgres test helpers |

## Description of Files

| File                | Description                                                         | Cluster  |
| :------------------ | :------------------------------------------------------------------ | :------- |
| `noesis_db.py`      | Async `psycopg` 3 connection pool and migrations runner             | Database |
| `noesis_seed.py`    | Loads `seed.yaml` into Postgres; hashes API keys with SHA-256       | Database |
| `noesis_settings.py` | `Settings` from `research/Noesis/.env` and the environment; fails loudly on bad values | Settings |
| `seed.yaml`         | The 5 demo sellers (from `docs/findings/openrouter_spike.md`) and accounts | Seed |
| `migrations/001_init.sql` | 10-table MVP schema       | Schema   |
| `migrations/002_checker.sql` | Checker tables: banks, probes, calibration, evidence       | Schema   |

## Description of Architecture

- `COMMON_DIR` (this dir) anchors `migrations/` and `seed.yaml`; `NOESIS_DIR` is its
  parent, `research/Noesis`, which holds `.env`
- Tests that need Postgres connect to `NOESIS_TEST_DATABASE_URL` and skip when it is
  not reachable (see `docs/onboarding.md`)
