import logging
import os

import psycopg

import helpers.hunit_test as hunitest
import research.Noesis.noesis_db as rnonodb
import research.Noesis.test.gateway_test_utils as rnteteut

_LOG = logging.getLogger(__name__)


def _write(dir_name: str, file_name: str, text: str) -> None:
    """
    Write `text` to `dir_name/file_name`.
    """
    with open(os.path.join(dir_name, file_name), "w") as f:
        f.write(text)


# #############################################################################
# Test_connect
# #############################################################################


class Test_connect(hunitest.TestCase):
    """
    Test opening a pool against the test Postgres.
    """

    def test1(self) -> None:
        """
        Test that the pool runs a query.
        """

        async def _body(pool) -> None:
            # Run test.
            rows = await rnteteut.fetch_all(pool, "SELECT 1 + 1")
            # Check outputs.
            self.assertEqual(rows, [(2,)])

        rnteteut.run_with_clean_db(self, _body)


# #############################################################################
# Test_apply_migrations
# #############################################################################


class Test_apply_migrations(hunitest.TestCase):
    """
    Test that migrations run once each, in order, all-or-nothing.
    """

    def test1(self) -> None:
        """
        Test that the real migrations are recorded and re-running is a no-op.
        """

        async def _body(pool) -> None:
            # Run test.
            applied_again = await rnonodb.apply_migrations(pool)
            recorded = await rnteteut.fetch_all(
                pool, "SELECT filename FROM schema_migrations"
            )
            tables = await rnteteut.fetch_all(
                pool,
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'",
            )
            # Check outputs.
            self.assertEqual(applied_again, [])
            self.assertIn(("001_init.sql",), recorded)
            for table in ["contracts", "requests", "sellers", "accounts"]:
                self.assertIn((table,), tables)

        rnteteut.run_with_clean_db(self, _body)

    def test2(self) -> None:
        """
        Test ordering, rollback of a failing file, and retry after a fix.
        """
        # Prepare inputs.
        mig_dir = self.get_scratch_space()
        files = {
            "001_a.sql": "CREATE TABLE IF NOT EXISTS mig_test_log (step TEXT);"
            "INSERT INTO mig_test_log VALUES ('a');",
            "002_b.sql": "INSERT INTO mig_test_log VALUES ('b');",
            "003_bad.sql": "INSERT INTO mig_test_log VALUES ('c');"
            "SELECT no_such_column FROM mig_test_log;",
        }
        for name, sql in files.items():
            _write(mig_dir, name, sql)
        names = sorted(files)

        async def _body(pool) -> None:
            async with pool.connection() as conn:
                await conn.execute("DROP TABLE IF EXISTS mig_test_log")
                await conn.execute(
                    "DELETE FROM schema_migrations WHERE filename = ANY(%s)",
                    (names,),
                )
            try:
                # Run test.
                with self.assertRaises(psycopg.errors.UndefinedColumn):
                    await rnonodb.apply_migrations(pool, migrations_dir=mig_dir)
                steps = await rnteteut.fetch_all(
                    pool, "SELECT step FROM mig_test_log"
                )
                recorded = await rnteteut.fetch_all(
                    pool,
                    "SELECT filename FROM schema_migrations "
                    "WHERE filename = ANY(%s) ORDER BY filename",
                    names,
                )
                # Check outputs: 001 and 002 applied, 003 rolled back entirely.
                self.assertEqual(steps, [("a",), ("b",)])
                self.assertEqual(recorded, [("001_a.sql",), ("002_b.sql",)])
                # Fix the bad file: only 003 runs now.
                _write(
                    mig_dir,
                    "003_bad.sql",
                    "INSERT INTO mig_test_log VALUES ('c');",
                )
                applied = await rnonodb.apply_migrations(
                    pool, migrations_dir=mig_dir
                )
                self.assertEqual(applied, ["003_bad.sql"])
            finally:
                async with pool.connection() as conn:
                    await conn.execute("DROP TABLE IF EXISTS mig_test_log")
                    await conn.execute(
                        "DELETE FROM schema_migrations WHERE filename = ANY(%s)",
                        (names,),
                    )

        rnteteut.run_with_clean_db(self, _body)
