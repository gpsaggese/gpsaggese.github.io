"""Tests for the kernel pool manager."""
import unittest
from unittest.mock import patch, MagicMock


class TestKernelPoolManager(unittest.TestCase):
    def test_get_main_kernel_creates_once(self):
        from backend.services.kernel_pool import KernelPoolManager
        pool = KernelPoolManager()
        with patch("backend.services.kernel_pool.get_or_create_kernel") as mock_create:
            mock_create.return_value = MagicMock()
            pool.get_main_kernel("s1")
            pool.get_main_kernel("s1")
            mock_create.assert_called_once_with("s1")

    def test_allocate_subagent_kernels(self):
        from backend.services.kernel_pool import KernelPoolManager
        pool = KernelPoolManager()
        with patch("backend.services.kernel_pool.get_or_create_kernel") as mock_create:
            mock_create.return_value = MagicMock()
            ids = pool.allocate_subagent_kernels("s1", 3)
            assert len(ids) == 3
            assert ids == ["s1_sub_0", "s1_sub_1", "s1_sub_2"]
            assert mock_create.call_count == 3

    def test_execute_on_subkernel(self):
        from backend.services.kernel_pool import KernelPoolManager
        pool = KernelPoolManager()
        with patch("backend.services.kernel_pool.get_or_create_kernel"):
            pool.allocate_subagent_kernels("s1", 1)
        with patch("backend.services.kernel_pool.execute_code") as mock_exec:
            mock_exec.return_value = ([{"text": "ok"}], None)
            outputs, error = pool.execute_on_subkernel("s1_sub_0", "print('hi')")
            assert error is None
            assert outputs[0]["text"] == "ok"

    def test_shutdown_subagent_kernels(self):
        from backend.services.kernel_pool import KernelPoolManager
        pool = KernelPoolManager()
        with patch("backend.services.kernel_pool.get_or_create_kernel"):
            pool.allocate_subagent_kernels("s1", 2)
        with patch("backend.services.kernel_pool.shutdown_kernel") as mock_shutdown:
            pool.shutdown_subagent_kernels("s1")
            assert mock_shutdown.call_count == 2
        assert pool._sub_kernels.get("s1") is None or len(pool._sub_kernels.get("s1", [])) == 0

    def test_shutdown_all(self):
        from backend.services.kernel_pool import KernelPoolManager
        pool = KernelPoolManager()
        with patch("backend.services.kernel_pool.get_or_create_kernel") as mock_create:
            mock_create.return_value = MagicMock()
            pool.get_main_kernel("s1")
            pool.allocate_subagent_kernels("s1", 1)
        with patch("backend.services.kernel_pool.shutdown_kernel") as mock_shutdown:
            pool.shutdown_all("s1")
            # Should shutdown sub kernel + main kernel = 2 calls
            assert mock_shutdown.call_count == 2

if __name__ == "__main__":
    unittest.main()
