"""CPU contract check: collective completion is joined through its Work handle."""
import unittest
from unittest.mock import Mock
from engine.dist import EPDistributed


class TestOverlapJoin(unittest.TestCase):
    def test_join_uses_work_completion_not_user_stream(self):
        work = Mock()
        EPDistributed.finish_combine(work)
        work.block_current_stream.assert_called_once_with()
        work.wait.assert_not_called()


if __name__ == '__main__':
    unittest.main()
