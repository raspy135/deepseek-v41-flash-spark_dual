import unittest

from server.app import APIError, parse_sampling


class OutputBudgetTests(unittest.TestCase):
    def test_default_and_null_use_128k(self):
        for body in ({}, {'max_tokens': None}, {'max_completion_tokens': None}):
            self.assertEqual(parse_sampling(body)['max_tokens'], 131072)

    def test_explicit_limits_and_precedence(self):
        self.assertEqual(parse_sampling({'max_tokens': 32})['max_tokens'], 32)
        self.assertEqual(parse_sampling({'max_completion_tokens': 65536})['max_tokens'], 65536)
        self.assertEqual(parse_sampling({'max_tokens': 4096, 'max_completion_tokens': 32768})
                         ['max_tokens'], 32768)

    def test_invalid_limits_rejected(self):
        for value in (0, -1, True, '131072', 1.5):
            with self.assertRaises(APIError):
                parse_sampling({'max_completion_tokens': value})


if __name__ == '__main__':
    unittest.main()
