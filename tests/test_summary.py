import unittest

from regression.summary import Summary


class TestSummary(unittest.TestCase):
    def testSummary(self):
        s = Summary()
        s.setTitle("Test title")
        s.append("aa", 123)
        s.append("bb", 234)
        res = s.get_summary()
        self.assertIsNotNone(res)


if __name__ == '__main__':
  unittest.main()
