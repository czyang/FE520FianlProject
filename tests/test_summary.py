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

    def testSetTitle(self):
        s = Summary()
        s.setTitle("Test title")
        self.assertEqual("Test title", s.title)

    def testAppendDate(self):
        s = Summary()
        s.appendDate()
        self.assertIsNotNone(s.dataTable)

    def testAppendTime(self):
        s = Summary()
        s.appendTime()
        self.assertIsNotNone(s.dataTable)

    def testGet_summary(self):
        s = Summary()
        s.appendTime()
        self.assertIsNotNone(s.get_summary())

if __name__ == '__main__':
  unittest.main()
