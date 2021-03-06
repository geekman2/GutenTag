# coding: utf-8
# ------------------------------------------------------------------------------
# Name:        module1
# Purpose:
# Author:      Devon Muraoka
# Created:
# Copyright:   (c) Devon Muraoka
# ------------------------------------------------------------------------------
from __future__ import absolute_import
import unittest
import WordVectors.lnFilter as lnFilter


class TestlnFilter(unittest.TestCase):
    def setup(self):
        test1 = u"This is an english"
        test2 = u"Ceci est français"
        test3 = u"Questo è italiano"
        test4 = u"dit is nederlands"
        test5 = u"to jest polski"
        tests = [test1, test2, test3, test4, test5]
        results = [True, False, False, False, False]
        return tests, results

    def testisEnglishLangid(self):
        tests, results = self.setup()
        testresult = []
        for item in tests:
            testresult.append(lnFilter.isEnglishLangid(item))
        self.assertEqual(results, testresult)

    def testisEnglishNltk(self):
        tests, results = self.setup()
        testresult = []
        for item in tests:
            testresult.append(lnFilter.isEnglishNltk(item))
        self.assertEqual(results, testresult)

if __name__ == '__main__':
    unittest.main()
