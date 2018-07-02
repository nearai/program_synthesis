import copy
import collections
import unittest
import os

import numpy as np
import pytest

from program_synthesis.karel.dataset import mutation
from program_synthesis.karel.dataset import refine_env
from program_synthesis.karel.dataset import parser_for_synthesis
from program_synthesis.karel.dataset import executor as executor_mod


class RefineEnvTest(unittest.TestCase):
    executor = executor_mod.KarelExecutor()

    def _test_step(self, obs, reward, done, should_be_done):
        for inp, out in zip(obs['inputs'], obs['outputs']):
            result, trace = self.executor.execute(
                obs['code'], None, inp, record_trace=True)
            self.assertEqual(out, result)
        if should_be_done:
            self.assertEqual(reward, 1)
            self.assertEqual(done, True)
        else:
            self.assertEqual(reward, 0)
            self.assertEqual(done, False)

    def testCreateSimple(self):
        # DEF run m( m)
        env = refine_env.KarelRefineEnv(self.simple_input_tests)

        # DEF run m( pickMarker m)
        obs, reward, done, info = env.step((mutation.ADD_ACTION,
                                            (3, 'pickMarker')))
        self.assertEqual(obs['code'], ('DEF', 'run', 'm(', 'pickMarker', 'm)'))
        self._test_step(obs, reward, done, False)

        # DEF run m( pickMarker move m)
        obs, reward, done, info = env.step((mutation.ADD_ACTION, (4, 'move')))
        self.assertEqual(obs['code'], ('DEF', 'run', 'm(', 'pickMarker',
                                       'move', 'm)'))
        self._test_step(obs, reward, done, False)

        # DEF run m( pickMarker move pickMarker m)
        obs, reward, done, info = env.step((mutation.ADD_ACTION,
                                            (5, 'pickMarker')))
        self.assertEqual(obs['code'], ('DEF', 'run', 'm(', 'pickMarker',
                                       'move', 'pickMarker', 'm)'))
        self._test_step(obs, reward, done, True)

    def testSimpleRemoveAction(self):
        # DEF run m( m)
        env = refine_env.KarelRefineEnv(self.simple_input_tests)
        # DEF run m( pickMarker m)
        env.step((mutation.ADD_ACTION, (3, 'pickMarker')))
        # DEF run m( pickMarker move m)
        env.step((mutation.ADD_ACTION, (4, 'move')))
        # DEF run m( pickMarker m)
        obs, reward, done, info = env.step((mutation.REMOVE_ACTION, (4, )))
        self.assertEqual(obs['code'], ('DEF', 'run', 'm(', 'pickMarker', 'm)'))
        self._test_step(obs, reward, done, False)

    def testSimpleReplaceAction(self):
        # DEF run m( m)
        env = refine_env.KarelRefineEnv(self.simple_input_tests)
        # DEF run m( pickMarker m)
        env.step((mutation.ADD_ACTION, (3, 'pickMarker')))
        # DEF run m( putMarker m)
        obs, reward, done, info = env.step((mutation.REPLACE_ACTION,
                                            (3, 'putMarker')))
        self.assertEqual(obs['code'], ('DEF', 'run', 'm(', 'putMarker', 'm)'))
        self._test_step(obs, reward, done, False)

    def testCreateModerate(self):
        # DEF run m( m)
        env = refine_env.KarelRefineEnv(self.moderate_input_tests)
        # DEF run m( turnLeft m)
        obs, reward, done, info = env.step((mutation.ADD_ACTION,
                                            (3, 'turnLeft')))
        self.assertEqual(obs['code'], ('DEF', 'run', 'm(', 'turnLeft', 'm)'))
        self._test_step(obs, reward, done, False)

        # DEF run m( turnLeft move m)
        obs, reward, done, info = env.step((mutation.ADD_ACTION, (4, 'move')))
        self.assertEqual(obs['code'],
                         ('DEF', 'run', 'm(', 'turnLeft', 'move', 'm)'))
        self._test_step(obs, reward, done, False)

        # DEF run m( turnLeft move pickMarker m)
        obs, reward, done, info = env.step((mutation.ADD_ACTION,
                                            (5, 'pickMarker')))
        self.assertEqual(
            obs['code'],
            ('DEF', 'run', 'm(', 'turnLeft', 'move', 'pickMarker', 'm)'))
        self._test_step(obs, reward, done, False)

        # DEF run m( turnLeft REPEAT R=2 r( move pickMarker r) m)
        obs, reward, done, info = env.step((mutation.WRAP_BLOCK,
                                            ('repeat', 0, 3, 6)))
        self.assertEqual(obs['code'],
                         ('DEF', 'run', 'm(', 'turnLeft', 'REPEAT', 'R=2',
                          'r(', 'move', 'pickMarker', 'r)', 'm)'))
        self._test_step(obs, reward, done, True)

    #def testCreateComplex(self):
    #    # DEF run m( REPEAT R=5 r( turnLeft IFELSE c( markersPresent c) i(
    #    # turnRight i) ELSE e( move e) r) pickMarker m)
    #    return

    def setUp(self):
        # DEF run m( pickMarker move pickMarker m)
        # Index 5 in train.pkl
        self.simple_input_tests = [{
            'input': [
                56, 1620, 1621, 1622, 1623, 1624, 1638, 1642, 1656, 1660, 1674,
                1678, 1692, 1696, 1710, 1714, 1728, 1729, 1730, 1731, 1732,
                2000, 2018, 2019, 2035, 2323, 2937, 3277, 3332, 3333
            ],
            'output': [
                74, 1620, 1621, 1622, 1623, 1624, 1638, 1642, 1656, 1660, 1674,
                1678, 1692, 1696, 1710, 1714, 1728, 1729, 1730, 1731, 1732,
                2019, 2035, 2323, 2937, 3277, 3332, 3333
            ]
        }, {
            'input': [
                1036, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628,
                1629, 1630, 1631, 1632, 1633, 1638, 1651, 1656, 1669, 1674,
                1687, 1692, 1705, 1710, 1723, 1728, 1741, 1746, 1759, 1764,
                1777, 1782, 1795, 1800, 1813, 1818, 1831, 1836, 1849, 1854,
                1867, 1872, 1885, 1890, 1903, 1908, 1909, 1910, 1911, 1912,
                1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1963,
                1964, 1969, 1971, 1982, 1983, 1986, 1989, 2007, 2019, 2024,
                2026, 2062, 2097, 2107, 2110, 2128, 2131, 2134, 2188, 2198,
                2220, 2417, 2449, 2545, 2656, 2669, 2838, 3084, 3190, 3342,
                3450, 3950, 4241, 4364, 4450, 4602, 4674
            ],
            'output': [
                1035, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628,
                1629, 1630, 1631, 1632, 1633, 1638, 1651, 1656, 1669, 1674,
                1687, 1692, 1705, 1710, 1723, 1728, 1741, 1746, 1759, 1764,
                1777, 1782, 1795, 1800, 1813, 1818, 1831, 1836, 1849, 1854,
                1867, 1872, 1885, 1890, 1903, 1908, 1909, 1910, 1911, 1912,
                1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1963,
                1964, 1969, 1971, 1982, 1983, 1986, 1989, 2019, 2024, 2026,
                2062, 2097, 2107, 2110, 2128, 2131, 2134, 2188, 2198, 2220,
                2332, 2417, 2449, 2545, 2669, 2838, 3084, 3190, 3342, 3450,
                3950, 4241, 4364, 4450, 4602, 4674
            ]
        }, {
            'input': [
                29, 1324, 1341, 1355, 1369, 1371, 1378, 1383, 1392, 1411, 1418,
                1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629,
                1630, 1631, 1632, 1633, 1634, 1635, 1636, 1638, 1654, 1656,
                1672, 1674, 1690, 1692, 1708, 1710, 1726, 1728, 1744, 1746,
                1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756,
                1757, 1758, 1759, 1760, 1761, 1762, 1970, 1973, 1981, 1985,
                1991, 2028, 2043, 2045, 2935, 4648
            ],
            'output': [
                47, 1324, 1341, 1355, 1369, 1371, 1378, 1383, 1392, 1411, 1418,
                1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629,
                1630, 1631, 1632, 1633, 1634, 1635, 1636, 1638, 1654, 1656,
                1672, 1674, 1690, 1692, 1708, 1710, 1726, 1728, 1744, 1746,
                1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1756,
                1757, 1758, 1759, 1760, 1761, 1762, 1970, 1981, 1985, 2028,
                2043, 2045, 2935, 4648
            ]
        }, {
            'input': [
                19, 1620, 1621, 1622, 1623, 1624, 1625, 1638, 1643, 1656, 1661,
                1674, 1679, 1692, 1697, 1710, 1715, 1728, 1733, 1746, 1751,
                1764, 1765, 1766, 1767, 1768, 1769, 1981, 2935
            ],
            'output': [
                37, 1620, 1621, 1622, 1623, 1624, 1625, 1638, 1643, 1656, 1661,
                1674, 1679, 1692, 1697, 1710, 1715, 1728, 1733, 1746, 1751,
                1764, 1765, 1766, 1767, 1768, 1769, 2611
            ]
        }, {
            'input': [
                705, 1320, 1323, 1333, 1339, 1359, 1360, 1378, 1379, 1620,
                1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630,
                1631, 1632, 1638, 1650, 1656, 1668, 1674, 1686, 1692, 1704,
                1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719,
                1720, 1721, 1722, 2001, 2955, 2997, 3266
            ],
            'output': [
                687, 1320, 1323, 1333, 1339, 1359, 1360, 1378, 1379, 1620,
                1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630,
                1631, 1632, 1638, 1650, 1656, 1668, 1674, 1686, 1692, 1704,
                1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719,
                1720, 1721, 1722, 2631, 2997, 3266
            ]
        }]

        # DEF run m( turnLeft REPEAT R=2 r( move pickMarker r) m)
        # Index 2 in train.pkl
        self.moderate_input_tests = [{
            'input': [
                361, 1388, 1389, 1620, 1621, 1622, 1623, 1624, 1625, 1638,
                1643, 1656, 1661, 1674, 1679, 1692, 1697, 1710, 1715, 1728,
                1729, 1730, 1731, 1732, 1733, 1965, 1981, 1982, 2017, 2035,
                2287, 2971, 4251
            ],
            'output': [
                73, 1388, 1389, 1620, 1621, 1622, 1623, 1624, 1625, 1638, 1643,
                1656, 1661, 1674, 1679, 1692, 1697, 1710, 1715, 1728, 1729,
                1730, 1731, 1732, 1733, 1965, 1981, 1982, 2035, 2287, 2647,
                4251
            ]
        }, {
            'input': [
                434, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628,
                1629, 1630, 1631, 1638, 1649, 1656, 1667, 1674, 1685, 1692,
                1703, 1710, 1721, 1728, 1739, 1746, 1757, 1764, 1775, 1782,
                1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792,
                1793, 1964, 1999, 2005, 2021, 2024, 2039, 2060, 2072, 3062,
                3066
            ],
            'output': [
                146, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628,
                1629, 1630, 1631, 1638, 1649, 1656, 1667, 1674, 1685, 1692,
                1703, 1710, 1721, 1728, 1739, 1746, 1757, 1764, 1775, 1782,
                1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1791, 1792,
                1793, 1964, 1999, 2005, 2021, 2024, 2039, 2060, 2738, 3066
            ]
        }, {
            'input': [
                343, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628,
                1629, 1630, 1631, 1632, 1633, 1638, 1651, 1656, 1669, 1674,
                1687, 1692, 1705, 1710, 1723, 1728, 1741, 1746, 1759, 1764,
                1777, 1782, 1795, 1800, 1813, 1818, 1831, 1836, 1849, 1854,
                1867, 1872, 1885, 1890, 1903, 1908, 1909, 1910, 1911, 1912,
                1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1967,
                1981, 2005, 2038, 2058, 2077, 2079, 2125, 2128, 2132, 2136,
                2218, 2647, 3479, 3584, 3997, 4323, 4566
            ],
            'output': [
                55, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629,
                1630, 1631, 1632, 1633, 1638, 1651, 1656, 1669, 1674, 1687,
                1692, 1705, 1710, 1723, 1728, 1741, 1746, 1759, 1764, 1777,
                1782, 1795, 1800, 1813, 1818, 1831, 1836, 1849, 1854, 1867,
                1872, 1885, 1890, 1903, 1908, 1909, 1910, 1911, 1912, 1913,
                1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921, 1967, 2005,
                2038, 2058, 2077, 2079, 2125, 2128, 2132, 2136, 2218, 2323,
                3479, 3584, 3997, 4323, 4566
            ]
        }, {
            'input': [
                542, 1374, 1390, 1407, 1409, 1425, 1441, 1442, 1463, 1477,
                1482, 1500, 1513, 1572, 1620, 1621, 1622, 1623, 1624, 1625,
                1626, 1627, 1638, 1645, 1656, 1663, 1674, 1681, 1692, 1699,
                1710, 1717, 1728, 1735, 1746, 1753, 1764, 1771, 1782, 1789,
                1800, 1807, 1818, 1825, 1836, 1843, 1854, 1861, 1872, 1879,
                1890, 1897, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915,
                1963, 1965, 1967, 1982, 1984, 1986, 1999, 2002, 2019, 2020,
                2040, 2053, 2056, 2058, 2071, 2072, 2075, 2076, 2092, 2094,
                2107, 2108, 2109, 2112, 2126, 2127, 2129, 2145, 2146, 2147,
                2162, 2163, 2165, 2166, 2180, 2182, 2184, 2197, 2200, 2215,
                2216, 2218, 2292, 2307, 2342, 2417, 2543, 2648, 2687, 2702,
                2865, 2993, 3082, 3151, 3155, 3170, 3171, 3281, 3333, 3439,
                3655, 3711, 3822, 4072, 4088, 4108, 4249, 4556, 4595, 4609,
                4793
            ],
            'output': [
                254, 1374, 1390, 1407, 1409, 1425, 1441, 1442, 1463, 1477,
                1482, 1500, 1513, 1572, 1620, 1621, 1622, 1623, 1624, 1625,
                1626, 1627, 1638, 1645, 1656, 1663, 1674, 1681, 1692, 1699,
                1710, 1717, 1728, 1735, 1746, 1753, 1764, 1771, 1782, 1789,
                1800, 1807, 1818, 1825, 1836, 1843, 1854, 1861, 1872, 1879,
                1890, 1897, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915,
                1963, 1965, 1967, 1982, 1984, 1986, 1999, 2002, 2019, 2020,
                2040, 2053, 2056, 2058, 2071, 2072, 2075, 2076, 2092, 2094,
                2107, 2108, 2109, 2112, 2126, 2127, 2129, 2145, 2146, 2147,
                2162, 2163, 2165, 2166, 2182, 2184, 2197, 2200, 2215, 2216,
                2218, 2292, 2307, 2342, 2417, 2543, 2648, 2687, 2702, 2846,
                2865, 2993, 3082, 3151, 3155, 3171, 3281, 3333, 3439, 3655,
                3711, 3822, 4072, 4088, 4108, 4249, 4556, 4595, 4609, 4793
            ]
        }, {
            'input': [
                689, 1319, 1336, 1394, 1406, 1409, 1412, 1620, 1621, 1622,
                1623, 1624, 1625, 1626, 1627, 1628, 1629, 1638, 1647, 1656,
                1665, 1674, 1683, 1692, 1701, 1710, 1719, 1728, 1737, 1746,
                1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1965,
                1982, 1985, 1987, 1988, 2005, 2036, 2038, 2039, 2058, 2292,
                2293, 2344, 2361, 2958, 2972, 3320, 4555
            ],
            'output': [
                367, 1319, 1336, 1394, 1406, 1409, 1412, 1620, 1621, 1622,
                1623, 1624, 1625, 1626, 1627, 1628, 1629, 1638, 1647, 1656,
                1665, 1674, 1683, 1692, 1701, 1710, 1719, 1728, 1737, 1746,
                1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755, 1965,
                1982, 1985, 1988, 2005, 2036, 2038, 2039, 2058, 2292, 2293,
                2344, 2361, 2634, 2972, 3320, 4555
            ]
        }]

        # DEF run m( REPEAT R=5 r( turnLeft IFELSE c( markersPresent c) i(
        # turnRight i) ELSE e( move e) r) pickMarker m)
        # Index 0 in train.pkl
        self.complex_input_tests = [{
            'input': [
                811, 1325, 1360, 1379, 1405, 1411, 1423, 1431, 1472, 1480,
                1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629,
                1630, 1631, 1632, 1633, 1634, 1635, 1638, 1653, 1656, 1671,
                1674, 1689, 1692, 1707, 1710, 1725, 1728, 1743, 1746, 1761,
                1764, 1779, 1782, 1797, 1800, 1815, 1818, 1819, 1820, 1821,
                1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831,
                1832, 1833, 1976, 1999, 2076, 2095, 2096, 2099, 2108, 2117,
                2130, 2635, 2636, 2766, 3016
            ],
            'output': [
                488, 1325, 1360, 1379, 1405, 1411, 1423, 1431, 1472, 1480,
                1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629,
                1630, 1631, 1632, 1633, 1634, 1635, 1638, 1653, 1656, 1671,
                1674, 1689, 1692, 1707, 1710, 1725, 1728, 1743, 1746, 1761,
                1764, 1779, 1782, 1797, 1800, 1815, 1818, 1819, 1820, 1821,
                1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831,
                1832, 1833, 1976, 1999, 2076, 2095, 2096, 2099, 2117, 2130,
                2635, 2636, 2766, 3016
            ]
        }, {
            'input': [
                1159, 1320, 1333, 1334, 1358, 1405, 1423, 1620, 1621, 1622,
                1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1638, 1648,
                1656, 1666, 1674, 1684, 1692, 1702, 1710, 1720, 1728, 1738,
                1746, 1756, 1764, 1774, 1782, 1792, 1800, 1810, 1818, 1819,
                1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1965,
                2043, 2077, 2089, 2093, 2113, 2453, 2775, 3267, 3979, 4364,
                4599, 4704
            ],
            'output': [
                817, 1320, 1333, 1334, 1358, 1405, 1423, 1620, 1621, 1622,
                1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1638, 1648,
                1656, 1666, 1674, 1684, 1692, 1702, 1710, 1720, 1728, 1738,
                1746, 1756, 1764, 1774, 1782, 1792, 1800, 1810, 1818, 1819,
                1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1965,
                2043, 2077, 2089, 2093, 2453, 2775, 3267, 3979, 4364, 4599,
                4704
            ]
        }, {
            'input': [
                1123, 1317, 1333, 1334, 1335, 1358, 1371, 1375, 1376, 1391,
                1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629,
                1638, 1647, 1656, 1665, 1674, 1683, 1692, 1701, 1710, 1719,
                1728, 1737, 1746, 1755, 1764, 1773, 1782, 1791, 1800, 1801,
                1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 2077, 2093,
                2109, 2110, 2974, 4560, 4706
            ],
            'output': [
                781, 1317, 1333, 1334, 1335, 1358, 1371, 1375, 1376, 1391,
                1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629,
                1638, 1647, 1656, 1665, 1674, 1683, 1692, 1701, 1710, 1719,
                1728, 1737, 1746, 1755, 1764, 1773, 1782, 1791, 1800, 1801,
                1802, 1803, 1804, 1805, 1806, 1807, 1808, 1809, 2093, 2109,
                2110, 2974, 4560, 4706
            ]
        }, {
            'input': [
                1100, 1352, 1620, 1621, 1622, 1623, 1638, 1641, 1656, 1659,
                1674, 1677, 1692, 1695, 1710, 1713, 1728, 1731, 1746, 1749,
                1764, 1765, 1766, 1767, 2054
            ],
            'output': [
                758, 1352, 1620, 1621, 1622, 1623, 1638, 1641, 1656, 1659,
                1674, 1677, 1692, 1695, 1710, 1713, 1728, 1731, 1746, 1749,
                1764, 1765, 1766, 1767
            ]
        }, {
            'input': [
                759, 1317, 1335, 1337, 1351, 1406, 1441, 1448, 1620, 1621,
                1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1638,
                1648, 1656, 1666, 1674, 1684, 1692, 1702, 1710, 1720, 1728,
                1738, 1746, 1756, 1764, 1774, 1782, 1792, 1800, 1801, 1802,
                1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 2022, 2023,
                2056, 2291, 3030, 3584
            ],
            'output': [
                436, 1317, 1335, 1337, 1351, 1406, 1441, 1448, 1620, 1621,
                1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1638,
                1648, 1656, 1666, 1674, 1684, 1692, 1702, 1710, 1720, 1728,
                1738, 1746, 1756, 1764, 1774, 1782, 1792, 1800, 1801, 1802,
                1803, 1804, 1805, 1806, 1807, 1808, 1809, 1810, 2022, 2023,
                2291, 3030, 3584
            ]
        }]


class TestComputeAddOps(object):

    parser = parser_for_synthesis.KarelForSynthesisParser(build_tree=True)

    def linearize_to_tokens(self, code):
        tokens, orig_spans = refine_env.ComputeAddOps.linearize(self.parser.parse(code))
        return tuple(refine_env.ComputeAddOps.idx_to_token[i]
                     for i in tokens), orig_spans

    @pytest.mark.parametrize('tokens,linearized', [
        ('DEF run m( move putMarker m)', (('move', 'putMarker'), ((3, 3),
                                                                  (4, 4)))),
        ('DEF run m( move '
         'IF c( not c( frontIsClear c) c) i( putMarker i) '
         'turnLeft m)', (('move', ('if', ('not', 'frontIsClear')), 'putMarker',
                          ('end-if', ('not', 'frontIsClear')), 'turnLeft'), (
                              (3, 3), (4, 11), (12, 12), (13, 13), (14, 14)))),
        ('DEF run m( move REPEAT R=5 r( putMarker r) turnLeft m)',
         (('move', ('repeat', 5), 'putMarker', ('end-repeat', 5), 'turnLeft'),
          ((3, 3), (4, 6), (7, 7), (8, 8), (9, 9)))),
        ('DEF run m( pickMarker '
         'IFELSE c( noMarkersPresent c) i( putMarker i) '
         'ELSE e( turnRight e) move m)',
         (('pickMarker', ('ifElse', 'noMarkersPresent'), 'putMarker',
           ('else', 'noMarkersPresent'), 'turnRight', (
               'end-ifElse', 'noMarkersPresent'), 'move'),
          ((3, 3), (4, 8), (9, 9), (10, 12), (13, 13), (14, 14), (15, 15)))),
    ])
    def testLinearizeSimple(self, tokens, linearized):
        assert  self.linearize_to_tokens(tokens) == linearized

    def _goal_reached_systematic(self, goal):
        goal_atree = refine_env.AnnotatedTree(code=goal)
        queue = collections.deque([(('DEF', 'run', 'm(', 'm)'), None)])
        closed = set()
        goal_reached = False

        while queue:
            current, prev = queue.popleft()
            if current in closed:
                continue
            closed.add(current)

            current_atree = refine_env.AnnotatedTree(code=current)
            assert refine_env.is_subseq(current_atree.linearized[0],
                    goal_atree.linearized[0])

            actions = set(
                refine_env.ComputeAddOps.run(current_atree, goal_atree))
            #bad_actions = set(
            #    a
            #    for a in refine_env.MutationActionSpace(code=current)
            #    .enumerate_additive_actions() if a not in actions)
            for action in actions:
                mutation_space = refine_env.MutationActionSpace(
                    atree=copy.deepcopy(current_atree))
                assert mutation_space.contains(action)
                mutation_space.apply(action)
                new_code = mutation_space.atree.code
                if new_code not in closed:
                    queue.append((new_code, current))
                assert refine_env.is_subseq(current_atree.linearized[0],
                                         mutation_space.atree.linearized[0])

            if not actions:
                assert current == goal
                goal_reached = True
                continue

            # Check that every other action is invalid
            #for bad_action in bad_actions:
            #    mutation_space = refine_env.MutationActionSpace(
            #        tree=copy.deepcopy(current_tree))
            #    mutation_space.apply(bad_action)
            #    new_code_linearized, _ = refine_env.ComputeAddOps.linearize(mutation_space.tree)
            #    assert not refine_env.is_subseq(new_code_linearized,
            #        goal_atree.linearized[0]))

        assert goal_reached

    @pytest.mark.parametrize('code', [
                '''DEF run m( m)''',
                '''DEF run m( move turnLeft m)''',
                '''DEF run m( move move move move move m)''',
                '''DEF run m(
                REPEAT R=10 r( putMarker move r)
                putMarker turnLeft m)''',
                '''DEF run m(
                IF c( markersPresent c) i(
                    move
                    turnLeft
                i)
                IF c( leftIsClear c) i(
                    move
                    turnLeft
                i)
            m)''',
                '''DEF run m(
                IF c( markersPresent c) i(
                    move
                    turnLeft
                i)
                IF c( markersPresent c) i(
                    move
                    turnLeft
                i)
            m)''',
                '''DEF run m(
                IF c( leftIsClear c) i(
                    IF c( leftIsClear c) i(
                        move
                    i)
                i)
            m)''',
                '''DEF run m(
                IF c( leftIsClear c) i(
                    turnLeft
                    IF c( leftIsClear c) i(
                        turnLeft
                    i)
                    turnLeft
                i)
            m)''',
                '''DEF run m(
                REPEAT R=5 r(
                    move
                    IFELSE c( markersPresent c) i(
                        move
                    i) ELSE e(
                        move
                        IFELSE c( markersPresent c) i(
                            move
                        i) ELSE e(
                            move
                        e)
                        move
                    e)
                r)
                move
            m)''',
            ])
    def testRun(self, code):
        self._goal_reached_systematic(tuple(code.split()))

    @pytest.mark.parametrize(
        'code',
        open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'testdata',
                'short_val_code.txt')).readlines()[:300])
    def testRunValCode(self, code):
        self._goal_reached_systematic(tuple(code.split()))


class SubseqTest(unittest.TestCase):
    def testSubseqInsertionsManual(self):
        self.assertEqual(
            refine_env.subseq_insertions('a', 'a'), [set(), set()])

        self.assertEqual(
            refine_env.subseq_insertions('aa', 'aa'), [set(), set(), set()])

        self.assertEqual(
            refine_env.subseq_insertions('ab', 'ab'), [set(), set(), set()])

        self.assertEqual(
            refine_env.subseq_insertions('a', 'ab'), [set(), {'b'}])

        self.assertEqual(
            refine_env.subseq_insertions('abb', 'xabyxaby'),
            [{'x'}, set(), {'a', 'x', 'y'},  {'y'}])

        self.assertEqual(
            refine_env.subseq_insertions('aa', 'aaabbbaaa'),
            [{'a', 'b'}, {'a', 'b'}, {'a', 'b'}])

        self.assertEqual(
            refine_env.subseq_insertions('aab', 'aaabbbccc'),
            [{'a'}, {'a'}, {'a', 'b'}, {'b', 'c'}])

        self.assertEqual(
            refine_env.subseq_insertions('aabc', 'aaabbbccc'),
            [{'a'}, {'a'}, {'a', 'b'}, {'b', 'c'}, {'c'}])

    def testSubseqInsertionsRandom(self):
        rng = np.random.RandomState(12345)
        for vocab_size in range(2, 6):
            vocab = set(range(vocab_size))
            b = rng.randint(vocab_size, size=10)
            for subseq_len in 2, 3, 8, 9:
                for _ in range(10):
                    a = b[np.sort(
                        rng.choice(
                            10, size=subseq_len, replace=False))]
                    insert_sets, left_bound, right_bound = refine_env.subseq_insertions(
                        a, b, debug=True)
                    for i, insert_set in enumerate(
                            refine_env.subseq_insertions(a, b)):
                        for insert in insert_set:
                            a_prime = np.concatenate((a[:i], [insert], a[i:]))
                            self.assertTrue(
                                refine_env.is_subseq(a_prime, b),
                                msg='a: {}, b: {}, a\': {}, loc: {}, '
                                'insert_set: {}, insert: {}, '
                                'left_bound: {}, right_bound {}'.format(
                                    a, b, a_prime, i, insert_set, insert,
                                    left_bound, right_bound))
                        for not_insert in vocab - insert_set:
                            a_prime = np.concatenate(
                                    (a[:i], [not_insert], a[i:]))
                            self.assertFalse(
                                refine_env.is_subseq(a_prime, b),
                                msg='a: {}, b: {}, a\': {}, loc: {}, '
                                'insert_set: {}, insert: {}, '
                                'left_bound: {}, right_bound {}'.format(
                                    a, b, a_prime, i, insert_set, insert,
                                    left_bound, right_bound))
