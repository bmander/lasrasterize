import unittest
import numpy as np
from lasrasterize.lib import (
    fillholes,
    BBox,
    infer_raster_resolution,
)
import os
import rasterio as rio
import math


class TestFillHoles(unittest.TestCase):
    def test_fillholes_no_nan(self):
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = fillholes(mat)
        np.testing.assert_array_equal(result, mat)

    def test_fillholes_with_nan(self):
        mat = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        expected = np.array([[1, 2.833333, 3], [4, 5, 6.166667], [7, 8, 9]])
        result = fillholes(mat)
        np.testing.assert_array_almost_equal(result, expected)

    def test_fillholes_all_nan(self):
        mat = np.full((3, 3), np.nan)
        result = fillholes(mat)
        self.assertTrue(np.isnan(result).all())

    def test_fillholes_with_radius(self):
        mat = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        expected = np.array([[1, 2.833333, 3], [4, 5, 6.166667], [7, 8, 9]])
        result = fillholes(mat, radius=1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_fillholes_zero_radius(self):
        mat = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        expected = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        result = fillholes(mat, radius=0)
        np.testing.assert_array_equal(result, expected)


class TestLidarToRasters(unittest.TestCase):
    def setUp(self):
        # construct filename from the position of this test file
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        test_las_filename = os.path.join(test_data_dir, "sine.las")

    def test_lidar_to_rasters(self):
        pass


if __name__ == "__main__":
    unittest.main()
