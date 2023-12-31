import os
import unittest

import laspy
import numpy as np
import rasterio as rio

from lasrasterize.lib import (BBox, Layerdef, fill_with_nearby_average,
                              infer_raster_resolution, lasdata_to_rasters,
                              lasfile_to_geotiff, points_to_raster_interpolate)


class TestFillHoles(unittest.TestCase):
    def test_fillholes_no_nan(self):
        mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        result = fill_with_nearby_average(mat)
        np.testing.assert_array_equal(result, mat)

    def test_fillholes_with_nan(self):
        mat = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        expected = np.array([[1, 2.833333, 3], [4, 5, 6.166667], [7, 8, 9]])
        result = fill_with_nearby_average(mat)
        np.testing.assert_array_almost_equal(result, expected)

    def test_fillholes_all_nan(self):
        mat = np.full((3, 3), np.nan)
        result = fill_with_nearby_average(mat)
        self.assertTrue(np.isnan(result).all())

    def test_fillholes_with_radius(self):
        mat = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        expected = np.array([[1, 2.833333, 3], [4, 5, 6.166667], [7, 8, 9]])
        result = fill_with_nearby_average(mat, radius=1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_fillholes_zero_radius(self):
        mat = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        expected = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
        result = fill_with_nearby_average(mat, radius=0)
        np.testing.assert_array_equal(result, expected)


class TestInferRasterResolution(unittest.TestCase):
    def setUp(self):
        # construct filename from the position of this test file
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        self.test_las_filename = os.path.join(test_data_dir, "sine.las")

    def test_infer_raster_resolution(self):
        # open the test file
        with laspy.open(self.test_las_filename) as f:
            lasdata = f.read()

            # infer the raster resolution
            resolution = infer_raster_resolution(lasdata)

            # assert that the resolution is about 1.73
            self.assertAlmostEqual(resolution, 1.7057, places=2)


class TestLasdataToRasters(unittest.TestCase):
    def setUp(self):
        # construct filename from the position of this test file
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        self.test_las_filename = os.path.join(test_data_dir, "test.las")

    def test_lasdata_to_rasters(self):
        # open the test file
        with laspy.open(self.test_las_filename) as f:
            lasdata = f.read()

            # create a layer definition
            layer_def = Layerdef(pulse_return=1, intensity=False)

            # convert the lasdata to rasters
            rasters = lasdata_to_rasters(
                lasdata, BBox(0, 0, 0.1, 0.1), [layer_def], 0.01, 0.01
            )

            # assert that the rasters are the correct shape
            self.assertEqual(rasters.shape, (1, 11, 11))

            # assert that the rasters are the correct type
            self.assertEqual(rasters.dtype, np.float64)

            self.assertAlmostEqual(rasters[0, 4, 0], 0.07)


class TestLasfileToGeotiff(unittest.TestCase):
    def setUp(self):
        # construct filename from the position of this test file
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        self.test_las_filename = os.path.join(test_data_dir, "sine.las")
        self.test_tif_filename = os.path.join(test_data_dir, "sine.tif")

    def tearDown(self):
        os.remove(self.test_tif_filename)

    def test_lasfile_to_geotiff(self):
        lasfile_to_geotiff(
            self.test_las_filename,
            self.test_tif_filename,
            [Layerdef(pulse_return=1, intensity=False)],
            1,
            1,
        )

        with rio.open(self.test_tif_filename) as f:
            self.assertEqual(f.count, 1)
            self.assertEqual(f.height, 10)
            self.assertEqual(f.width, 10)

            A = f.read(1)
            self.assertAlmostEqual(A[0, 0], -0.13)
            self.assertAlmostEqual(A[9, 9], -0.125, places=2)


class TestPointsToRasterInterpolate(unittest.TestCase):
    def test_points_to_raster_interpolate(self):
        # donut of 5 with a hole in the middle
        mat = np.array([[0, 0, 5], [0, 1, 5], [0, 2, 5],
                        [1, 0, 5], [1, 2, 5],
                        [2, 0, 5], [2, 1, 5], [2, 2, 5]]).transpose()
        bbox = BBox(0, 0, 2, 2)
        resolution = 1

        raster = points_to_raster_interpolate(mat, bbox, resolution,
                                              resolution)

        expected = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])

        np.testing.assert_array_equal(raster, expected)

        raster2 = points_to_raster_interpolate(mat, bbox, 0.5, 0.5)

        expected2 = np.array([[5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5]])

        np.testing.assert_array_equal(raster2, expected2)

    def test_points_to_raster_interpolate_methods(self):
        # gradient from left to right with hole in the middle
        mat = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4],
                        [0, 1, 0], [4, 1, 4],
                        [0, 2, 0], [4, 2, 4],
                        [0, 3, 0], [1, 3, 1], [2, 3, 2],
                        [3, 3, 3], [4, 3, 4]]).transpose()
        bbox = BBox(0, 0, 4, 3)
        resolution = 1

        raster = points_to_raster_interpolate(mat, bbox, resolution,
                                              resolution)

        expected = np.array([[0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4],
                             [0, 1, 2, 3, 4]])

        np.testing.assert_array_equal(raster, expected)

        raster_linear = points_to_raster_interpolate(mat, bbox, resolution,
                                                     resolution,
                                                     method="linear")

        np.testing.assert_array_equal(raster_linear, expected)

        raster_cubic = points_to_raster_interpolate(mat, bbox, resolution,
                                                    resolution,
                                                    method="cubic")

        np.testing.assert_array_almost_equal(raster_cubic, expected)


if __name__ == "__main__":
    unittest.main()
