import os
import unittest

import laspy
import numpy as np
import rasterio as rio

from lasrasterize.lib import (Layerdef, fill_with_nearby_average,
                              infer_raster_resolution, lasdata_to_rasters,
                              lasfile_to_geotiff, points_to_raster_interpolate,
                              points_to_raster_grid_and_fill)


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
                lasdata, (0, 0.1), 10, 10, 0.01, 0.01, [layer_def],
            )

            # assert that the rasters are the correct shape
            self.assertEqual(rasters.shape, (1, 10, 10))

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
    def test_points_to_raster_interpolate_uniform(self):
        # donut of 5 with a hole in the middle
        mat = np.array([[0, 0, 5], [0, 1, 5], [0, 2, 5],
                        [1, 0, 5], [1, 2, 5],
                        [2, 0, 5], [2, 1, 5], [2, 2, 5]]).transpose()

        res = 1

        interp_nearest = points_to_raster_interpolate(mat, (0, 3), 3, 3, res,
                                                      res,
                                                      method="nearest")

        expected_interp_nearest = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])

        np.testing.assert_array_equal(interp_nearest, expected_interp_nearest)

        interp_linear = points_to_raster_interpolate(mat, (0, 3), 3, 3, res,
                                                     res,
                                                     method="linear")

        expected_interp_linear = np.array([[np.nan, np.nan, np.nan],
                                           [5, 5, np.nan],
                                           [5, 5, np.nan]])

        np.testing.assert_array_equal(interp_linear, expected_interp_linear)

        interp_cubic = points_to_raster_interpolate(mat, (0, 3), 3, 3, res,
                                                    res,
                                                    method="cubic")
        expected_interp_cube = np.array([[np.nan, np.nan, np.nan],
                                         [5, 5, np.nan],
                                         [5, 5, np.nan]])

        np.testing.assert_array_equal(interp_cubic, expected_interp_cube)

        raster2 = points_to_raster_interpolate(mat, (0, 3), 6, 6, 0.5, 0.5,
                                               method="nearest")

        expected2 = np.array([[5, 5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5, 5],
                              [5, 5, 5, 5, 5, 5]])

        np.testing.assert_array_equal(raster2, expected2)

    def test_points_to_raster_interpolate_methods(self):
        # gradient from left to right with hole in the middle
        mat = np.array([[0, 0, 0], [1, 0, 1], [2, 0, 2], [3, 0, 3], [4, 0, 4],
                        [0, 1, 0], [4, 1, 4],
                        [0, 2, 0], [4, 2, 4],
                        [0, 3, 0], [1, 3, 1], [2, 3, 2],
                        [3, 3, 3], [4, 3, 4]]).transpose()

        res = 1

        raster = points_to_raster_interpolate(mat, (0, 3), 4, 3, res,
                                              res,
                                              method="nearest")

        expected = np.array([[0, 1, 3, 4],
                             [0, 0, 4, 4],
                             [0, 1, 3, 4]])

        np.testing.assert_array_almost_equal(raster, expected)

        raster_linear = points_to_raster_interpolate(mat, (0, 3), 4,
                                                     3,
                                                     res, res,
                                                     method="linear")

        expected_linear = np.array([[0., 1.333333, 2.666667, 4.],
                                    [0., 1.333333, 2.666667, 4.],
                                    [0., 1.333333, 2.666667, 4.]])

        np.testing.assert_array_almost_equal(raster_linear, expected_linear)

        raster_cubic = points_to_raster_interpolate(mat, (0, 3), 4,
                                                    3,
                                                    res, res,
                                                    method="cubic")

        expected_cubic = np.array([[0., 1.333333, 2.666667, 4.],
                                   [0., 1.333333, 2.666667, 4.],
                                   [0., 1.333333, 2.666667, 4.]])

        np.testing.assert_array_almost_equal(raster_cubic, expected_cubic)


class TestOrientation(unittest.TestCase):
    def test_interpolate(self):
        # square of four points, larger z value on the top than the bottom
        mat = np.array([[0, 0, 0], [2, 0, 0],
                        [0, 2, 1], [2, 2, 1]]).transpose()
        res = 1
        raster = points_to_raster_interpolate(mat, (0, 2), 2, 2, res, res,
                                              method="nearest")

        np.testing.assert_array_almost_equal(raster,
                                             np.array([[1, 1], [0, 0]]))

        raster = points_to_raster_interpolate(mat, (0, 2), 2, 2, res, res,
                                              method="linear")

        np.testing.assert_array_almost_equal(raster,
                                             np.array([[1, 1], [0, 0]]))

        raster = points_to_raster_interpolate(mat, (0, 2), 2, 2, res, res,
                                              method="cubic")

        np.testing.assert_array_almost_equal(raster,
                                             np.array([[1, 1], [0, 0]]))

    def test_grid_and_fill(self):
        # square of four points, larger z value on the top than the bottom
        mat = np.array([[0.01, 0.01, 0], [1.99, 0.01, 0],
                        [0.01, 1.99, 1], [1.99, 1.99, 1]]).transpose()
        res = 1
        raster = points_to_raster_grid_and_fill(mat, (0, 2), 2, 2, res, res)

        np.testing.assert_array_almost_equal(raster,
                                             np.array([[1, 1], [0, 0]]))


class TestOutOfBounds(unittest.TestCase):
    def test_interpolate(self):
        # square of four points; corners of the box (0, 0, 4, 4)
        mat = np.array([[0, 0, 1], [4, 0, 1],
                        [0, 4, 1], [4, 4, 1]]).transpose()
        res = 1

        raster = points_to_raster_interpolate(mat, (1, 3), 2, 2, res, res,
                                              method="nearest")
        np.testing.assert_array_almost_equal(raster,
                                             np.array([[1, 1], [1, 1]]))

        raster = points_to_raster_interpolate(mat, (1, 3), 2, 2, res, res,
                                              method="linear")
        np.testing.assert_array_almost_equal(raster,
                                             np.array([[1, 1], [1, 1]]))

        raster = points_to_raster_interpolate(mat, (1, 3), 2, 2, res, res,
                                              method="cubic")
        np.testing.assert_array_almost_equal(raster,
                                             np.array([[1, 1], [1, 1]]))

    def test_grid_and_fill(self):
        # square of four points; corners of the box (0, 0, 4, 4)
        mat = np.array([[0, 0, 1], [4, 0, 1],
                        [0, 4, 1], [4, 4, 1]]).transpose()
        res = 1

        raster = points_to_raster_grid_and_fill(mat, (1, 3), 2, 2, res, res)
        np.testing.assert_array_almost_equal(raster,
                                             np.array([[np.nan, np.nan],
                                                       [np.nan, np.nan]]))


class TestEmpty(unittest.TestCase):
    def test_empty(self):
        mat = np.array([[]])
        resolution = 1

        self.assertRaises(ValueError, points_to_raster_interpolate, mat,
                          (0, 1),
                          1, 1, resolution, resolution)

        mat = points_to_raster_grid_and_fill(mat, (0, 1), 1, 1, resolution,
                                             resolution)

        expected = np.array([[np.nan]])

        np.testing.assert_array_equal(mat, expected)


class TestSinglePoint(unittest.TestCase):
    def test_middle(self):
        mat = np.array([[.5, .5, 5]]).transpose()
        resolution = 1

        self.assertRaises(ValueError, points_to_raster_interpolate, mat,
                          (0, 1),
                          1, 1, resolution, resolution)

        mat = points_to_raster_grid_and_fill(mat, (0, 1), 1, 1, resolution,
                                             resolution)

        expected = np.array([[5]])

        np.testing.assert_array_equal(mat, expected)


if __name__ == "__main__":
    unittest.main()
