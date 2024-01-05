import unittest
import numpy as np
from lasrasterize.lib import (
    fillholes,
    BBox,
    infer_raster_resolution,
    Layerdef,
    lasdata_to_rasters,
    lasfile_to_geotiff,
)
import os
import rasterio as rio
import laspy


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
            self.assertAlmostEqual(A[9, 9], -0.51, places=2)


if __name__ == "__main__":
    unittest.main()
