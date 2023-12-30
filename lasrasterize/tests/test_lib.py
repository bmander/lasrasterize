import unittest
import numpy as np
from lasrasterize.lib import (
    PointCloud,
    fillholes,
    pointcloud_to_rasters,
    BBox,
    lidar_to_rasters,
    to_geotiff,
    infer_raster_resolution,
    las_to_raster,
)
import os
import rasterio as rio
import math


class TestPointCloud(unittest.TestCase):
    def setUp(self):
        self.point_cloud = PointCloud()
        self.point_cloud.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.point_cloud.intensity = np.array([0.5, 0.6, 0.7])
        self.point_cloud.return_num = np.array([1, 2, 3])
        self.point_cloud.num_returns = np.array([3, 2, 3])

    def test_bbox(self):
        bbox = self.point_cloud.bbox
        self.assertEqual(bbox.left, 1)
        self.assertEqual(bbox.bottom, 2)
        self.assertEqual(bbox.right, 7)
        self.assertEqual(bbox.top, 8)

    def test_get_layer_positive(self):
        layer = self.point_cloud.get_layer(2)
        np.testing.assert_array_equal(layer.points, np.array([[4, 5, 6]]))
        np.testing.assert_array_equal(layer.intensity, np.array([0.6]))
        np.testing.assert_array_equal(layer.return_num, np.array([2]))
        np.testing.assert_array_equal(layer.num_returns, np.array([2]))

    def test_get_layer_negative(self):
        layer = self.point_cloud.get_layer(-1)
        np.testing.assert_array_equal(layer.points, np.array([[4, 5, 6], [7, 8, 9]]))
        np.testing.assert_array_equal(layer.intensity, np.array([0.6, 0.7]))
        np.testing.assert_array_equal(layer.return_num, np.array([2, 3]))
        np.testing.assert_array_equal(layer.num_returns, np.array([2, 3]))

    def test_get_layer_zero(self):
        with self.assertRaises(ValueError):
            self.point_cloud.get_layer(0)

    def test_from_laspy(self):
        # construct filename from the position of this test file
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        test_las_filename = os.path.join(test_data_dir, "test.las")

        point_cloud = PointCloud.from_laspy(test_las_filename)

        np.testing.assert_array_equal(
            point_cloud.points,
            np.array([[0.01, 0.04, 0.07], [0.02, 0.05, 0.08], [0.03, 0.06, 0.09]]),
        )
        np.testing.assert_array_almost_equal(
            point_cloud.intensity, np.array([0.0, 0.501961, 1.0])
        )
        np.testing.assert_array_equal(point_cloud.return_num, np.array([1, 2, 2]))
        np.testing.assert_array_equal(point_cloud.num_returns, np.array([1, 2, 3]))


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


class TestPointCloudToRasters(unittest.TestCase):
    def setUp(self):
        # construct filename from the position of this test file
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        test_las_filename = os.path.join(test_data_dir, "sine.las")

        self.point_cloud = PointCloud.from_laspy(test_las_filename)

    def test_pointcloud_to_rasters(self):
        rasters, shape = pointcloud_to_rasters(
            self.point_cloud, BBox(0, 0, 10, 10), 1, 1, 1
        )
        assert "elev" in rasters
        assert "intensity" in rasters

        np.testing.assert_array_almost_equal(
            rasters["elev"][0],
            np.array(
                [
                    -0.13,
                    -0.264,
                    -0.8,
                    -0.515,
                    -0.98,
                    -0.835,
                    -0.783333,
                    -0.445,
                    0.06,
                    -0.02,
                    -0.18,
                ]
            ),
        )

        np.testing.assert_array_almost_equal(
            rasters["intensity"][0],
            np.array(
                [
                    257.0,
                    205.59451,
                    256.976471,
                    171.318301,
                    256.968627,
                    256.973529,
                    256.975163,
                    128.484314,
                    0.0,
                    85.666667,
                    257.0,
                ]
            ),
        )


class TestLidarToRasters(unittest.TestCase):
    def setUp(self):
        # construct filename from the position of this test file
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        test_las_filename = os.path.join(test_data_dir, "sine.las")

        self.rasters, self.bbox, self.shape = lidar_to_rasters(
            test_las_filename, [1, 2], 1, 1, 1
        )

    def test_lidar_to_rasters(self):
        assert "elev" in self.rasters
        assert "intensity" in self.rasters

        np.testing.assert_array_almost_equal(
            self.rasters["elev"][0, 0, :],
            np.array(
                [
                    -0.13,
                    -0.264,
                    -0.8,
                    -0.515,
                    -0.98,
                    -0.835,
                    -0.783333,
                    -0.445,
                    0.06,
                    -0.06,
                ]
            ),
        )

        self.assertTrue(np.all(np.isnan(self.rasters["elev"][1])))

        np.testing.assert_array_almost_equal(
            self.rasters["intensity"][0, 0, :],
            np.array(
                [
                    257.0,
                    205.59451,
                    256.97647,
                    171.3183,
                    256.96863,
                    256.97354,
                    256.97516,
                    128.48431,
                    0.0,
                    128.5,
                ]
            ),
            decimal=5,
        )

        self.assertTrue(np.all(np.isnan(self.rasters["intensity"][1])))

        self.assertEqual(self.bbox.left, 0.04)
        self.assertEqual(self.bbox.bottom, 0.16)
        self.assertEqual(self.bbox.right, 9.94)
        self.assertEqual(self.bbox.top, 9.97)

        self.assertEqual(self.shape, (10, 10))

        # FILEPATH: /workspaces/lasrasterize/lasrasterize/tests/test_lib.py


class TestToGeoTiff(unittest.TestCase):
    def setUp(self):
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        test_las_filename = os.path.join(test_data_dir, "sine.las")

        self.themes, self.bbox, self.shape = lidar_to_rasters(
            test_las_filename, [1, 2], 1, 1, 1
        )

        self.crs = "EPSG:2285"  # NAD83 / Washington North (ftUS)
        self.fn_out = "test_output.tif"

    def tearDown(self):
        if os.path.exists(self.fn_out):
            os.remove(self.fn_out)

    def test_to_geotiff(self):
        to_geotiff(self.themes, self.bbox, self.shape, self.crs, self.fn_out)

        # Check if the file was created
        self.assertTrue(os.path.exists(self.fn_out))

        # Open the file and check its properties
        with rio.open(self.fn_out) as ds:
            self.assertEqual(ds.count, len(self.themes) * len(self.themes["elev"]))
            self.assertEqual(ds.height, self.shape[1])
            self.assertEqual(ds.width, self.shape[0])
            self.assertEqual(ds.crs.to_string(), self.crs)

            np.testing.assert_almost_equal(ds.read(1), self.themes["elev"][0])
            np.testing.assert_almost_equal(ds.read(2), self.themes["elev"][1])
            np.testing.assert_almost_equal(ds.read(3), self.themes["intensity"][0])
            np.testing.assert_almost_equal(ds.read(4), self.themes["intensity"][1])


class TestInferRasterResolution(unittest.TestCase):
    def test_infer_raster_resolution(self):
        # Create a mock LasData object
        class MockLasData:
            class MockHeader:
                number_of_points_by_return = [100]
                min = [0, 0, 0]
                max = [10, 10, 0]

            header = MockHeader()

        las_file = MockLasData()

        # Call the function with the mock object
        result = infer_raster_resolution(las_file)

        # Calculate the expected result
        area = 10 * 10
        specific_area = area / 100
        expected_result = round(specific_area**0.5 * math.sqrt(2), 2)

        # Assert that the result is as expected
        self.assertEqual(result, expected_result)


class TestLasToRasters(unittest.TestCase):
    def setUp(self):
        # construct filename from the position of this test file
        test_dir = os.path.dirname(os.path.realpath(__file__))
        test_data_dir = os.path.join(test_dir, "data")
        self.test_las_filename = os.path.join(test_data_dir, "sine.las")

        self.test_geotiff_filename = "test_output.tif"

    def tearDown(self):
        if os.path.exists(self.test_geotiff_filename):
            os.remove(self.test_geotiff_filename)

    def test_las_to_rasters(self):
        las_to_raster(
            self.test_las_filename,
            self.test_geotiff_filename,
            crs="EPSG:2285",
            return_num=[1],
            theme=["elev", "intensity"],
        )

        # Check if the file was created
        self.assertTrue(os.path.exists(self.test_geotiff_filename))

        # Open the file and check its properties
        with rio.open(self.test_geotiff_filename) as ds:
            self.assertEqual(ds.count, 2)
            self.assertEqual(ds.height, 8)
            self.assertEqual(ds.width, 8)
            self.assertEqual(ds.crs.to_string(), "EPSG:2285")

            expected = np.array(
                [
                    [-0.13, -0.8, -0.07, -0.98, -0.93, -0.78, 0.06, -0.5657143],
                    [0.92, 0.65, 0.79, -0.46, -0.36, -0.97, -0.67, -0.48714286],
                    [0.83, 0.5, 1.0, -0.0626087, 0.31, -0.22611111, -0.99, -0.505],
                    [-1.0, -0.99, 0.16, 0.67, 0.58, 0.75, -0.98, -0.40428573],
                    [-0.52, -0.91, -0.99, 0.3, 0.58, 0.51, -0.71, -0.32333332],
                    [0.55, -0.3, -0.91, -0.33, 0.11, 0.95, -0.1, -0.15],
                    [0.78, 0.86, -0.74, -0.025625, -0.069375, 0.07285714, -0.24, -0.51],
                    [
                        0.31133333,
                        0.22461538,
                        -0.01230769,
                        0.01692308,
                        -0.54,
                        0.77,
                        -0.08461539,
                        -0.05461539,
                    ],
                ],
                dtype=np.float32,
            )

            np.testing.assert_almost_equal(ds.read(1), expected)


if __name__ == "__main__":
    unittest.main()
