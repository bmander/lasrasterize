import unittest
import numpy as np
from lasrasterize.lib import PointCloud
import os

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

        np.testing.assert_array_equal(point_cloud.points,  np.array([[0.01, 0.04, 0.07],
 [0.02, 0.05, 0.08],
 [0.03, 0.06, 0.09]]))
        np.testing.assert_array_almost_equal(point_cloud.intensity, np.array([0. , 0.501961, 1. ]))
        np.testing.assert_array_equal(point_cloud.return_num, np.array([1, 2, 2]))
        np.testing.assert_array_equal(point_cloud.num_returns, np.array([1, 2, 3]))


if __name__ == '__main__':
    unittest.main()