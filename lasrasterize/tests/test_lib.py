import unittest
import numpy as np
from lasrasterize.lib import PointCloud
from unittest.mock import Mock

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
        mock_laspy_file = Mock()
        mock_laspy_file.points = {"point": {"X": np.array([1, 4, 7]), "Y": np.array([2, 5, 8]), "Z": np.array([3, 6, 9])}}
        mock_laspy_file.header.scale = np.array([1, 1, 1])
        mock_laspy_file.return_num = np.array([1, 2, 3])
        mock_laspy_file.get_num_returns.return_value = np.array([3, 2, 1])

        point_cloud = PointCloud.from_laspy(mock_laspy_file)

        np.testing.assert_array_equal(point_cloud.points, self.point_cloud.points)
        np.testing.assert_array_equal(point_cloud.intensity, self.point_cloud.intensity)
        np.testing.assert_array_equal(point_cloud.return_num, self.point_cloud.return_num)
        np.testing.assert_array_equal(point_cloud.num_returns, self.point_cloud.num_returns)

if __name__ == '__main__':
    unittest.main()