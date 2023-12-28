import laspy
import rasterio
import numpy as np
from collections import namedtuple

SQRT_TWO = 2**0.5

BBox = namedtuple("BBox", ['left', 'bottom', 'right', 'top'])

class PointCloud:
    """A lidar PointCloud, with points, intensity, return_nums, and num_returns per point."""

    def __init__(self):
        self.points = None
        self.intensity = None
        self.return_num = None
        self.num_returns = None

    @classmethod
    def from_laspy(cls, laspy_file):
        """Instantiate class from LAS file."""

        ret = cls()

        pts = laspy_file.points["point"]
        ret.points = np.column_stack((pts["X"], pts["Y"], pts["Z"]))*laspy_file.header.scale

        ret.intensity = pts["intensity"]/255
        ret.return_num = laspy_file.return_num

        # cast to signed into so subtraction can result in negative numbers
        ret.num_returns = laspy_file.get_num_returns().astype(int)

        return ret

    @property
    def bbox(self):
        """Finds horizontal bounding box."""

        left, bottom, _ = self.points.min(axis=0)
        right, top, _ = self.points.max(axis=0)
        return BBox(left, bottom, right, top)

    def get_layer(self, layer):
        """Returns a new PointCloud consisting of only points from a given
        layer. If 'layer' is positive, returns points with the corresponding
        return number. If 'layer' is negative, returns points with a return
        number counting from the last return, e.g. -1 filters to last-return
        points."""

        if layer==0:
            raise ValueError("Layer must be positive or negative, not zero.")

        if layer < 0:
            layer = self.num_returns + layer + 1

        mask = (self.return_num == layer)

        ret = PointCloud()
        ret.points = self.points[mask]
        ret.intensity = self.intensity[mask]
        ret.return_num = self.return_num[mask]
        ret.num_returns = self.num_returns[mask]

        return ret

def las_to_raster(las_file, raster_file):
    # Load LAS file
    las = laspy.file.File(las_file, mode='r')

    # Get coordinates
    x = las.x
    y = las.y
    z = las.z

    # Create a raster data array (this is a placeholder, you'll need to implement the actual conversion)
    data = np.zeros((len(x), len(y)))

    # Save as raster
    with rasterio.open(raster_file, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1], count=1, dtype=str(data.dtype)) as dst:
        dst.write(data, 1)

    print(f"Raster file saved as {raster_file}")