import laspy
import rasterio
import numpy as np
from collections import namedtuple
from scipy import ndimage as nd

SQRT_TWO = 2**0.5

BBox = namedtuple("BBox", ["left", "bottom", "right", "top"])


class PointCloud:
    """A lidar PointCloud, with points, intensity, return_nums, and num_returns per point."""

    def __init__(self):
        self.points = None
        self.intensity = None
        self.return_num = None
        self.num_returns = None

    @classmethod
    def _from_lasdata(cls, las: laspy.LasData):
        """Instantiate class from LASData object."""

        ret = cls()

        ret.points = np.column_stack((las.x, las.y, las.z))
        ret.intensity = np.array(las.intensity) / 255.0
        ret.return_num = np.array(las.return_num)

        # cast to signed into so subtraction can result in negative numbers
        ret.num_returns = np.array(las.num_returns).astype(int)

        return ret

    @classmethod
    def from_laspy(cls, laspy_filename: str):
        """Instantiate class from LASData file."""

        with laspy.open(laspy_filename) as laspy_file:
            las = laspy_file.read()

            return cls._from_lasdata(las)

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

        if layer == 0:
            raise ValueError("Layer must be positive or negative, not zero.")

        if layer < 0:
            layer = self.num_returns + layer + 1

        mask = self.return_num == layer

        ret = PointCloud()
        ret.points = self.points[mask]
        ret.intensity = self.intensity[mask]
        ret.return_num = self.return_num[mask]
        ret.num_returns = self.num_returns[mask]

        return ret


def fillholes(mat, radius=1):
    """Fill holes in matrix A.

    For each element in 'mat' that is nan, fill with the average of non-nan values within a given radius.
    """

    if radius == 0:
        return mat

    mat = mat.copy()

    nans = np.isnan(mat)
    valid_mask = np.logical_not(nans).astype(int)

    mat[nans] = 0

    kernel = np.ones((2 * radius + 1, 2 * radius + 1))
    neighbor_sum = nd.convolve(mat, kernel)
    neighbor_valid = nd.convolve(valid_mask, kernel)

    # Element-wise division, but ensure x/0 is nan
    with np.errstate(divide="ignore", invalid="ignore"):
        mat_mean = neighbor_sum / neighbor_valid

    ret = np.where(nans, mat_mean, mat)

    return ret


def las_to_raster(las_file, raster_file):
    # Load LAS file
    las = laspy.file.File(las_file, mode="r")

    # Get coordinates
    x = las.x
    y = las.y
    z = las.z

    # Create a raster data array (this is a placeholder, you'll need to implement the actual conversion)
    data = np.zeros((len(x), len(y)))

    # Save as raster
    with rasterio.open(
        raster_file,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=str(data.dtype),
    ) as dst:
        dst.write(data, 1)

    print(f"Raster file saved as {raster_file}")
