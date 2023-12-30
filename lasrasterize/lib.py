from typing import IO, Iterable
import laspy
import rasterio as rio
import numpy as np
from collections import namedtuple
from scipy import ndimage as nd
from collections import defaultdict

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
    def _from_lasdata(cls, las: laspy.LasData) -> "PointCloud":
        """Instantiates a PointCloud object from a LASData object."""

        ret = cls()

        ret.points = np.column_stack((las.x, las.y, las.z))
        ret.intensity = np.array(las.intensity) / 255.0
        ret.return_num = np.array(las.return_num)

        # cast to signed into so subtraction can result in negative numbers
        ret.num_returns = np.array(las.num_returns).astype(int)

        return ret

    @classmethod
    def from_laspy(cls, laspy_file: str | laspy.LasData) -> "PointCloud":
        """Instantiates a PointCloud object from a LAS file."""

        if isinstance(laspy_file, laspy.LasData):
            return cls._from_lasdata(laspy_file)

        with laspy.open(laspy_file) as laspy_file:
            las = laspy_file.read()

            return cls._from_lasdata(las)

    @property
    def bbox(self) -> BBox:
        """Returns a BBox object representing the bounding box of the point"""

        left, bottom, _ = self.points.min(axis=0)
        right, top, _ = self.points.max(axis=0)
        return BBox(left, bottom, right, top)

    def get_layer(self, layer: int) -> "PointCloud":
        """Returns a new PointCloud consisting of only points from a given layer.

        If 'layer' is positive, returns points with the corresponding return number.
        If 'layer' is negative, returns points with a return number counting from the last return,
        e.g. -1 filters to last-return points.

        Args:
            layer (int): The layer number. Positive values correspond to the return number,
                         negative values count from the last return.

        Returns:
            PointCloud: A new PointCloud object consisting of only points from the specified layer.

        Raises:
            ValueError: If the layer is zero.
        """

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


def fillholes(mat, radius: int = 1) -> np.ndarray:
    """Fills holes in the input matrix.

    For each element in 'mat' that is nan, this function fills it with the average of non-nan values within a given radius.

    Args:
        mat (np.ndarray): The input matrix with potential nan values.
        radius (int, optional): The radius within which to average non-nan values. Defaults to 1.

    Returns:
        np.ndarray: The input matrix with nan values filled.
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


def pointcloud_to_rasters(
    point_cloud: PointCloud,
    bbox: BBox,
    xres: int | float,
    yres: int | float,
    fill_radius: int,
) -> tuple[dict[str, np.ma.array], tuple[int, int]]:
    """Converts a PointCloud object into a pair of rasters.

    Args:
        point_cloud (PointCloud): The point cloud to convert.
        bbox (BBox): The bounding box to use for the conversion, in map units.
        xres (int | float): The resolution in the x direction, in map units.
        yres (int | float): The resolution in the y direction, in map units.
        fill_radius (int): The radius to use when filling holes, in pixels.

    Returns:
        tuple: A tuple of (rasters, shape). 'rasters' is a dictionary of rasters, with keys 'elev' and 'intensity'. 'shape' is a tuple of (n_cols, n_rows).
    """

    n_rows = int((bbox.top - bbox.bottom) / yres) + 1
    n_cols = int((bbox.right - bbox.left) / xres) + 1

    # get grid position of each point
    j = ((point_cloud.points[:, 0] - bbox.left) / xres).astype(int)
    i = ((bbox.top - point_cloud.points[:, 1]) / yres).astype(int)

    # set up nan-filled raster of the appropriate size
    elev = np.full((n_rows, n_cols), np.nan)
    intensity = np.full((n_rows, n_cols), np.nan)

    # fill in grid positions with elevation information
    # a large number of grid positions will not correspond
    # to any lidar points and, as a result, will have NaN values
    elev[i, j] = point_cloud.points[:, 2]
    intensity[i, j] = point_cloud.intensity

    elev = fillholes(elev, fill_radius)
    intensity = fillholes(intensity, fill_radius)

    return {
        "elev": np.ma.array(elev, mask=np.isnan(elev)),
        "intensity": np.ma.array(intensity, mask=np.isnan(intensity)),
    }, (n_cols, n_rows)


def lidar_to_rasters(
    las_filename: str,
    layers: Iterable[int],
    xres: int | float,
    yres: int | float,
    fill_radius: int,
) -> tuple[dict[str, np.ndarray], BBox, tuple[int, int]]:
    """Converts a LAS file to rasters.

    This function takes a LAS file and an iterable of layers, and converts each layer into a raster.
    The rasters are returned along with the bounding box and the shape of the rasters.

    Args:
        las_file (laspy.file.File): The LAS file to convert.
        layers (Iterable[int]): An iterable of numbers specifying which return values to convert into a layer.
                                For example, (1, 2, -1) would result in layers for the first, second, and last LIDAR return.
        xres (int | float): The resolution in the x direction.
        yres (int | float): The resolution in the y direction.
        fill_radius (int): The radius to use when filling holes.

    Returns:
        tuple: A tuple containing a dictionary mapping theme names to stacks of rasters, the bounding box, and a tuple containing the width and height of the rasters.
               Each stack of rasters has shape (num_rasters, m, n), where num_rasters is the length of the 'layers' arg.
    """

    # get lidar points, intensity, return num, and number of returns for each
    # pulse
    point_cloud = PointCloud.from_laspy(las_filename)

    # find bounding box of the data
    bbox = point_cloud.bbox

    # produce dict of {theme name -> list of rasters}, where the list has a raster
    # for each layer in 'layers'.
    raster_layers = defaultdict(list)
    shape = None
    for layer in layers:
        layer_pc = point_cloud.get_layer(layer)

        rasters, layer_shape = pointcloud_to_rasters(
            layer_pc, bbox, xres, yres, fill_radius
        )

        # make sure the shape of every layer is the same
        shape = shape or layer_shape
        assert shape == layer_shape

        for name, matrix in rasters.items():
            raster_layers[name].append(matrix)

    for name, layers in raster_layers.items():
        raster_layers[name] = np.stack(layers).astype(np.float32)

    return (raster_layers, bbox, shape)


def to_geotiff(
    themes: dict[str, np.ndarray],
    bbox: BBox,
    shape: tuple[int, int],
    crs: str,
    fn_out: str | IO,
) -> None:
    """Converts a processed LAS file to a GeoTiff.

    Args:
        themes (dict): A dictionary where the keys are theme names and the values are numpy arrays representing the theme layers with shape [n_layers, height, width].
        bbox (BoundingBox): A BoundingBox object that defines the spatial extent of the output GeoTiff.
        shape (tuple): A tuple defining the shape (width, height) of the output GeoTiff.
        crs (str): A string defining the coordinate reference system, e.g., "+init=epsg:2926".
        fn_out (str or file-like object): The filename our a file-like object to which to write the GeoTiff.

    Returns:
        None. The function writes directly to the file specified by `fn_out`.

    """

    width, height = shape
    transform = rio.transform.from_bounds(
        bbox.left, bbox.bottom, bbox.right, bbox.top, width, height
    )

    a_theme = list(themes.values())[0]

    n_themes = len(themes)
    n_layers = len(a_theme)

    with rio.open(
        fn_out,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=n_themes * n_layers,
        dtype=a_theme.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
        nodata=np.nan,
    ) as new_dataset:
        i = 1
        for theme_name, layers in themes.items():
            print(f"writing theme:'{theme_name}'")
            for layer in layers:
                new_dataset.write(layer, i)
                i += 1


def infer_raster_resolution(las_file: laspy.LasData) -> float:
    """
    Infers the raster resolution of a given LAS file.

    This function calculates the raster resolution by finding the number of points in the first return,
    calculating the area of the bounding box of the LAS file, and then finding the area per point.
    The resolution is then calculated as the width of a square with the same area as the area per point.

    Args:
        las_file (laspy.LasData): The LAS file for which to infer the raster resolution.

    Returns:
        float: The inferred raster resolution of the LAS file.
    """

    # find number of points in first return
    first_return_count = las_file.header.number_of_points_by_return[0]

    # find area of the bounding box of the LAS file
    left, bottom, _ = las_file.header.min
    right, top, _ = las_file.header.max
    area = (right - left) * (top - bottom)

    # find area per point
    specific_area = area / first_return_count

    res = round(specific_area**0.5 * SQRT_TWO, 2)

    return res


def las_to_raster(
    file_in: str,
    file_out: str,
    crs: str,
    return_num: list[int],
    theme: list[str],
    xres: float = None,
    yres: float = None,
    fill_radius: int = 2,
) -> None:
    """Convert LAS file to GeoTIFF raster.

    Args:
        file_in (str): Input LAS filename.
        file_out (str): Output GeoTIFF filename.
        crs (str): Coordinate reference system in whatever format
            is accepted by the rasterio file constuctor. E.g. 'epsg:2926'.
        return_num (list[int]): Return number(s) to rasterize, each in their own layer.
            Negative numbers indicate position relative to last return; e.g. -1 is the last return.
        theme (list[str]): Theme(s) to inclide. Choices are 'elev' and 'intensity'.
        xres (float, optional): Width of one pixel in output GeoTIFF, in the horizontal
            units of the CRS. If omitted, the LAS file will be used to make a reasonable guess.
        yres (float, optional): Height of one pizel in output GeoTIFF, in the horizontal
            units of the CRS. If omitted, the LAS file will be used to make a reasonable guess.
        fill_radius (int, optional): Fill raster holes with average values within FILL_RADIUS pixels.
    """

    with laspy.open(file_in) as f:
        las_data = f.read()

        default_res = infer_raster_resolution(las_data)

        xres = xres or default_res
        yres = yres or default_res

        themes, bbox, shape = lidar_to_rasters(
            las_data,
            layers=return_num,
            xres=xres,
            yres=yres,
            fill_radius=fill_radius,
        )

        themes = {
            theme_name: layers
            for theme_name, layers in themes.items()
            if theme_name in theme
        }

        to_geotiff(themes, bbox, shape, crs, file_out)
