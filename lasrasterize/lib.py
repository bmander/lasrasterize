from collections import namedtuple
from typing import Iterable, Optional, Union

import laspy
import numpy as np
import rasterio as rio
from scipy import ndimage as nd


BBox = namedtuple("BBox", ["left", "bottom", "right", "top"])


class Layerdef(namedtuple("Laslayer_definition",
                          ["pulse_return", "intensity"])):
    """Defines a layer of a LAS file.

    Args:
        pulse_return (int): The pulse return included in this layer. Positive
          values count from the first return, negative values count from the
          last return.
        intensity (bool): Whether to output a raster of intensity values."""


def resolution(p: float, rho: float) -> float:
    """Find the length of the side of a square that has probability p of
    containing at last one point given a point density of rho. Derived from
    the poisson distribution.
    """
    return np.sqrt(np.log(1 - p) / -rho)


def infer_raster_resolution(lasdata: laspy.LasData, p: float = 0.95) -> float:
    """
    Infers the raster resolution of a given LAS file.

    This function uses the first return of the LAS file to infer the raster
    resolution. The resolution is chosen such that, assuming a uniform
    distribution of points, the probability that a given raster cell will
    contain at least one point is equal to the given probability p. For a
    point density of 1.0 and p=0.95, a pixel width will be about 1.73.

    Args:
        lasdata (laspy.LasData): The LAS file for which to infer the raster
          resolution.
        p (float, optional): The probability that a given raster cell will
          contain at least one point.

    Returns:
        float: The inferred raster resolution of the LAS file.
    """

    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")

    # find number of points in first return
    first_return_count = lasdata.header.number_of_points_by_return[0]

    # find area of the bounding box of the LAS file
    left, bottom, _ = lasdata.header.min
    right, top, _ = lasdata.header.max
    area = (right - left) * (top - bottom)

    # find density of points in first return
    density = first_return_count / area

    return resolution(p, density)


def fillholes(mat, radius: int = 1) -> np.ndarray:
    """Fills holes in the input matrix.

    For each element in 'mat' that is nan, this function fills it with the
    average of non-nan values within a given radius.

    Args:
        mat (np.ndarray): The input matrix with potential nan values.
        radius (int, optional): The radius within which to average non-nan
          values. Defaults to 1.

    Returns:
        np.ndarray: The input matrix with nan values filled.
    """

    if radius < 0:
        raise ValueError("Radius cannot be negative.")

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


def lasdata_to_rasters(
    lasdata: laspy.LasData,
    bbox: BBox,
    layer_defs: Iterable[Layerdef],
    xres: Union[int, float],
    yres: Union[int, float],
    fill_holes: bool = True,
    fill_radius: int = 2,
) -> np.ndarray:
    """Converts a lasdata object to a raster.

    Args:
        lasdata (laspy.LasData): LasData object to convert.
        bbox (BBox): The bounding box to use for the conversion, in map units.
        xres (int | float): The resolution in the x direction, in map units.
        yres (int | float): The resolution in the y direction, in map units.
        layer_defs (Iterable[Laslayer_definition]): An iterable of
          Laslayer_definition objects, each defining a layer to output.
        fill_holes (bool, optional): Whether to fill holes in the raster.
          Defaults to True.
        fill_radius (int, optional): The radius to use when filling holes, in
          pixels.

    Returns:
        np.ndarray: An float array containing the elevation or intensity
          raster, with shape (n_layers, m, n). Null values are filled with
          np.nan.
    """

    n_rows = int((bbox.top - bbox.bottom) / yres) + 1
    n_cols = int((bbox.right - bbox.left) / xres) + 1

    i = ((bbox.top - np.array(lasdata.y)) / yres).astype(int)
    j = ((np.array(lasdata.x) - bbox.left) / xres).astype(int)

    # set up nan-filled raster of the appropriate size
    rasters = np.full((len(layer_defs), n_rows, n_cols), np.nan)

    for k, layer_def in enumerate(layer_defs):
        # get a mask to filter out points that don't belong in this layer
        if layer_def.pulse_return < 0:
            abs_pulse_return = lasdata.num_returns + layer_def.pulse_return + 1
        else:
            abs_pulse_return = layer_def.pulse_return
        mask = (lasdata.return_num == abs_pulse_return).astype(bool)

        # get grid position of each point
        i_layer, j_layer = i[mask], j[mask]

        # set up nan-filled raster of the appropriate size
        raster = np.full((n_rows, n_cols), np.nan)

        if layer_def.intensity:
            values = lasdata.intensity[mask]
        else:
            values = lasdata.z[mask]

        # fill in grid positions with elevation information
        # a large number of grid positions will not correspond
        # to any lidar points and, as a result, will have NaN values
        raster[i_layer, j_layer] = values

        if fill_holes:
            raster = fillholes(raster, fill_radius)

        rasters[k] = raster

    return rasters


def lasfile_to_geotiff(
    las_filename: str,
    geotiff_filename: str,
    layer_defs: Iterable[Layerdef],
    xres: Optional[Union[int, float]] = None,
    yres: Optional[Union[int, float]] = None,
    fill_radius: int = 2,
    crs: str = None,
) -> None:
    """Converts a LAS file to a GeoTiff.

    Args:
        las_filename (str): The path to the LAS file to convert.
        geotiff_filename (str): The path to the GeoTiff to output.
        layer_defs (Iterable[Laslayer_definition]): An iterable of
          Laslayer_definition objects, each defining a layer to output.
        xres (int | float | None, optional): The resolution in the x
          direction, in map units. If None, the resolution will be inferred
          from the LAS file. Defaults to None.
        yres (int | float | None, optional): The resolution in the y
          direction, in map units. If None, the resolution will be inferred
          from the LAS file. Defaults to None.
        fill_radius (int, optional): The radius to use when filling holes, in
          pixels.
        crs (str, optional): The CRS of the output GeoTiff. If None, the CRS
          will be inferred from the LAS file. Defaults to None.

    Raises:
        ValueError: If xres or yres is negative.
    """

    if xres is not None and xres < 0:
        raise ValueError("xres cannot be negative")
    if yres is not None and yres < 0:
        raise ValueError("yres cannot be negative")

    lasdata: laspy.LasData = laspy.read(las_filename)

    # find bounding box of the data
    bbox = BBox(
        lasdata.header.x_min,
        lasdata.header.y_min,
        lasdata.header.x_max,
        lasdata.header.y_max,
    )

    if xres is None or yres is None:
        xres = yres = infer_raster_resolution(lasdata)

    rasters = lasdata_to_rasters(lasdata, bbox, layer_defs, xres, yres,
                                 fill_radius)

    if crs is None:
        crs = lasdata.header.parse_crs()

    n_layers, height, width = rasters.shape
    transform = rio.transform.from_bounds(
        bbox.left, bbox.bottom, bbox.right, bbox.top, width, height
    )

    with rio.open(
        geotiff_filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=n_layers,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        compress="lzw",
        nodata=np.nan,
    ) as new_dataset:
        for i, layer in enumerate(rasters):
            new_dataset.write(layer, i + 1)
