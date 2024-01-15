import argparse

from .lib import Layerdef, lasfile_to_geotiff


def main():
    """Main function."""

    parser = argparse.ArgumentParser(
        description="Convert LAS file to GeoTIFF raster. The output GeoTIFF "
        "will have one band for each layer definition. A layer definition "
        "consists of a return number and a theme. The theme can be 'elev' in "
        "which case the band will be a float32 elevation in the units of the "
        "CRS, or 'intensity'. If the return number is positive it indicates "
        "the absolute return number. If the return number is negative it "
        "indicates the position relative to the last return; e.g. -1 is the "
        "last return.",
        epilog="Examples: \n"
        "\tlasrasterize --crs epsg:2285 -n 1 -t elev -n -1 -t elev -n -1 -t "
        "intensity /poth/to/lasfile.las /path/to/raster.tif\n"
        "\t:: Create a GeoTIFF with three layers: the first return elevation, "
        "the last return elevation, and the last return intensity.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("file_in", help="Input LAS filename.")
    parser.add_argument("file_out", help="Output GeoTIFF filename.")
    parser.add_argument(
        "--crs",
        help="Coordinate reference system of the LAS file and output GeoTIFF. "
        "Accepts a string in any format accepted by the rasterio file "
        "constructor. E.g. 'epsg:2926'. If omitted the CRS will be read from "
        "the LAS file, but the LAS file may not have a CRS, or may have the "
        "wrong CRS.",
    )
    parser.add_argument(
        "-n",
        "--return_num",
        required=True,
        action="append",
        type=int,
        help="Return number(s) to rasterize, each in their own layer. "
        "Negative numbers indicate "
        "position relative to last return; e.g. -1 is the last return.",
    )
    parser.add_argument(
        "-t",
        "--theme",
        required=True,
        action="append",
        type=str,
        choices=["elev", "intensity"],
        help="Theme(s) to inclide. Choices are 'elev' and 'intensity'.",
    )
    parser.add_argument(
        "--xres",
        type=float,
        default=None,
        help="Width of one pixel in output GeoTIFF, in the horizontal"
        " units of the CRS. If omitted, the LAS file will be used to"
        " make a reasonable guess.",
    )
    parser.add_argument(
        "--yres",
        type=float,
        default=None,
        help="Height of one pizel in output GeoTIFF, in the horizontal"
        " units of the CRS. If omitted, the LAS file will be used to"
        " make a reasonable guess.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="gridandfill",
        choices=["gridandfill", "nearest", "linear", "cubic"],
        help="Interpolation strategy. 'gridandfill' is the default and will "
        "value a pixel as the average of all points falling withn that pixel,"
        " and then fill holes with the average of neighboring pixels. "
        "'nearest', 'linear', and 'cubic' will interpolate a grid using "
        "scipy.interpolate.griddata with the specified method.",
    )
    parser.add_argument(
        "--fill_radius",
        "-r",
        type=int,
        default=2,
        help="Fill raster holes with average values within FILL_RADIUS "
        "pixels.",
    )

    args = parser.parse_args()

    # make a list of layer definitions
    if len(args.return_num) != len(args.theme):
        raise ValueError(
            "The number of return numbers must match the number of themes."
        )

    layer_defs = []
    for return_num, theme in zip(args.return_num, args.theme):
        if theme not in ("elev", "intensity"):
            raise ValueError("Theme must be 'elev' or 'intensity'.")

        layer_defs.append(
            Layerdef(
                pulse_return=return_num,
                intensity=theme == "intensity",
            )
        )

    lasfile_to_geotiff(
        args.file_in,
        args.file_out,
        layer_defs,
        args.xres,
        args.yres,
        args.crs,
        args.strategy,
        fill_radius=args.fill_radius,
    )


if __name__ == "__main__":
    main()
