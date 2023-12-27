# lasrasterize

`lasrasterize` is a command line tool and associated library used to convert lidar LAS files into geotiff raster files.

## Installation

To install `lasrasterize`, clone the repository and run the setup script:

```bash
git clone https://github.com/yourusername/lasrasterize.git
cd lasrasterize
python setup.py install
```

## Usage

You can use `lasrasterize` from the command line as follows:

```bash
lasrasterize input.las output.tif
```

This will convert the input LAS file into a geotiff raster file.

## Development

To run the unit tests, use the following command:

```bash
python -m unittest discover lasrasterize
```

## License

`lasrasterize` is released under the MIT license. See the LICENSE file for details.