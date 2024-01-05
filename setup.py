from setuptools import setup, find_packages

setup(
    name="lasrasterize",
    version="0.1",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lasrasterize = lasrasterize.cli:main",
        ],
    },
    install_requires=["laspy==2.5.1", "rasterio==1.3.9", "numpy==1.26.2"],
    author="Brandon Martin-Anderso",
    author_email="badhill@gmail.com",
    description="A command line tool and associated library used to convert lidar LAS files into GeoTIFF raster files.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bmander/lasrasterize",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
