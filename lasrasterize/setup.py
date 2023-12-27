from setuptools import setup, find_packages

setup(
    name='lasrasterize',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'lasrasterize = lasrasterize.cli:main',
        ],
    },
    install_requires=[
        # Add your project dependencies here
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A command line tool and associated library used to convert lidar LAS files into geotiff raster files.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/lasrasterize',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)