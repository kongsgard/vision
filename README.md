# Vision

A collection of some image processing and point cloud processing algorithms.

## Installation

1. Clone the repository:

```
git clone https://github.com/kongsgard/vision
```

2. Follow the instructions on https://python-poetry.org/docs/#installation to install poetry. Next, install the project dependencies:

```
poetry install
```

All set!

## Usage

Run the CLI with the following command:

```
python vision
```

## Tests

Tests are implemented using pytest and can be run all together by this simple command:

```
pytest
```

## Alternative C++ Implementation

Some image processing functions are implemented directly in C++. See the [C++ README](vision/cpp/README.md) for more information.

## Point Cloud Segmentation

Before:
![Pineapple on table](/images/pineapple_on_table.png)

After:
![Segmented pineapple](/images/pineapple_segmented.png)

## Aside: Python Point Cloud Processing Libraries

- [Open3D](https://github.com/intel-isl/Open3D)
- [PyVista](https://github.com/pyvista/pyvista)
- [PCL](https://github.com/PointCloudLibrary)
- [pclpy](https://github.com/davidcaron/pclpy)
- [pyntcloud](https://github.com/daavoo/pyntcloud)
- [libLAS](https://github.com/libLAS)
- [PDAL](https://github.com/PDAL)
- [Laspy](https://github.com/grantbrown/laspy)
- [plyfile](https://github.com/dranjan/python-plyfile)
- [point_cloud_utils](https://github.com/fwilliams/point-cloud-utils)
- [pptk](https://github.com/heremaps/pptk)
- [PyLidar](https://github.com/lakshmanmallidi/PyLidar3)
- [pylas](https://github.com/perrygeo/pylas)
- [LAS](https://github.com/WarrenWeckesser/las)
