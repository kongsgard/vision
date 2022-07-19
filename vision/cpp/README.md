The code in this directory serves as an alternate C++ implementation of the image processing algorithms.

## Installation

Navigate to this directory in a terminal, and then run the following commands:

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

## Usage

Navigate to the `build/Release` directory, and then run this command:

```
./Main.exe <path-to-image-file>
```
