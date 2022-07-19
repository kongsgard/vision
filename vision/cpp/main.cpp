#include <Eigen/Core>
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "Image.hpp"

using namespace vision;
using namespace image;

Image read_image(const std::filesystem::path &file_path) {
  std::ifstream infile(file_path, std::ios::binary);
  if (infile.good()) {
    std::vector<uint8_t> vector_buffer((std::istreambuf_iterator<char>(infile)),
                                       (std::istreambuf_iterator<char>()));
    std::vector<double> vector_buffer_double(vector_buffer.begin(),
                                             vector_buffer.end());
    Eigen::MatrixXd image_double =
        Eigen::Map<Eigen::Matrix<double, 480, 640, Eigen::RowMajor>>(
            vector_buffer_double.data());

    infile.close();
    return Image(image_double.cast<uint8_t>());
  } else {
    throw std::exception();
  }
}

void write_image(std::filesystem::path &file_path, const Image &image) {
  std::vector<uint8_t> image_data(image.data_.rows() * image.data_.cols());
  Eigen::MatrixXcRowMajor::Map(image_data.data(), image.data_.rows(),
                               image.data_.cols()) = image.data_;

  stbi_write_bmp(file_path.string().c_str(), image.data_.cols(),
                 image.data_.rows(), 1, image_data.data());
}

Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]");

int main(int argc, char *argv[]) {
  if (argc != 2) {
    throw std::runtime_error(
        "Please pass a filename as a command line argument.");
  }
  std::filesystem::path file_path = argv[1];
  Image image = read_image(file_path);
  write_image(file_path.replace_filename("plane_cpp.bmp"), image);

  Image image_blurred = image.Blur(5, 1.0);
  write_image(file_path.replace_filename("plane_blurred_cpp.bmp"),
              image_blurred);

  auto center_of_gravity = image.CenterOfGravity();
  std::cout << "Center of gravity: "
            << center_of_gravity.transpose().format(CleanFmt) << std::endl;

  auto principal_axes = image.PrincipalAxes();
  std::cout << "Principal axes: " << principal_axes.row(0).format(CleanFmt)
            << " and " << principal_axes.row(1).format(CleanFmt) << std::endl;

  return 0;
}
