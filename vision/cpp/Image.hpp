#pragma once

#include <Eigen/Dense>
#include <optional>

namespace Eigen {
typedef Eigen::Matrix<double, 2, 2, Eigen::RowMajor> Matrix2dRowMajor;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixXdRowMajor;
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixXcRowMajor;
typedef Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    MatrixXbRowMajor;
} // namespace Eigen

namespace vision {
namespace image {

class Image {
public:
  Image(Eigen::MatrixXcRowMajor data) : data_(data){};

  Image Blur(size_t kernel_size, double sigma);
  Eigen::MatrixXbRowMajor Segment();
  Eigen::Vector2d CenterOfGravity();
  Eigen::Matrix2d PrincipalAxes();

  Eigen::MatrixXcRowMajor data_;
  std::optional<Eigen::MatrixXbRowMajor> data_segmented_;
};

} // namespace image
} // namespace vision

bool operator==(const vision::image::Image &image1,
                const vision::image::Image &image2);
