#include "Image.hpp"

#include <iostream>
#include <numbers>

namespace vision {
namespace image {

Image Image::Blur(size_t kernel_size, double sigma) {
  auto gaussian_kernel = [](size_t kernel_size, double sigma) {
    Eigen::MatrixXdRowMajor kernel(kernel_size, kernel_size);

    double center = (kernel_size - 1) / 2;
    for (size_t s = 0, num_rows = kernel.rows(), num_cols = kernel.cols();
         s < num_rows; ++s) {
      for (size_t t = 0; t < num_cols; ++t) {
        kernel(s, t) = exp(-0.5 * (pow((s - center) / sigma, 2.0) +
                                   pow((t - center) / sigma, 2.0))) /
                       (2 * std::numbers::pi * sigma * sigma);
      }
    }

    double sum = kernel.sum();
    for (size_t s = 0, num_rows = kernel.rows(), num_cols = kernel.cols();
         s < num_rows; ++s) {
      for (size_t t = 0; t < num_cols; ++t) {
        kernel(s, t) /= sum;
      }
    }

    return kernel;
  };

  auto pad = [](Eigen::MatrixXcRowMajor &data, size_t pad_width) {
    Eigen::MatrixXdRowMajor data_double = data.cast<double>();
    Eigen::MatrixXdRowMajor data_padded(data.rows() + 2 * pad_width,
                                        data.cols() + 2 * pad_width);
    data_padded.block((data_padded.rows() - data.rows() + 1) / 2,
                      (data_padded.cols() - data.cols() + 1) / 2, data.rows(),
                      data.cols()) = data_double;

    // Pad corners
    data_padded.topRightCorner(pad_width, pad_width)
        .setConstant(data_double(0, data.cols() - 1));
    data_padded.topLeftCorner(pad_width, pad_width)
        .setConstant(data_double(0, 0));
    data_padded.bottomLeftCorner(pad_width, pad_width)
        .setConstant(data_double(data.rows() - 1, 0));
    data_padded.bottomRightCorner(pad_width, pad_width)
        .setConstant(data_double(data.rows() - 1, data.cols() - 1));

    // Pad sides
    for (size_t i = 0; i < data.rows(); ++i) {
      data_padded.row(i + pad_width)
          .head(pad_width)
          .setConstant(data_double(i, 0));
      data_padded.row(i + pad_width)
          .tail(pad_width)
          .setConstant(data_double(i, data.cols() - 1));
    }
    for (size_t i = 0; i < data.cols(); ++i) {
      data_padded.col(i + pad_width)
          .head(pad_width)
          .setConstant(data_double(0, i));
      data_padded.col(i + pad_width)
          .tail(pad_width)
          .setConstant(data_double(data.rows() - 1, i));
    }

    return data_padded;
  };

  auto kernel = gaussian_kernel(kernel_size, sigma);
  auto data = pad(data_, (kernel_size - 1) / 2);

  Eigen::MatrixXcRowMajor output(data_.rows(), data_.cols());
  for (size_t y = 0, num_rows = data.rows(), num_cols = data.cols();
       y < num_cols - kernel_size + 1; ++y) {
    for (size_t x = 0; x < num_rows - kernel_size + 1; ++x) {
      output(x, y) = kernel
                         .cwiseProduct(data(Eigen::seq(x, x + kernel_size - 1),
                                            Eigen::seq(y, y + kernel_size - 1)))
                         .sum();
    }
  }

  return Image(output);
}

Eigen::MatrixXbRowMajor Image::Segment() {
  Eigen::MatrixXbRowMajor output = data_.cast<bool>();
  for (size_t i = 0, num_rows = output.rows(), num_cols = output.cols();
       i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      output(i, j) = 1 ? data_(i, j) < 240 : 0;
    }
  }

  return output;
}

Eigen::Vector2d Image::CenterOfGravity() {
  if (!data_segmented_) {
    data_segmented_ = Segment();
  }
  int num_points = (*data_segmented_).count();

  Eigen::Vector2d mean_vector{0, 0};
  for (size_t i = 0, num_rows = (*data_segmented_).rows(),
              num_cols = (*data_segmented_).cols();
       i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      mean_vector += Eigen::Vector2d{i, j} * (*data_segmented_)(i, j);
    }
  }
  return mean_vector / num_points;
}

Eigen::Matrix2d Image::PrincipalAxes() {
  if (!data_segmented_) {
    data_segmented_ = Segment();
  }
  int num_points = (*data_segmented_).count();
  auto mean_vector = CenterOfGravity();

  Eigen::Matrix2dRowMajor covariance_matrix{{0, 0}, {0, 0}};
  for (size_t i = 0, num_rows = (*data_segmented_).rows(),
              num_cols = (*data_segmented_).cols();
       i < num_rows; ++i) {
    for (size_t j = 0; j < num_cols; ++j) {
      if ((*data_segmented_)(i, j) == 0) {
        continue;
      }
      covariance_matrix += (Eigen::Vector2d{i, j} - mean_vector) *
                           (Eigen::Vector2d{i, j} - mean_vector).transpose();
    }
  }
  covariance_matrix = covariance_matrix / (num_points - 1);

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2dRowMajor> eigensolver;
  eigensolver.compute(covariance_matrix);
  Eigen::Vector2d eigen_values = eigensolver.eigenvalues().real().reverse();
  Eigen::Matrix2dRowMajor eigen_vectors =
      eigensolver.eigenvectors().real().rowwise().reverse();

  return eigen_vectors;
}

} // namespace image
} // namespace vision

bool operator==(const vision::image::Image &image1,
                const vision::image::Image &image2) {
  if (image1.data_ != image2.data_) {
    return false;
  }
  return true;
}
