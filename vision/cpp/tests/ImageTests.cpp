#include "../Image.hpp"
#include <gtest/gtest.h>

using namespace vision;
using namespace image;

TEST(ImageTests, GaussianBlur) {
  Image image(Eigen::MatrixXcRowMajor{
      {0, 10, 0, 10}, {10, 0, 10, 0}, {0, 10, 0, 10}, {10, 0, 10, 0}});
  auto image_blurred = image.Blur(5, 1.0);

  Image image_blurred_expected(Eigen::MatrixXcRowMajor{
      {3, 4, 5, 6}, {4, 4, 5, 5}, {5, 5, 4, 4}, {6, 5, 4, 3}});

  ASSERT_TRUE(image_blurred_expected == image_blurred);
}

TEST(ImageTests, Segment) {
  Image image(
      Eigen::MatrixXcRowMajor{{240, 245, 5}, {150, 230, 210}, {250, 250, 250}});
  auto data_segmented = image.Segment();

  Eigen::MatrixXbRowMajor data_segmented_expected{
      {0, 0, 1}, {1, 1, 1}, {0, 0, 0}};

  ASSERT_TRUE(data_segmented_expected == data_segmented);
}

TEST(ImageTests, CenterOfGravity) {
  Image image(
      Eigen::MatrixXcRowMajor{{255, 0, 0}, {255, 255, 255}, {255, 0, 0}});
  Eigen::Vector2d center_of_gravity = image.CenterOfGravity();

  Eigen::Vector2d center_of_gravity_expected{1.0, 1.5};

  ASSERT_TRUE(center_of_gravity_expected == center_of_gravity);
}

TEST(ImageTests, PrincipalAxes) {
  Image image(
      Eigen::MatrixXcRowMajor{{255, 255, 255}, {255, 0, 0}, {255, 255, 255}});
  auto eigen_vectors = image.PrincipalAxes();

  Eigen::Matrix2d eigen_vectors_expected{{0, 1}, {1, 0}};

  ASSERT_TRUE(eigen_vectors_expected == eigen_vectors);
}
