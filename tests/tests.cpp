#include <../include/tensor.h>
#include <gtest/gtest.h>

TEST(TensorTest, Creation)
{
    // Scalar
    Tensor tensor = Tensor(5.0);
    EXPECT_EQ(tensor.shape(), std::vector<std::size_t>({}));
    EXPECT_THROW(tensor(0), std::invalid_argument);
    EXPECT_EQ(tensor.item(), 5.0);

    // 1D
    std::vector<float> v = {1.0, 2.0, 3.0};
    Tensor tensor2 = Tensor(v);
    EXPECT_EQ(tensor2.shape(), std::vector<std::size_t>({3}));
    EXPECT_EQ(tensor2(0), 1.0);
    EXPECT_EQ(tensor2(1), 2.0);
    EXPECT_EQ(tensor2(2), 3.0);
    EXPECT_THROW(tensor2(3), std::invalid_argument);
    EXPECT_THROW(tensor2.item(), std::runtime_error);

    // 2D
    std::vector<std::vector<float>> v_2 = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    Tensor tensor3 = Tensor(v_2);
    EXPECT_EQ(tensor3.shape(), std::vector<std::size_t>({2, 3}));
    EXPECT_EQ(tensor3.stride(), std::vector<std::size_t>({3, 1}));
    EXPECT_EQ(tensor3(0, 0), 1.0);
    EXPECT_EQ(tensor3(0, 1), 2.0);
    EXPECT_EQ(tensor3(0, 2), 3.0);
    EXPECT_EQ(tensor3(1, 0), 4.0);
    EXPECT_EQ(tensor3(1, 1), 5.0);
    EXPECT_EQ(tensor3(1, 2), 6.0);
    EXPECT_THROW(tensor3(2, 0), std::invalid_argument);
    EXPECT_THROW(tensor3(0, 3), std::invalid_argument);
    EXPECT_THROW(tensor3.item(), std::runtime_error);
}