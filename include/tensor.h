#pragma once
#include <iostream>
#include <vector>

class Tensor
{
private:
    std::vector<float> _data;
    std::vector<std::size_t> _shape;
    std::vector<std::size_t> _stride; // tells you how much to jump in flattened array when moving down a row/col

public:
    Tensor(float data);                           // 0-D tensor (scalar)
    Tensor(std::vector<float> data);              // 1-D tensor
    Tensor(std::vector<std::vector<float>> data); // 2-D tensor
    const float &item() const; // Read-only variant of getting a value of a tensor with a single element
    float &item();             // Write variant of getting a value of a tensor with a single element
    // &operator() allows for read/write access for 1-D and 2D arrays using parentheses indexing
    const float &operator()(std::size_t i) const;
    float &operator()(std::size_t i);
    const float &operator()(std::size_t i, std::size_t j) const;
    float &operator()(std::size_t i, std::size_t j);
    const std::vector<std::size_t> &shape() const;
    const std::vector<std::size_t> &stride() const;

    // Allows tensors to be printed correctly using std::cout<< (friend means the function can access private variables)
    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj);
};