/**
 * @file mnist.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2021-12-28
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "../../include/MNIST/mnist.h"

MNIST_DATASET * read_dataset(){
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./data/");

    return new MNIST_DATASET(dataset); // ineffient :)
}