/**
 * @file mnist.h
 * 
 * @brief This contains functions that let ypu read from the MNIST digits dataset
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "mnist/include/mnist/mnist_reader.hpp"

#define MNIST_DATASET_LOCATION "/home/fvolcic/NeuralNetworks/data"

using MNIST_DATASET = mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t>; 

/**
 * @brief Read the MNIST dataset into memory for network training
 * 
 * @return mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> *
 * @note The returned MNIST_DATASET is dynamically allocated
 * 
 */
MNIST_DATASET * read_dataset();

