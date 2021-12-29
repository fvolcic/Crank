/**
 * This example is how to use the MNIST dataset. 
 * 
 * To compile, use "g++ examples/MNIST/mnist_display.cpp src/MNIST/mnist.cpp -o examples/mnist/mnist_display_example"
 * To run, use "./examples/MNIST/mnist_display_example"
 * 
 */

#include "../../include/mnist/mnist.h"
#include <random>
#include <limits>
#include <iostream>

int main(){
    int index = 100; 
    MNIST_DATASET * dataset = read_dataset(); 
    dataset -> display_image(index, dataset->test_images); 
    std::cout << (int) dataset->test_labels[index] << std::endl;
}