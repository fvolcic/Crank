/**
 * This example is how to use the MNIST dataset. 
 * 
 * To compile, use "g++ examples/MNIST/mnist_display.cpp src/mnist/mnist.cpp -o examples/MNIST/mnist_display_example"
 * To run, use "./examples/MNIST/mnist_display_example"
 * 
 */

#include "../../include/mnist/mnist.h"
#include <iostream>

int main(){

    int index = 100; 
    
    // Load the MNIST dataset into memory
    // It is assumed that the MNIST files are in the data directory 
    MNIST_DATASET * dataset = read_dataset(); 

    // Get and display an image from the testing set
    dataset -> display_image(index, dataset->test_images); 
    std::cout << (int) dataset->test_labels[index] << std::endl;

    // Get and display an image from the training set
    dataset -> display_image(index, dataset->training_images); 
    std::cout << (int) dataset->training_labels[index] << std::endl;
}