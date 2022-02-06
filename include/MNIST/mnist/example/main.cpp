//=======================================================================
// Copyright (c) 2017 Adrian Schneider
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

#include <iostream>
#include "mnist/mnist_reader.hpp"

int main(int argc, char* argv[]) {
    // MNIST_DATA_LOCATION set by MNIST cmake config
    std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
        mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);

    std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
    std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
    std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
    std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

    for(int k = 0; k < 5; ++k){

        std::cout << dataset.test_images[k].size() << std::endl; 

    for(int i = 0; i < 28; ++i){
        for(int j = 0; j < 28; ++j){
            if(dataset.test_images[k][28 * i + j] != 0
            || j == 0
            || j == 27
            ){
                std::cout << "*"; 
            }else{
                std::cout << " "; 
            }
            
        }
        std::cout << std::endl; 
    }
    std::cout << "label: " << (int)dataset.test_labels[k] << std::endl; 

    }
    return 0;
}
