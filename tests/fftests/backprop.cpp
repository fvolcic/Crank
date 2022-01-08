/**
 * @file backprop.cpp
 *
 * @brief This file contains all the test cases for testing backprop
 * @version 0.1
 * @date 2022-01-06
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "../../include/ff/ff.h"
#include "../unit_test_framework.h"
#include <vector>
#include <iostream> 

using namespace std;

TEST(basic_test_1){
    
    int num_layers = 3;
    std::vector<int> layer_sizes = {3, 2, 1};
    
    NeuralNetworkFF net(num_layers, layer_sizes); 

    vector<double> input = {0.5, 0.5, 0.5};
    vector<double> output = {1}; 

    net.train_on_example(input, output); 
    net.update_weights(0.5); 

    net.update_weights(0); 

    vector<double> final_output;
    input = {0, 0.1, 0}; 
    net.forwardPass(input, final_output); 

    std::cout << input[0] << std::endl; 

}

TEST_MAIN()