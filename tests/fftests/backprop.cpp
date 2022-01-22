/**
 * @file backprop.cpp
 *
 * @brief This file contains all the test cases for testing backprop
 * @version 0.1
 * @date 2022-01-06
 *
 * @copyright Copyright (c) 2022
 *
 * @note To compile: 
 *          g++ tests/fftests/backprop.cpp src/ff/* -D NN_DEBUG -g3 -o bin/backprop_tests
 *       To run:
 *          ./bin/backprop_tests 
 * 
 */

#include "../../include/ff/ff.h"
#include "../unit_test_framework.h"
#include <vector>
#include <iostream> 

using namespace std;

TEST(basic_test_1){
    
    int num_layers = 3;
    std::vector<int> neuron_counts = {1, 1, 1}; 
    std::vector<std::vector<std::vector< double >>> weights = {
                                                             {{}}, // No weights for layer 1
                                                             {{ 0.5 }}, // Weights going into layer 2
                                                             {{ 0.5 }}}; // Weights going into layer 3
    std::vector< std::vector< double >> bias = { {}, 
                                                {0.5}, // Bias for neurons in layer 1 
                                                {0.5}}; // Bias for neurons in layer 2

    NeuralNetworkFF net(num_layers, neuron_counts, weights, bias); 

    std::vector< double > sample_input_1 = {1}; 
    std::vector< double > sample_output_1 = {1}; 

    std::vector< double > sample_input_2 = {0};
    std::vector< double > sample_output_2 = {0}; 

    net.train_on_example( sample_input_1, sample_output_1 ); 

    ASSERT_ALMOST_EQUAL(net.neurons.back()[0].get_dLoss_dBias(), -0.123485, 0.00001); // The bias B_21
    ASSERT_ALMOST_EQUAL(net.neurons.back()[0].get_dLoss_dWeight()[0], -0.090275, 0.000001); // The weight W_21
    ASSERT_ALMOST_EQUAL(net.neurons[1][0].get_dLoss_dBias(), -0.012139, 0.000001); // The bias B_11
    ASSERT_ALMOST_EQUAL(net.neurons[1][0].get_dLoss_dWeight()[0], -0.01213935, 0.00000001); // The weight W_11


}

TEST(basic_test_2){
    int num_layers = 3;
    std::vector<int> neuron_counts = {2, 2, 2};
    std::vector<std::vector<std::vector< double >>> weights = {
                                                             {{}}, // No weights for layer 1
                                                             {{ 0.5 }, { 0.5 }, { 0.5 }, { 0.5 }}, // Weights going into layer 2
                                                             {{ 0.5 }, { 0.5 }, { 0.5 }, { 0.5 }}}; // Weights going into layer 3
    
     std::vector< std::vector< double >> bias = { {}, 
                                                { 0.5, 0.5 }, // Bias for neurons in layer 1 
                                                { 0.5, 0.5 }}; // Bias for neurons in layer 2

}

TEST_MAIN()