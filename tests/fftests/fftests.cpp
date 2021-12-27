#include "../unit_test_framework.h"
#include "../../include/ff/ff.h"
#include <vector> 
#include <cmath> 

TEST(forward_pass_1){

    int num_layers = 2;
    std::vector<int> neuron_counts = {1, 1}; 
    std::vector<std::vector<std::vector< double >>> weights = {{{}}, {{ 1 }}};
    std::vector<std::vector< double >> bias = { {}, {1} }; 

    // Construct the neural network with the given parameters 
    NeuralNetworkFF net(num_layers, neuron_counts, weights, bias); 

    // After the forward pass
    std::vector<double> input = {1}; 
    std::vector<double> output = net.forwardPass(input); 

    // Computed assuming that the input is treated as the output of the first layer
    ASSERT_ALMOST_EQUAL(output[0], 0.88, 0.01); 
}

TEST(forward_pass_2){
    int num_layers = 3;
    std::vector<int> neuron_counts = {1, 1, 1}; 
    std::vector<std::vector<std::vector< double >>> weights = {{{}},{{ 1 }}, {{ 0.5 }}};
    std::vector<std::vector< double >> bias = { {}, {1}, {1} }; 

    // Construct the neural network with the given parameters 
    NeuralNetworkFF net(num_layers, neuron_counts, weights, bias); 

    // After the forward pass
    std::vector<double> input = {1}; 
    std::vector<double> output = net.forwardPass(input); 

    // Computed assuming that the input is treated as the output of the first layer
    ASSERT_ALMOST_EQUAL(output[0], 0.808, 0.001); 
}

TEST(forward_pass_3){
    //Parameters
    int num_layers = 3;
    std::vector<int> neuron_count = {2, 2, 2};
    std::vector<std::vector<std::vector< double >>> weights = {{}, {{1, 1}, {1, 1}}, {{1, 1}, {1, 1}}};
    std::vector<std::vector< double >> bias = { {}, {2, 2}, {2, 2}};

    //Construct the neural network with given parameters
    NeuralNetworkFF n3(num_layers, neuron_count, weights, bias);

    //After the forward pass
    std::vector<double> input = {1, 1};
    std::vector<double> output = n3.forwardPass(input);

    ASSERT_ALMOST_EQUAL(output[0], sigmoid(2 + 2 * sigmoid(4)), 0.01);
    ASSERT_ALMOST_EQUAL(output[1], sigmoid(2 + 2 * sigmoid(4)), 0.01); 
}

// Helper

double sigmoid(double x){
    return 1 / (1 + exp(-x)); 
}

TEST(or_gate_forward_pass){
    int num_layers = 2;
    std::vector<int> neuron_counts = {2, 1}; 
    std::vector<std::vector<std::vector< double >>> weights = {{}, {{ 1, 1 }}};
    std::vector<std::vector< double >> bias = { {}, {-10} }; 

    // Construct the neural network with the given parameters 
    NeuralNetworkFF net(num_layers, neuron_counts, weights, bias); 

    // After the forward pass
    std::vector<double> input1 = {0, 0}; 
    std::vector<double> output1 = net.forwardPass(input1); 

    std::vector<double> input2 = {1, 0}; 
    std::vector<double> output2 = net.forwardPass(input2);

    std::vector<double> input3 = {0, 1}; 
    std::vector<double> output3 = net.forwardPass(input3);

    std::vector<double> input4 = {1, 1}; 
    std::vector<double> output4 = net.forwardPass(input4);

    ASSERT_ALMOST_EQUAL(output1[0], sigmoid(-10), 0.1); 
    ASSERT_ALMOST_EQUAL(output2[0], sigmoid(10), 0.1);
    ASSERT_ALMOST_EQUAL(output3[0], sigmoid(10), 0.1);
    ASSERT_ALMOST_EQUAL(output4[0], sigmoid(30), 0.1);

}

TEST_MAIN(); 