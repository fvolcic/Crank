/**
 * @file ff.cpp
 * 
 * @brief 
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "../../include/ff/ff.h"
#include "../../include/ff/learning_functions.h"
#include "../../include/utils.h"
#include <limits>
#include <iostream>
#include <sstream>

std::vector<std::vector<double>> random_bias_helper(std::vector<int> neuron_counts)
{
    std::vector<std::vector<double>> output;
    output.resize(neuron_counts.size());

    for (int i = 0; i < output.size(); ++i)
    {
        output[i].resize(neuron_counts[i]);
    }

    for (int i = 0; i < output.size(); ++i)
    {
        for (int j = 0; j < output[i].size(); ++j)
        {
            output[i][j] = random_range(-0.3, 0.3);
        }
    }

    return output;
}

std::vector<std::vector<std::vector<double>>> random_weight_helper(std::vector<int> neuron_counts)
{

    std::vector<std::vector<std::vector<double>>> output;
    output.resize(neuron_counts.size());

    for (int i = 0; i < output.size(); ++i)
    {
        output[i].resize(neuron_counts[i]);
    }

    for (int i = 1; i < output.size(); ++i)
    {
        for (int j = 0; j < output[i].size(); ++j)
        {
            output[i][j].resize(neuron_counts[i - 1]);
            for (int k = 0; k < output[i][j].size(); ++k)
            {
                output[i][j][k] = random_range(-0.3, 0.3);
            }
        }
    }

    return output;
}

NeuralNetworkFF::NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts)
    : NeuralNetworkFF(num_layers, neuron_counts, random_weight_helper(neuron_counts), random_bias_helper(neuron_counts))
{
}

NeuralNetworkFF::NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts, const std::vector<std::vector<std::vector<double>>> &weights, const std::vector<std::vector<double>> &bias)
{
    //Initializes first layer of Neural Network
    neurons.resize(num_layers);
    neurons[0].resize(neuron_counts[0]);
    for (int i = 0; i < neuron_counts[0]; ++i)
    {
        neurons[0][i] = Neuron();
    }

    for (int x = 1; x < neuron_counts.size(); ++x)
    {
        neurons[x].resize(neuron_counts[x]);
        for (int y = 0; y < neuron_counts[x]; ++y)
        {
            neurons[x][y].setBias(bias[x][y]);
            neurons[x][y].setWeights(weights[x][y]);
        }
    }
}

NeuralNetworkFF::~NeuralNetworkFF() {}

void NeuralNetworkFF::forwardPass(std::vector<double> &input, std::vector<double> &output)
{

    if (maxLayerSize == -1)
        findMaxLayerSize();

    std::vector<double> intermediate_result;
    intermediate_result.resize(maxLayerSize);

    // Setup all the input values for the neural network
    for (int i = 0; i < neurons[0].size(); ++i)
    {
        neurons[0][i].setOutput(input[i]);
        intermediate_result[i] = input[i];
    }

    // Compute the forward pass for the network
    for (int i = 1; i < neurons.size(); ++i)
    {
        for (int j = 0; j < neurons[i].size(); ++j)
        {
            neurons[i][j].computeInput(intermediate_result);
        }

        for (int j = 0; j < neurons[i].size(); ++j)
        {
            intermediate_result[j] = neurons[i][j].getOutput();
        }
    }

    for (int i = 0; i < neurons.back().size(); ++i)
    {
        output.push_back(intermediate_result[i]);
    }
}

std::vector<double> NeuralNetworkFF::forwardPass(std::vector<double> &input)
{
    std::vector<double> output;
    forwardPass(input, output);
    return output;
}

void NeuralNetworkFF::findMaxLayerSize()
{
    maxLayerSize = 0;
    for (int i = 0; i < neurons.size(); ++i)
    {
        if (maxLayerSize < neurons[i].size())
        {
            maxLayerSize = neurons[i].size();
        }
    }
}

////////////////////////////////////////////////////////////////////
// BELOW ARE THE TRAINING AND BACK PROP FUNCTIONS                 //
////////////////////////////////////////////////////////////////////

void NeuralNetworkFF::train_on_example(std::vector<double> &input, std::vector<double> &expected_output)
{

    std::vector<double> output;
    forwardPass(input, output);

    //BACK PROP PORTION

    // Step 1: compute dLoss/dActivation for the final layer in the network
    for (int i = 0; i < output.size(); ++i)
    {
        double dLoss_dAct_val = 2 * (neurons.back()[i].getActivation() - expected_output[i]);
        neurons.back()[i].set_dLoss_dActivation(dLoss_dAct_val);
    }

    // Step 2: compute dActivation_dInput for the final layer in the network
    for (int i = 0; i < output.size(); ++i)
    {
        Neuron &neuron = neurons.back()[i];
        neuron.set_dActivation_dInput(neuron.getActivationFunction()->derivative(neuron.getInput()));
    }

    // Now compute derivate of the bias and the weights for the first layer in the neural network
    for (int i = 0; i < output.size(); ++i)
    {
        Neuron &neuron = neurons.back()[i];
        calculate_dLoss_dWeight_and_dLoss_dBias(neuron, neurons.size() - 1);
    }

    // Call the backprop to train rest of the network
    back_propagation(get_num_layers() - 2);
}

void NeuralNetworkFF::back_propagation(int layer)
{
    // Calculate dLoss_dActivation for the current layer
    for (int i = 0; i < neurons[layer].size(); ++i)
    {
        Neuron &neuron = neurons[layer][i];
        this->calculate_dLoss_dActivation(neuron, layer, i);
    }

    // Step 2: compute dActivation_dInput
    for (int i = 0; i < neurons[layer].size(); ++i)
    {
        Neuron &neuron = neurons[layer][i];
        calculate_dActivation_dInput(neuron, layer);
        // neuron.set_dActivation_dInput( neuron.getActivationFunction()->derivative( neuron.getInput() ));
    }

    // Now compute derivate of the bias and the weights for the first layer in the neural network
    for (int i = 0; i < neurons[layer].size(); ++i)
    {
        Neuron &neuron = neurons[layer][i];
        calculate_dLoss_dWeight_and_dLoss_dBias(neuron, layer);
    }

    if (layer != 1)
        back_propagation(layer - 1);
}

void NeuralNetworkFF::calculate_dActivation_dInput(Neuron &neuron, int layer)
{
    neuron.set_dActivation_dInput(neuron.getActivationFunction()->derivative(neuron.getInput()));
}

void NeuralNetworkFF::calculate_dLoss_dActivation(Neuron &neuron, int layer, int index)
{
    double dL_dA = 0;

    for (int j = 0; j < neurons[layer + 1].size(); ++j)
    {
        Neuron &neuron = neurons[layer + 1][j];
        dL_dA += neuron.weights[index] * neuron.get_dActivation_dInput() * neuron.get_dLoss_dActivation();
    }

    neurons[layer][index].set_dLoss_dActivation(dL_dA);
}

// void NeuralNetworkFF::calculate_dLoss_dBias(Neuron & neuron, int layer){

// }

void NeuralNetworkFF::calculate_dLoss_dWeight_and_dLoss_dBias(Neuron &neuron, int layer)
{

    double dL_dB = neuron.get_dLoss_dActivation() * neuron.get_dActivation_dInput();
    neuron.set_dLoss_dBias(dL_dB);
    neuron.add_dLoss_dBias_data_point(dL_dB);

    // Computes the derivate of the loss with respect to each weight
    double dA_dZ = neuron.get_dActivation_dInput();
    std::vector<double> dLoss_dWeights = std::vector<double>(neuron.weights.size(), 0);

    double dLoss_dZ = neuron.get_dLoss_dActivation() * dA_dZ;
    for (int j = 0; j < neuron.weights.size(); ++j)
    {
        double dZ_dW = neurons[layer - 1][j].getActivation();
        double dL_dW = dLoss_dZ * dZ_dW;
        dLoss_dWeights[j] = dL_dW;
        neuron.dLoss_dWeight[j] = dL_dW;
    }

    neuron.add_dLoss_dWeight_data_point(dLoss_dWeights);
}

void NeuralNetworkFF::update_weights(double learning_rate, bool reset)
{

    // Update layer by layer, neuron by neuron within each layer
    for (int layer = 0; layer < neurons.size(); ++layer)
    {
        for (Neuron &neuron : neurons[layer])
        {
            neuron.update_weights_bias(learning_rate, reset);
        }
    }
}

size_t NeuralNetworkFF::get_num_layers()
{
    return neurons.size();
}

template <typename ExamplesIterator, typename ExpectIterator>
void NeuralNetworkFF::train(ExamplesIterator examples_iter, ExamplesIterator examples_end,
                            ExpectIterator expect_iter, ExpectIterator expect_end,
                            TrainConfig *config)
{

    // If a nullptr is passed for the config, then train with the default parameters on the network
    if (!config)
        config = TrainConfig();

    int max_examples = config->num_training_examples;

    // Set the max number of examples. Assumes that max() is more than will ever be attempted by a reasonable human
    if (config->num_training_examples == -1)
        max_examples = std::numeric_limits<int>::max();

    LearningRateFunctionBase *learning_rate_function = config->learning_function;
    ConstantLearningFunction ConstantRateFunction;

    // If the learning rate is nullptr, then set it to be the ConstantLearningRate function
    if (!learning_rate_function)
        learning_rate_function = &ConstantRateFunction;

    // The current number of examples that we have seen
    int example_index = 0;
    int batch_size = config->batch_size;

    while (examples_iter != examples_end && expect_iter != expect_end && example_index < max_examples)
    {

        ++example_index;

        train_on_example(*examples_iter, *expect_iter);
        examples_iter += 1;
        expect_iter += 1;

        // Update the weights and bias' in the neural network
        if (example_index % batch_size == 0)
            update_weights(learning_rate_function->get_learning_rate(), true);

        if (config->verbose && config->verbose_count % example_index == 0)
            std::cout << "Trained on " << example_index << " examples. " << std::endl;
    }
}

template <typename ExamplesIterator, typename ExpectIterator, typename OutputCmp>
NeuralNetworkFF::TestResults NeuralNetworkFF::test(ExamplesIterator examples_iter, ExamplesIterator examples_end,
                                                   ExpectIterator expect_iter, ExpectIterator expect_end, OutputCmp output_cmp,
                                                   TestConfig *config)
{

    if (!config)
        config = TestConfig();

    int max_examples;
    if(config->max_examples == -1)
        max_examples = std::numeric_limits<int>::max(); 

    // Variables needed for the test results
    int num_correct = 0; 
    int num_incorrect = 0;
    
    int example_index = 0; 
    
    // Setup for the training
    std::vector < double > output = std::vector < double > ( neurons.back().size() ) ; 

    while (examples_iter != examples_end && expect_iter != expect_end && example_index < max_examples){
        ++example_index; 
        if( output_cmp(forwardPass(*examples_iter), *expect_iter ))
            ++num_correct;
        else    
            ++num_incorrect;

    }

    // The testing return struct. 
    TestResults results;
    results.correct = num_correct;
    results.incorrect = num_incorrect;
    results.correct_rate = num_correct + num_incorrect; 
    results.correct_rate = (double) num_correct / num_incorrect; 

    return results;
    
}

std::ostream & operator<<(std::ostream & os, const NeuralNetworkFF::TestResults & results){
    os << "Test Results:\n\n"; 
    os << "    Total Correct: " << results.correct << "\n";
    os << "    Total Incorrect: " << results.incorrect << "\n";
    os << "    Total Examples: " << results.num_examples << "\n";
    os << "    Correct Rate: " << results.correct_rate << "\n";
    return os; 
}