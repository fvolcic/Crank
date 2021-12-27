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
#include <random>

double random_nn()
{
    static std::uniform_real_distribution<> distr(0, 1);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return distr(gen);
}

double random_range(double a, double b)
{
    return random_nn() * (b - a) + a;
}

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

NeuralNetworkFF::NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts, std::vector<std::vector<std::vector<double>>> &weights, std::vector<std::vector<double>> &bias)
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
        neurons.resize(neuron_counts[x]);
        for (int y = 0; y < neuron_counts[x]; ++y)
        {
            neurons[x][y].setBias(bias[x][y]);
            neurons[x][y].setWeights(weights[x][y]);
        }
    }
}

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
        for (int j = 0; i < neurons[i].size(); ++j)
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

    for (int i = 0; i < neurons.size(); ++i)
    {
        if (maxLayerSize < neurons[i].size())
            maxLayerSize = neurons[i].size();
    }
}