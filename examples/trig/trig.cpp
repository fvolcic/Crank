/**
 * @file trig.cpp
 * 
 * @brief This is a detailed example on how to train and test the neural network using the training and teting func
 * @version 0.1
 * @date 2022-01-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "../../include/utils.h"
#include "../../include/ff/ff.h"
#include <vector>
#include <cmath>
#include <iostream>

double value = 0;

class ExampleIterator
{

public:
    /**
     * @brief Iterator operation to update the iterator
     */
    void operator+=(int val)
    {
    }

    /**
     * @brief add one to the iterator
     * 
     */
    void operator++()
    {
    }

    /**
     * @brief Dereference the iterator. Must return std::vector < double > equal in size to
     *        input layer of the neural network
     * 
     * @return std::vector< double > 
     */
    std::vector<double> operator*()
    {
        return {value};
    }
};

class ExpectIterator
{

public:
    /**
     * @brief Iterator operation to update the iterator
     */
    void operator+=(int val)
    {
        value = random_range(0, 1);
    }

    /**
     * @brief add one to the iterator
     * 
     */
    void operator++()
    {
        operator+=(1);
    }

    /**
     * @brief Dereference the iterator. Must return std::vector < double > equal in size to
     *        input layer of the neural network
     * 
     * @return std::vector< double > 
     */
    std::vector<double> operator*()
    {
        return {cos(value)};
    }
};

class OutputCmp
{

public:
    /**
     * @brief Must have the vectors as cons
     */
    bool operator()(const std::vector<double> &vec1, const std::vector<double> &vec2)
    {
        return abs(vec2[0] - vec1[0]) < 0.01;
    }
};

int main()
{

    int num_layers = 3;
    std::vector<int> neuron_counts = {1, 3, 1};

    NeuralNetworkFF net(num_layers, neuron_counts);

    ExampleIterator examples;
    ExpectIterator expect;

    NeuralNetworkFF::TrainConfig train_config;

    net.train(examples, examples, expect, expect, &train_config);

    OutputCmp outputcmp;

    NeuralNetworkFF::TestConfig test_config;

    NeuralNetworkFF::TestResults results = net.test(examples, examples, expect, expect, outputcmp, &test_config);

    std::cout << results << std::endl;
}