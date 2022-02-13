/**
 * @file trig.cpp
 *
 * @brief This is a detailed example on how to train and test the neural network using the training and teting func
 * @version 0.1
 * @date 2022-01-28
 *
 * @copyright Copyright (c) 2022
 *
 * @note
 *      to compile:
 *          g++ examples/trig/trig.cpp -Ofast -o bin/trig_example
 *      
 *      to run:
 *          ./bin/trig_example    
 */

#include "../../include/utils.h"
#include "../../include/ff/ff.h"
#include "../../include/ff/learning_functions.h"
#include <vector>
#include <cmath>
#include <iostream>

double value = 0;

/**
 * @brief This is the ExampleIterator. It is used to provide examples for training the net 
 * 
 */
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

    bool operator!=(const ExampleIterator &rhs)
    {
        return true;
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

/**
 * @brief This iterator is used in the testing and training. It provides the "expected" output
 * 
 */
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
     * @brief Since we are training on a set number of examples,
     *        we dont need to worry about reaching the "end" 
     * 
     */
    bool operator!=(const ExpectIterator &rhs)
    {
        return true;
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

/**
 * @brief This class is used for the testing phase. It will take the expected output and
 *        the actual output, and simply tell us if the actual output should be considered "correct"
 * 
 */
class OutputCmp
{

public:
    /**
     * @brief Must have the vectors as cons
     */
    bool operator()(const std::vector<double> &vec1, const std::vector<double> &vec2)
    {
        return std::abs((double) vec2[0] - vec1[0]) < 0.01;
    }
};

int main()
{

    // Step 1: Create the neural network with random weights and bias'
    int num_layers = 3;
    std::vector<int> neuron_counts = {1, 3, 1};
    NeuralNetworkFF net(num_layers, neuron_counts);

    // Step 2: Create the examples and expectation iterators 
    ExampleIterator examples;
    ExpectIterator expect;

    // Step 3: Set up the training configuration 
    NeuralNetworkFF::TrainConfig train_config;
    train_config.batch_size = 1; 
    train_config.learning_function = new ConstantLearningFunction(0.1);
    train_config.verbose = false; 
    train_config.verbose_count = 1000;
    train_config.num_training_examples = 100000;

    // Step 4: Train
    net.train(examples, examples, expect, expect, &train_config);

    std::cout << "Training complete!\n" << std::endl;

    // Step 5: Setup the testing configuration
    OutputCmp outputcmp;
    NeuralNetworkFF::TestConfig test_config;
    test_config.max_examples = 1000;

    // Step 6: Test and print the test results
    NeuralNetworkFF::TestResults results = net.test(examples, examples, expect, expect, outputcmp, &test_config);
    std::cout << results << std::endl;

    // Step 7: Play around and see how well it was trained based on your training parameters
    while(1){
        std::cout << "Please enter a value in range [0, 1]! or -1 to quit: ";
        double value;
        std::cin >> value; 
        if(value == -1){
            break; 
        }

        std::vector<double> input = {value}; 
        std::vector<double> output = net.forwardPass(input); 

        std::cout << "Prediction: " << output[0] << " | Actual: " << cos(value) << "\n" << std::endl;

    }

    delete train_config.learning_function;

}