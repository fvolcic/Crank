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
 *          g++ examples/MNIST/mnist.cpp -Ofast -o bin/mnist_example
 *
 *      to run:
 *          ./bin/mnist_example
 */

#include "../../include/crank.h"
#include "../../include/mnist/mnist.h"
#include <vector>
#include <cmath>
#include <iostream>

static int example_index = 0;
static MNIST_DATASET *dataset;

std::vector<std::vector<uint8_t>> *images;
std::vector<uint8_t> *labels;

/**
 * @brief This is the ExampleIterator. It is used to provide examples for training the net
 *
 */
class ExampleIterator
{

public:
    ExampleIterator(bool update=false) : example(std::vector<double>(784)), update(update) {}

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
    std::vector<double> &operator*()
    {
        for (int i = 0; i < 784; ++i)
        {
            example[i] = (*images)[example_index][i];
        }

        return example;
    }

    std::vector<double> example;
    bool update; 
};

std::vector<double> expect;

/**
 * @brief This iterator is used in the testing and training. It provides the "expected" output
 *
 */
class ExpectIterator
{

public:
    ExpectIterator(bool update=false) : expect(std::vector<double>(10)), update(update) {}

    /**
     * @brief Iterator operation to update the iterator
     */
    void operator+=(int val)
    {   
        if(update)
            example_index += val;
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
    std::vector<double> &operator*()
    {
        for (auto &val : expect)
            val = 0;
        expect[(*labels)[example_index]] = 1;
        return expect;
    }

    std::vector<double> expect;
    bool update; 
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
        int max_index = 0;
        double max = vec1[0];
        for (int i = 0; i < vec1.size(); ++i)
        {
            if (vec1[i] > max)
            {
                max = vec1[i];
                max_index = i;
            }
        }

        return vec2[max_index];
    }
};

int main()
{
    dataset = read_dataset(); // Read MNIST from memory
    
    // Step 1: Create the neural network with random weights and bias'
    int num_layers = 3;
    std::vector<int> neuron_counts = {784, 100, 10};
    NeuralNetworkFF net(num_layers, neuron_counts);

    // Step 2: Create the examples and expectation iterators
    ExampleIterator examples;
    ExpectIterator expect(true);
    ExampleIterator examples_end(false); 
    ExpectIterator expect_end(false);
    OutputCmp outputcmp;

    NeuralNetworkFF::TestConfig test_config;
    test_config.max_examples = 10000;
    example_index = 0;
    images = &dataset->test_images; 
    labels = &dataset->test_labels; 
    NeuralNetworkFF::TestResults results = net.test(examples, examples_end, expect, expect_end, outputcmp, &test_config);
    std::cout << "Pretraining Results" << std::endl; 
    std::cout << results << std::endl;

    // Step 3: Set up the training configuration
    NeuralNetworkFF::TrainConfig train_config;
    train_config.batch_size = 60;
    train_config.learning_function = new ConstantLearningFunction(0.35);
    train_config.verbose = true;
    train_config.verbose_count = 5000;
    train_config.num_training_examples = 60000;

    // Step 4: Train
    images = &dataset->training_images;
    labels = &dataset->training_labels; 
    example_index = 0;  
    net.train(examples, examples_end, expect, expect_end, &train_config);

    std::cout << "Training complete!\n"
              << std::endl;

    // Step 5: Setup the testing configuration
    

    // Step 6: Test and print the test results
    example_index = 0;
    images = &dataset->test_images; 
    labels = &dataset->test_labels; 
    results = net.test(examples, examples, expect, expect, outputcmp, &test_config);
    std::cout << "Post Training Results" << std::endl; 
    std::cout << results << std::endl;

    delete train_config.learning_function;
}