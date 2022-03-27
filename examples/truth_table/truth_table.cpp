/**
 * @file trig.cpp
 *
 * @brief This is an example training on an XOR gate (Not linearly seperable)
 * @version 0.1
 * @date 2022-01-28
 *
 * @copyright Copyright (c) 2022
 *
 * @note
 *      to compile:
 *          g++ examples/truth_table/truth_table.cpp -Ofast -o bin/truth_table_example
 *      to run:
 *          ./bin/truth_table_example
 * 
 */

#include "../../include/utils.h"
#include "../../include/ff/ff.h"
#include "../../include/ff/learning_functions.h"
#include <vector>
#include <cmath>
#include <iostream>

// The value which the truth table is taking on
double valueA = 0;
double valueB = 0; 

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
        return {valueA, valueB};
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
        valueA = round( random_range(0, 1) );
        valueB = round( random_range(0, 1) );
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
        if(valueA != valueB)
            return {1}; 
        return {0}; 
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
        return (round(vec1[0]) == vec2[0]);
    }
};

int main()
{

    // Step 1: Create the neural network with random weights and bias'
    int num_layers = 3;
    std::vector<int> neuron_counts = {2, 3, 1};
    NeuralNetworkFF net(num_layers, neuron_counts);

    // Step 2: Create the examples and expectation iterators 
    ExampleIterator examples;
    ExpectIterator expect;

    // Set up testing params for pre-testing
    OutputCmp outputcmp;
    NeuralNetworkFF::TestConfig test_config;
    test_config.max_examples = 10000;

    NeuralNetworkFF::TestResults results = net.test(examples, examples, expect, expect, outputcmp, &test_config);
    std::cout << "Pre-training results" << std::endl; 
    std::cout << results << std::endl;

    // Step 3: Set up the training configuration 
    NeuralNetworkFF::TrainConfig train_config;
    train_config.batch_size = 1; // TODO Seems like when batch size is > 2, training completely fails
    train_config.learning_function = new ConstantLearningFunction(0.1);
    train_config.verbose = false; 
    train_config.verbose_count = 10000;
    train_config.num_training_examples = 60000;

    // Step 4: Train
    net.train(examples, examples, expect, expect, &train_config);

    std::cout << "Training complete!\n" << std::endl;

    // Retest the net after training 
    results = net.test(examples, examples, expect, expect, outputcmp, &test_config);
    std::cout << "Post-training results" << std::endl; 
    std::cout << results << std::endl;

    net.save_to_file("TT_Grad_descent");

    while(false){

        double value1, value2;

        std::cout << "Please input value 1: ";
        std::cin >> value1;
        std::cout << "Please input value 2: ";
        std::cin >> value2; 

        std::vector< double > input = {value1, value2}; 
        std::vector < double > output = net.forwardPass(input); 

        std::cout << "\n Predicted " << output[0] << "\n" << std::endl;

    }

}