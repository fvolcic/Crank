/**
 * @file file_read.cpp
 * 
 * @brief This program showcases how one would go about reading in a net from a net file
 * @version 0.1
 * @date 2022-02-13
 * 
 * @copyright Copyright (c) 2022
 * 
 * @note 
 *      to compile:
 *          g++ examples/file_io/file_read.cpp -o bin/file_read
 * 
 */

#include "../../include/ff/ff.h"
#include <vector> 
#include <iostream> 


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

int main(){
    NeuralNetworkFF net("truth_table.net"); 

    ExampleIterator examples;
    ExpectIterator expect;

    OutputCmp outputcmp;
    NeuralNetworkFF::TestConfig test_config;
    test_config.max_examples = 10000;

    NeuralNetworkFF::TestResults results = net.test(examples, examples, expect, expect, outputcmp, &test_config);

    std::cout << results << std::endl;

}
