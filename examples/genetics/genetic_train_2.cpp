/**
 * @file genetic_train_2.cpp
 *
 * @brief
 * @version 0.1
 * @date 2022-03-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#include "../../include/crank.h"
#include "../../include/genetics/genetics.h"

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
        valueA = round(random_range(0, 1));
        valueB = round(random_range(0, 1));
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
        if (valueA != valueB)
            return {1};
        return {0};
    }
};

class Fitness
{
public:
    double operator()(NeuralNetworkFF *network)
    {
        ExampleIterator example;
        ExpectIterator expect;

        double err = 0;

        for (int i = 0; i < 1000; ++i)
        {
            auto output = network->forwardPass(*example);
            auto correct = *expect;

            for (int j = 0; j < output.size(); ++j)
            {
                err += pow(output[j] - correct[j], 2);
            }

            example += 1;
            expect += 1;
        }
    
        return err; 
    }
};


int main(){
    GeneticTrainer<Fitness> trainer(100, 3, {2, 2, 1}); 
    trainer.train(80, true); 
}