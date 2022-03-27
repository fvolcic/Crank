/**
 * @file genetic_train.cpp
 *
 * @brief This file contains a sample implementation of a genetic training algorithm on the neural network.
 *        Here, a genetic training algorithm is used to find a solution to the xor truth table.      
 * 
 * @version 0.1
 * @date 2022-03-26
 *
 * @copyright Copyright (c) 2022
 *
 * 
 *  To compile:
 *      g++ -Iinclude examples/genetics/genetic_train.cpp -Ofast -o bin/truth_table_genetic -D NN_DEBUG
 *  To run:
 *      ./bin/truth_table_genetic
 * 
 */

#include "../../include/crank.h"
#include <utility>
#include <vector>
#include <queue>
#include <vector>
#include <cmath>
#include <iostream>
#include <sstream>

struct gen_offspring_config
{

    gen_offspring_config(int rate, double size) : mutation_rate(rate), mutation_size(size) {}

    int mutation_rate = 8;
    double mutation_size = 0.2;
};

/**
 * @brief take two neural networks, and return a new neural network that is the offspring of the parent
 *
 * @param parent1
 * @param parent2
 * @return NeuralNetworkFF*
 */
NeuralNetworkFF *generate_offspring(NeuralNetworkFF *parent1, NeuralNetworkFF *parent2, const gen_offspring_config *config)
{
    std::stringstream ss; 
    parent1->to_external_repr(ss); 
    NeuralNetworkFF * child_net = new NeuralNetworkFF(ss); 

    for(int layer = 0; layer < child_net->neurons.size(); ++layer){
        for(int neuron_index = 0; neuron_index < child_net->neurons[layer].size(); ++neuron_index){
            auto & neuron = child_net->neurons[layer][neuron_index];

            auto & neuron_other = parent2->neurons[layer][neuron_index];

            for(int weight_index = 0; weight_index < neuron.weights.size(); ++weight_index){
                neuron.weights[weight_index] = (neuron.weights[weight_index] + neuron_other.weights[weight_index])/2;

                if(random_range(0, 10) > config->mutation_rate){
                    neuron.weights[weight_index] += random_range(-config->mutation_size, config->mutation_size); 
                }
            }

            neuron.bias = (neuron.bias + neuron_other.bias) / 2; 
            if(random_range(0, 10) > config->mutation_rate){
                    neuron.bias += random_range(-config->mutation_size, config->mutation_size); 
            }

        }
    }

    return child_net;

}


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


/**
 * @brief Return a fitness score for a given neural network
 *
 * @param network
 * @return double
 */
double measure_fitness(NeuralNetworkFF *network)
{

    ExampleIterator example; 
    ExpectIterator expect; 

    double err = 0; 

    for(int i = 0; i < 1000; ++i){
        auto output = network->forwardPass(*example); 
        auto correct = *expect; 
    
        for(int j = 0; j < output.size(); ++j){
            err += pow( output[j] - correct[j], 2 );
        }

    example+=1;
    expect+=1;

    }

    //std::cout << "Fitness of network (0x" << (*((int*)network)) << ") is " << 1/err << std::endl; 
    return 1/(1+err); 

}


class NetworkCompare
{

public:
    bool operator()(std::pair<double, NeuralNetworkFF *> first, std::pair<double, NeuralNetworkFF *> second)
    {
        return first.first < second.first;
    }
};

int main()
{

    int num_organisms = 100;
    int k_count = 10;
    int num_generations = 50;

    int num_layers = 3;
    std::vector<int> neuron_counts = {2, 3, 1};

    std::vector<NeuralNetworkFF *> organisms;
    std::priority_queue<std::pair<double, NeuralNetworkFF *>, std::vector<std::pair<double, NeuralNetworkFF *>>, NetworkCompare> networks_by_fitness;

    // Initially load the organisms
    for (int i = 0; i < num_organisms; ++i)
    {
        organisms.push_back(new NeuralNetworkFF(num_layers, neuron_counts));
    }

    // Determine the fitness of the networks

    for (int generation = 0; generation < num_generations; ++generation)
    {
        networks_by_fitness = std::priority_queue<std::pair<double, NeuralNetworkFF *>, std::vector<std::pair<double, NeuralNetworkFF *>>, NetworkCompare>(); 

        for (auto organism : organisms)
        {
            networks_by_fitness.push({measure_fitness(organism), organism});
        }

        std::cout << "Generation " << generation << " best Fitness: " << networks_by_fitness.top().first << std::endl;

        //if(generation % 1 == 0){

        //string save_name = "genetics/network_save_generation_" + std::to_string(generation);
        //networks_by_fitness.top().second->save_to_file(save_name); 

        //}

        std::vector<NeuralNetworkFF *> top_organisms;

        for (int i = 0; i < k_count; ++i)
        {
            top_organisms.push_back(networks_by_fitness.top().second);
            networks_by_fitness.pop();
        }

        while(!networks_by_fitness.empty()){
            delete networks_by_fitness.top().second;
            networks_by_fitness.pop(); 
        }

        organisms.clear();
        for (int i = 0; i < num_organisms; ++i)
        {
            int index_a = floor((double)i / top_organisms.size());

            if(index_a >= num_organisms){
                index_a = (int) random_range(0, num_organisms - 1); 
            }

            int index_b = i % top_organisms.size();
            
            gen_offspring_config config(0, 100);
            organisms.push_back(generate_offspring(top_organisms[index_a], top_organisms[index_b], &config));
        }

        for(int i = 0; i < num_organisms - organisms.size(); ++i){
            organisms.push_back(new NeuralNetworkFF(num_layers, neuron_counts));
        }
    }
}