/**
 * @file genetics.h
 *
 * @brief This file contains functions for training a neural network using a genetic training approach.
 * @version 0.1
 * @date 2022-03-27
 *
 * @copyright Copyright (c) 2022
 *
 */

#include <vector>
#include <iostream>
#include <queue>
#include "../ff/ff.h"
#include "../utils.h"

#ifndef GENETIC_H
#define GENETIC_H

// Fitness measurement function
// Offspring generation function

struct OffsringGenConfig
{
    int mutation_rate = 0;
    double mutation_size = 10;
};

class OffspringGenerator
{
public:

    OffspringGenerator()  {

    }

    inline NeuralNetworkFF *operator()(NeuralNetworkFF *parent1, NeuralNetworkFF *parent2, const OffsringGenConfig *config = &OffspringGenerator::default_config)
    {
        std::stringstream ss;
        parent1->to_external_repr(ss);
        NeuralNetworkFF *child_net = new NeuralNetworkFF(ss);

        for (int layer = 0; layer < child_net->neurons.size(); ++layer)
        {
            for (int neuron_index = 0; neuron_index < child_net->neurons[layer].size(); ++neuron_index)
            {
                auto &neuron = child_net->neurons[layer][neuron_index];

                auto &neuron_other = parent2->neurons[layer][neuron_index];

                for (int weight_index = 0; weight_index < neuron.weights.size(); ++weight_index)
                {
                    neuron.weights[weight_index] = (neuron.weights[weight_index] + neuron_other.weights[weight_index]) / 2;

                    if (random_range(0, 10) > config->mutation_rate)
                    {
                        neuron.weights[weight_index] += random_range(-config->mutation_size, config->mutation_size);
                    }
                }

                neuron.bias = (neuron.bias + neuron_other.bias) / 2;
                if (random_range(0, 10) > config->mutation_rate)
                {
                    neuron.bias += random_range(-config->mutation_size, config->mutation_size);
                }
            }
        }

        return child_net;
    }

    static OffsringGenConfig default_config;
};

OffsringGenConfig OffspringGenerator::default_config = OffsringGenConfig();

class NetworkCompare
{

public:
    inline bool operator()(std::pair<double, NeuralNetworkFF *> first, std::pair<double, NeuralNetworkFF *> second)
    {
        return first.first > second.first;
    }
};

template <typename FitnessFunction, typename OffspringGen = OffspringGenerator>
class GeneticTrainer
{

public:
    inline GeneticTrainer(int num_organisms, int num_layers, const std::vector<int> & counts)
    {

        for(int i = 0; i < num_organisms; ++i){
            organisms.push_back(new NeuralNetworkFF(num_layers, counts)); 
        }

    }

    inline std::pair<double, NeuralNetworkFF *> iterate_generation(int num_organisms)
    {

        std::priority_queue<std::pair<double, NeuralNetworkFF *>,
                            std::vector<std::pair<double, NeuralNetworkFF *>>,
                            NetworkCompare>
            networks_by_fitness;

        double top_fitness;

        for (auto organism : organisms)
        {

            networks_by_fitness.push({measure_fitness(organism),
                                      organism});
        }

        top_fitness = networks_by_fitness.top().first;

        std::vector<NeuralNetworkFF *> top_organisms;

        for (int i = 0; i < select_from_top_k; ++i)
        {
            top_organisms.push_back(networks_by_fitness.top().second);
            networks_by_fitness.pop();
        }

        while (!networks_by_fitness.empty())
        {
            delete networks_by_fitness.top().second;
            networks_by_fitness.pop();
        }

        organisms.clear();
        for (int i = 0; i < num_organisms; ++i)
        {
            auto parents = select_parents(top_organisms);
            organisms.push_back(generate_offspring(parents.first, parents.second));
        }

        for (int i = 1; i < top_organisms.size(); ++i)
        {
            delete top_organisms[i];
        }

        return {top_fitness, top_organisms[0]};
    }

    inline NeuralNetworkFF *train(int generations, bool verbose = false)
    {

        std::pair<double, NeuralNetworkFF *> generation_result; // = iterate_generation(100);

        for (int i = 0; i < generations; ++i)
        {
            generation_result = iterate_generation(100);
            if (i < generations - 1)
            {
                delete generation_result.second;
            }

            if (verbose)
            {
                std::cout << "Generation: " << i << " | Top fitness: " << generation_result.first << "\n";
            }
        }

        return generation_result.second;
    }

    inline std::pair<NeuralNetworkFF *, NeuralNetworkFF *> select_parents(std::vector<NeuralNetworkFF *> &parent_pool)
    {

        return {
            parent_pool[(int)random_range(0, parent_pool.size() - 1)],
            parent_pool[(int)random_range(0, parent_pool.size() - 1)]};
    }

    FitnessFunction measure_fitness;
    OffspringGen generate_offspring;

    std::vector<NeuralNetworkFF *> organisms;
    std::priority_queue<std::pair<double, NeuralNetworkFF *>> networks_by_fitness;

    int current_generation = 0;
    int organisms_per_generation = 100;
    int select_from_top_k = 20;
};

#endif