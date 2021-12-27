/**
 * @file ff.h
 * 
 * @brief 
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "neuron.h"
#include "activation.h"
#include "sigmoid.h"
#include <vector>
#include <algorithm>
#include 
/**
 * @brief A class that represents the possible activation functions one can use
 * 
 */
enum class ActivationFunctions
{
     Sigmoid
};

class NeuralNetworkFF
{
public:
     /**
         * @brief Create a new neural network with default randomized weights and bias'
         *       : The default activation function is the sigmoid
         * @param num_layers - The number of layers in the neural network
         * @param neuron_counts - The number of neurons in each layer of the neural network 
         */
     NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts);

     /**
     * @brief Construct the neural network using predefined weights and bias'
     * 
     * @param num_layers - Number of layers in the neural network
     * @param neuron_counts - The number of neurons in each layer of the neural network
     * @param weights - Weights of each neuron
     * @param bias - A bias term for each neuron in the neural network
     */
     NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts, std::vector<std::vector<std::vector<double>>> &weights, std::vector<std::vector<double>> &bias);

     /**
         * @brief Construct a new Neural Network object
         * 
         * @param network - Neural Network that you want copied 
         */
     NeuralNetworkFF(const NeuralNetworkFF &network);

     /**
         * @brief Destroy the Neural Network object
         * 
         */
     ~NeuralNetworkFF();

     /**
         * @brief Compute a forward pass of a neural network given the values of the input layer
         *      - This method copies the output to a output vector
         * 
         * @param input - Input layer of Neural Network
         * @param output - Last Layer of Neural Network 
         */
     void forwardPass(std::vector<double> &input, std::vector<double> &output);

     /**
         * @brief Compute a forward pass of a neural network given the values of the input layer
         *       -
         * @param input Input layer of the Neural Network
         * @return std::vector<int> - Return a copy of the output vector
         */
     std::vector<double> forwardPass(std::vector<double> &input);

     // TODO Add Move Constructor
     // TODO Implement move semantics for the network

     /**
         * @brief Block the default construction of a Neural Network
         * 
         */
     NeuralNetworkFF() = delete; // No default constructed neural network

private:

     /**
      * @brief This goes through all the layers and sets the maxLayer variable
      * 
      */
     void findMaxLayerSize(); 

     std::vector<std::vector<Neuron>> neurons; // Where all the neurons are stored internally in the network

     int maxLayerSize = -1; // The layer in the network with the most neurons
};