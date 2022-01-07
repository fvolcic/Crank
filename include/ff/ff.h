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

#pragma once

#include "neuron.h"
#include "activation.h"
#include "sigmoid.h"
#include <vector>
#include <algorithm>
#include <iterator>

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
     NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts, const std::vector<std::vector<std::vector<double>>> &weights, const std::vector<std::vector<double>> &bias);

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

     // Training Functions defined below

     /**
     * @brief This takes an input example and the expected output,
     *        and then comptues the derivative of weights and bias'.
     *        After that, the partial derivative is added to the average partial derivative
     *        for each weight and bias.
     * 
     * @param input 
     * @param expected_output 
     */
     void train_on_example(std::vector<double> &input, std::vector<double> &expected_output);

     /**
     * @brief Update the weights and bias' based on the gradients computed in backprop
     * 
     * @param learning_rate - the learning rate to use to update the weights 
     */
     void update_weights(double learning_rate, bool reset = true);

     // TODO Add Move Constructor
     // TODO Implement move semantics for the network

     /**
      * @brief Block the default construction of a Neural Network
      * 
      */
     NeuralNetworkFF() = delete; // No default constructed neural network



protected:
     /**
      * @brief This goes through all the layers and sets the maxLayer variable
      * 
      */
     void findMaxLayerSize();

private:

     /**
      * @brief Run backprop on the network starting at layer layer
      * 
      * @param layer - the starting layer for back prop
      */
     void back_propagation( int layer );

     /**
      * @brief Compute the loss with respect to the activation for a given neuron
      * 
      * @param neuron - the neuron to compute the function on
      * @param layer - the layer which the neuron resides
      */
     void calculate_dLoss_dActivation(Neuron & neuron, int layer, int index);

     /**
      * @brief 
      * 
      * @param neuron - neuron to compute the function on
      * @param layer - the layer which the neuron resides
      */
     void calculate_dActivation_dInput(Neuron & neuron, int layer); 
     
     /**
      * @brief 
      * 
      * @param neuron - neuron to compute the function on
      * @param layer - the layer which the neuron resides 
      */
     void calculate_dLoss_dWeight_and_dLoss_dBias(Neuron & neuron, int layer);

     

     std::vector<std::vector<Neuron>> neurons; // Where all the neurons are stored internally in the network

     int maxLayerSize = -1; // The layer in the network with the most neurons
};