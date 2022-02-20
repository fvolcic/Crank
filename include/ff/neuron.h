/**
 * @file neuron.h
 *
 * @brief Header for the neuron used in the forward feed neural network
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include "activation.h"
// #include "ff.h"

class NeuralNetworkFF; 

class Neuron
{

friend class NeuralNetworkFF;

public:
    /**
     * @brief Construct a new default Neuron with a sigmoid activation
     * 
     */
    Neuron();

    /**
     * @brief Create a sigmoid neuron with specified weights and a bias
     * 
     * @param bias - The bias for the given neuron
     * @param weights - A vector of weights for computing a forward pass
     */
    Neuron(double bias, std::vector<double> weights);

    /**
     * @brief Construct a new Neuron object with a custom activation function
     * 
     * @param bias - Bias for a given neuron
     * @param weights - Vector of weights based on the connection of current neuron to the previous layers neurons
     * @param activationFunction - Activation function used for a neuron to compute cost
     */
    Neuron(double bias, std::vector<double> weights, ActivationBase *activationFunction);

    /**
     * @brief Copy Constructor for Neuron 
     * 
     * @param n1 - Neuron you want to copy from
     */
    Neuron(const Neuron &n1);

    /**
     * @brief Set the input for a neuron
     * 
     * @param input 
     */
    void setInput(double input);

    
    /**
     * @brief Get the Input object
     * 
     * @return double 
     */
    double getInput();
    
    /**
     * @brief Compute the input of a neuron given the activations of the previous layer of the network
     * 
     * @param previousLayer - vector of activations of the previous layer 
     */
    void computeInput(std::vector<double> previousLayer, int previousLayerSize);

    /**
     * @brief get the output of a neuron after the activation function has been applied
     * 
     */
    double getOutput();

    /**
     * @brief Manually set the output value of a neuron
     * 
     */
    void setOutput(double output);

    /**
     * @brief Set the Bias object
     * 
     * @param bias - Bias that the neuron is to be set to
     */
    void setBias(double bias);

    /**
     * @brief Get the Bias object
     * 
     * @return Neuron Bias 
     */
    double getBias();
    
    /**
     * @brief set the weight of the weight at index weight_index
     * 
     * @param weight_index 
     * @param weight 
     */
    inline void setWeight(int weight_index, double weight){
        weights[weight_index] = weight; 
    } 

    /**
     * @brief Set the Weights object
     * 
     * @param weights - Weight vector that the neuron will have 
     */
    void setWeights(std::vector<double> weights);
    
    /**
     * @brief Get the Weights object
     * 
     * @return Neuron Weights Vector 
     */
    std::vector<double> getWeights();

    /**
     * @brief Set the Activation object
     * 
     * @param activation - Setting the activation of current neuron
     */
    void setActivation(double activation);

    /**
     * @brief Get the Activation object
     * 
     * @return Neuron Activation Value 
     */
    double getActivation();


    /**
     * @brief Set the Activation Base object
     * 
     * @param activationFunc - Set the activation function of a neuron to activationFunc
     */
    void setActivationBase(ActivationBase *activationFunc);

    /**
     * @brief Get the Activation Function object
     * 
     * @return Activation Function of current Neuron
     */
    ActivationBase *getActivationFunction();

    /**
     * @brief Set the dLoss_dActivation value of the neuron
     * 
     * @param value 
     */
    void set_dLoss_dActivation(double value);

    /**
     * @brief Get the dLoss_dActivation value of the neuron
     * 
     */
    double get_dLoss_dActivation();

    /**
     * @brief Set the dActivation_dInput value of the neuron
     * 
     * @param value 
     */
    void set_dActivation_dInput(double value);

    /**
     * @brief Get the dActivation_dInput value of the neuron
     * 
     */
    double get_dActivation_dInput();

    /**
     * @brief Set derivative of the loss function with respect to bias for a neuron 
     * 
     * @param value 
     */
    void set_dLoss_dBias(double value);

    /**
     * @brief Get the dLoss dBias object
     * 
     * @return derivative of the loss function with respect to bias for a neuron 
     */
    double get_dLoss_dBias();


    /**
     * @brief Set the dLoss dWeight object
     * 
     * @param Set vector of the derivatives of the Loss Function with respect to the weight of a neuron 
     */
    void set_dLoss_dWeight(std::vector<double> vec);
    
    /**
     * @brief Get the dLoss dWeight object
     * 
     * @return The vector of the derivatives of the Loss Function with respect to the weight of a neuron
     */
    std::vector<double> get_dLoss_dWeight();

    /**
     * @brief Add a dLoss_dWeight data point to the neuron
     * 
     * @param data_points 
     */
    void add_dLoss_dWeight_data_point(std::vector<double> & data_points); 

    /**
     * @brief Add a dLoss_dBias data point to the neuron
     * 
     * @param data_point 
     */
    void add_dLoss_dBias_data_point(double data_point); 

    /**
     * @brief This resets the average derivate's
     * 
     */
    void reset_partial_averages();

    /**
     * @brief Update the weights and the bias' in the neural network
     * 
     * @param learning_rate 
     * @param reset 
     */
    void update_weights_bias(double learning_rate, bool reset = true);


#ifndef NN_DEBUG
private:
#endif

    ActivationBase *activationBase; // The base class for the activation function

    double input;                   // The value of the input to the neuron    
    double activation;              // The value after the activation function has been applied to this neuron

    double bias; // The bias for the neuron

    std::vector<double> weights; // A vector of all the weights for connections into the neuron

    // These are used for computing the partial derivatives

    double average_dLoss_dBias = 0;             // the average bias partial derivative
    std::vector<double> average_dLoss_dWeight; // The average weights partial derivatives
    int num_examples_dBias = 0;                   // The current number of training examples
    int num_examples_dWeight = 0;           

    // Backpropagation variables
    double dLoss_dActivation = 0;
    double dActivaton_dInput = 0;

    double dLoss_dBias = 0;
    std::vector<double> dLoss_dWeight; 

};

#endif