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

#include <vector>
#include "activation.h"

class Neuron
{

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
     * @brief Compute the input of a neuron given the activations of the previous layer of the network
     * 
     * @param previousLayer - vector of activations of the previous layer 
     */
    void computeInput(std::vector<double> previousLayer);

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


private:
    ActivationBase *activationBase; // The base class for the activation function
    double activation;              // The value after the activation function has been applied to this neuron

    double bias; // The bias for the neuron

    std::vector<double> weights; // A vector of all the weights for connections into the neuron

    // These are used for computing the partial derivatives

    double average_bias_dL = 0;             // the average bias partial derivative
    std::vector<double> average_weights_dL; // The average weights partial derivatives
    int num_examples = 0;                   // The current number of training examples

    // Backpropagation variables
    double dLoss_dActivation = 0;
    double dActivaton_dInput = 0;
};
