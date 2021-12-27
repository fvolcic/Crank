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
    Neuron( double bias, std::vector<double> weights, ActivationBase * activationFunction );
    
    /**
     * @brief Copy Constructor for Neuron 
     * 
     * @param n1 - Neuron you want to copy from
     */
    Neuron( const Neuron &n1);
    
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
     * @brief Set the Weights object
     * 
     * @param weights - Weight vector that the neuron will have 
     */
    void setWeights(std::vector<double> weights);

    /**
     * @brief Set the Activation object
     * 
     * @param activation - Setting the activation of current neuron
     */
    void setActivation(double activation);
    /**
     * @brief Set the Activation Base object
     * 
     * @param activationFunc - Set the activation function of a neuron to activationFunc
     */
    void setActivationBase(ActivationBase *activationFunc);

    /**
     * @brief Get the Bias object
     * 
     * @return Neuron Bias 
     */
    double getBias();

    /**
     * @brief Get the Weights object
     * 
     * @return Neuron Weights Vector 
     */
    std::vector<double> getWeights();

    /**
     * @brief Get the Activation object
     * 
     * @return Neuron Activation Value 
     */
    double getActivation();

    /**
     * @brief Get the Activation Function object
     * 
     * @return Activation Function of current Neuron
     */
    ActivationBase * getActivationFunction();


private:
    ActivationBase *activationBase; // The base class for the activation function
    double activation;              // The value after the activation function has been applied to this neuron

    double bias;

    std::vector<double> weights;
};

