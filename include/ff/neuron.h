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
    

private:
    
    ActivationBase * activationBase; // The base class for the activation function
    double activation; // The value after the activation function has been applied to this neuron

    double bias;

    std::vector<double> weights;
};