/**
 * @file neuron.cpp
 * 
 * @brief 
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "../../include/ff/neuron.h"
#include "../../include/ff/sigmoid.h"

Neuron::Neuron():bias(1), weights(), activation(0){
    
    this->activationBase = new Sigmoid(); 
}

Neuron:: Neuron(double bias, std::vector<double> weights): bias