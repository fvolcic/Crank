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

Neuron::Neuron() : bias(0), weights(), activation(0)
{

    this->activationBase = new Sigmoid();
}

Neuron::Neuron(double bias, std::vector<double> weights) : bias(bias), weights(weights), activation(0)
{
    this->activationBase = new Sigmoid();
}

Neuron::Neuron(double bias, std::vector<double> weights, ActivationBase *activationFunction) : bias(bias), weights(weights),
                                                                                               activation(0), activationBase(activationFunction) {}

Neuron::Neuron(const Neuron &n1)
{
    bias = n1.bias;
    weights = n1.weights;
    activation = n1.activation;
    activationBase = n1.activationBase;
}

void Neuron::setInput(double input)
{
    activation = (*activationBase).compute(input);
}

void Neuron::computeInput(std::vector<double> previousLayer)
{
    double sum = 0;
    for (int i = 0; i < previousLayer.size(); ++i)
    {
        sum += previousLayer[i] * weights[i];
    }
    sum += bias; 
    activation = (*activationBase).compute(sum);
}

double Neuron::getOutput()
{
    return activation;
}

void Neuron::setOutput(double output){
    activation = output; 
}

void Neuron::setBias(double bias){
    this->bias = bias;
}

void Neuron::setWeights(std::vector<double> weights){
    this->weights = weights;
}

void Neuron::setActivation(double activation){
    this->activation = activation;
}

void Neuron::setActivationBase(ActivationBase *activationFunc){
    activationBase = activationFunc;
}

double Neuron::getBias(){
    return bias;
}

std::vector<double> Neuron::getWeights(){
    return weights;
}

double Neuron::getActivation(){
    return activation;
}

ActivationBase * Neuron::getActivationFunction(){
    return activationBase;
}