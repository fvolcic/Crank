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

#ifndef NEURON_CPP
#define NEURON_CPP

#include "../../include/ff/neuron.h"
#include "../../include/ff/sigmoid.h"

Neuron::Neuron() : bias(0), weights(), activation(0)
{

    this->activationBase = new Sigmoid(); // TODO : Add neuron deconstructer
}

Neuron::Neuron(double bias, std::vector<double> weights) : bias(bias), weights(weights), activation(0), 
                                                            dLoss_dWeight(weights.size(), 0), average_dLoss_dWeight(weights.size(), 0)
{ 
    this->activationBase = new Sigmoid(); // TODO: Add neurom deconstructor
}

Neuron::Neuron(double bias, std::vector<double> weights, ActivationBase *activationFunction) : bias(bias), weights(weights),
                                                                                               activation(0), activationBase(activationFunction), 
                                                                                               dLoss_dWeight(weights.size(), 0),
                                                                                               average_dLoss_dWeight(weights.size()) {}

Neuron::Neuron(const Neuron &n1)
{
    bias = n1.bias;
    weights = n1.weights;
    activation = n1.activation;

    // TODO: Double check this. I think that the activation base is dynamically allocated, thus we need to deep copy this
    // I think that this works now. Maybe remove the TODO? 
    activationBase = n1.activationBase->clone();
}

void Neuron::setInput(double input)
{
    activation = activationBase->compute(input);
    this->input = input;
}

void Neuron::computeInput(std::vector<double> previousLayer, int previousLayerSize)
{
    double sum = 0;

    for (int i = 0; i < previousLayerSize; ++i)
    {
        sum += previousLayer[i] * weights[i];
    }
    sum += bias;
    input = sum;

    activation = (*activationBase).compute(sum);
}

double Neuron::getOutput()
{
    return activation;
}

void Neuron::setOutput(double output)
{
    activation = output;
}

void Neuron::setBias(double bias)
{
    this->bias = bias;
}

// TODO: Make this take the input by reference rather than by copy
void Neuron::setWeights(std::vector<double> weights)
{
    this->weights = weights;
    this->average_dLoss_dWeight.resize(weights.size()); 
    this->dLoss_dWeight.resize(weights.size()); 

}

void Neuron::setActivation(double activation)
{
    this->activation = activation;
}

void Neuron::setActivationBase(ActivationBase *activationFunc)
{
    activationBase = activationFunc;
}

double Neuron::getBias()
{
    return bias;
}

std::vector<double> Neuron::getWeights()
{
    return weights;
}

double Neuron::getActivation()
{
    return activation;
}

ActivationBase *Neuron::getActivationFunction()
{
    return activationBase;
}

void Neuron::set_dActivation_dInput(double value)
{
    dActivaton_dInput = value;
}

double Neuron::get_dActivation_dInput()
{
    return dActivaton_dInput;
}

void Neuron::set_dLoss_dActivation(double value)
{
    dLoss_dActivation = value;
}

double Neuron::get_dLoss_dActivation()
{
    return dLoss_dActivation;
}

double Neuron::getInput()
{
    return input;
}

void Neuron::set_dLoss_dBias(double val)
{
    dLoss_dBias = val;
}

double Neuron::get_dLoss_dBias()
{
    return dLoss_dBias;
}

void Neuron::set_dLoss_dWeight(std::vector<double> val)
{
    dLoss_dWeight = val;
}

std::vector<double> Neuron::get_dLoss_dWeight()
{
    return dLoss_dWeight;
}

void Neuron::add_dLoss_dBias_data_point(double data_points)
{
    average_dLoss_dBias *= num_examples_dBias;
    ++num_examples_dBias;
    average_dLoss_dBias += data_points;
    average_dLoss_dBias /= num_examples_dBias;
}

void Neuron::add_dLoss_dWeight_data_point(std::vector<double> &data_points)
{
    for (int i = 0; i < average_dLoss_dWeight.size(); ++i)
    {
        average_dLoss_dWeight[i] *= num_examples_dWeight;
        average_dLoss_dWeight[i] += data_points[i];
    }
    ++num_examples_dWeight;
    for (int i = 0; i < average_dLoss_dWeight.size(); ++i)
    {
        average_dLoss_dWeight[i] /= num_examples_dWeight;
    }
}

void Neuron::reset_partial_averages()
{
    for (int i = 0; i < average_dLoss_dWeight.size(); ++i)
        average_dLoss_dWeight[i] = 0;
    average_dLoss_dBias = 0;
    num_examples_dWeight = 0;
    num_examples_dBias = 0;
}

void Neuron::update_weights_bias(double learning_weight, bool reset){
    bias = getBias() - average_dLoss_dBias * learning_weight; 

    if(reset){
        average_dLoss_dBias = 0;
        num_examples_dBias = 0;
    }

    for(int i = 0; i < weights.size(); ++i){
        weights[i] = getWeights()[i] - average_dLoss_dWeight[i] * learning_weight;
        
        if(reset)
            average_dLoss_dWeight[i] = 0;
    }

    if(reset)
        num_examples_dWeight = 0;
    
}


Neuron::~Neuron(){
    //std::cout << "here | deleting 0x" << *(long long*)this->activationBase << std::endl; 
    // TODO : There is a bug here and I cant delete the activation base.. weird. 
    //delete this->activationBase; 
}

#endif