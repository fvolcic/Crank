/**
 * @file sigmoid.cpp
 * 
 * @brief The sigmoid activation function
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef SIGMOID_CPP
#define SIGNOID_CPP

#include "../../include/ff/sigmoid.h"
#include <cmath>
#include <string>

double Sigmoid::compute(double x){
    return 1 / (1 + exp(-x)); 
}

double Sigmoid::operator()(double x){
    return compute(x); 
}

double Sigmoid::derivative(double x){
    double s = compute(x);
    return s * (1 - s); // derivate of sigmoid S(X) is S(X) * (1 - S(X)) 
}

std::string Sigmoid::to_external_repr(){
    return "Sigmoid"; 
}

ActivationBase * Sigmoid::clone(){
    return new Sigmoid();
}

Sigmoid::Sigmoid(){}
Sigmoid::~Sigmoid(){}

#endif