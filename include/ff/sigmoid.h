/**
 * @file sigmoid.h
 * 
 * @brief 
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef SIGMOID_H
#define SIGMOID_H

#include "activation.h"

class Sigmoid : public ActivationBase
{

public:

    Sigmoid();
    ~Sigmoid(); 

    /**
     * @brief returns the value of the activation function at a given x
     * 
     * @param x - the x value
     * @return double 
     */
    virtual double operator()(double x); 

    /**
     * @brief Returns the value of the activation function at a given x
     * 
     * @param x - the x value
     * @return double 
     */
    virtual double compute(double x); 

    /**
     * @brief Returns the derivate of the activation function at a given x
     * 
     * @param x 
     * @return double 
     */
    virtual double derivative(double x);

    /**
     * @brief the external repr of the Sigmoid
     * 
     * @return std::string 
     */
    virtual std::string to_external_repr();

    virtual ActivationBase * clone(); 

private:
};

#endif