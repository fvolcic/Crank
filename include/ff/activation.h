/**
 * @file activation.h
 * 
 * @brief This contains the base class for the activation functions
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

/**
 * @brief The activation function base class
 * 
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

class ActivationBase{

public:

    ActivationBase(); 
    virtual ~ActivationBase() = 0; 

    /**
     * @brief returns the value of the activation function at a given x
     * 
     * @param x - the x value
     * @return double 
     */
    virtual double operator()(double x) = 0; 

    /**
     * @brief Returns the value of the activation function at a given x
     * 
     * @param x - the x value
     * @return double 
     */
    virtual double compute(double x) = 0; 

    /**
     * @brief Returns the derivate of the activation function at a given x
     * 
     * @param x 
     * @return double 
     */
    virtual double derivative(double x) = 0; 

};

#endif