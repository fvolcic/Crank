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

#include <string> 

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

    /**
     * @brief Get the external representation of the activation function
     * 
     * @return std::string 
     */
    virtual std::string to_external_repr() = 0; 

};


class Linear : public ActivationBase
{

public:

    inline Linear(double slope) : slope(slope){}

    /**
     * @brief returns the value of the activation function at a given x
     * 
     * @param x - the x value
     * @return double 
     */
    inline virtual double operator()(double x){
        return x * slope; 
    }

    /**
     * @brief Returns the value of the activation function at a given x
     * 
     * @param x - the x value
     * @return double 
     */
    inline virtual double compute(double x){
        return x * slope; 
    }

    /**
     * @brief Returns the derivate of the activation function at a given x
     * 
     * @param x 
     * @return double 
     */
    inline virtual double derivative(double x){
        return slope;
    }

    inline virtual std::string to_external_repr(){
        return "Linear";
    }

private: 

    double slope; 

};

#endif