/**
 * @file identity.cpp
 * 
 * @brief 
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#ifndef IDENTITY_CPP
#define IDENTITY_CPP

#include "../../include/ff/identity.h"

double Identity::compute(double x){
    return 3 * x; 
}

double Identity::derivative(double x){
    return 3; // Derivative of identity function is 1 
}

double Identity::operator()(double x){
    return compute(x); 
}

//Identity::Identity(){}
//Identity::~Identity(){}

#endif