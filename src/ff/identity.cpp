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

#include "../../include/ff/identity.h"

double Identity::compute(double x){
    return x; 
}

double Identity::derivative(double x){
    return 1; // Derivative of identity function is 1 
}