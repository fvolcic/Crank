/**
 * @file utils.h
 * @brief Common utility functions
 * @version 0.1
 * @date 2022-01-28
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <random>
#include <sstream>
#include <vector> 
#include <iterator> 

#ifndef UTILS_H 
#define UTILS_H

inline double random_nn()
{
    static std::uniform_real_distribution<> distr(0, 1);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return distr(gen);
}

inline double random_range(double a, double b)
{
    return random_nn() * (b - a) + a;
}

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

#endif 