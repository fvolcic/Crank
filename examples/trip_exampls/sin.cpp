/**
 * @file sin.cpp
 * 
 * @brief This example program trains a neural network on the trigonometric sin function
 * @version 0.1
 * @date 2022-01-23
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include "../../include/ff/ff.h"

using namespace std; 

int main(){

    int num_layers = 4;
    std::vector<int> neuron_counts = {1, 5, 5, 1};
    std::vector<std::vector<std::vector< double >>> weights = {
                                                             {{}}, // No weights for layer 1
                                                             {{  }}, // Weights going into layer 2
                                                             {{ 0.5 }}}; // Weights going into layer 3
    std::vector< std::vector< double >> bias = { {},
                                                {0.5}, // Bias for neurons in layer 1
                                                {0.5}}; // Bias for neurons in layer 2

    NeuralNetworkFF net(num_layers, neuron_counts); 

    vector< double > input = {0};
    vector< double > output = {0}; 

    int examples = 0;
    double err = 0;

    for(int i = 0; i < 10000; ++i){
        input[0] = ((double) rand() / RAND_MAX); 
        output[0] = sin(input[0]) * cos(input[0]); 

        auto calced = net.forwardPass(input);

        ++examples; 
        err += abs( calced[0] - output[0] );
    }

    cout << "Tested on " << examples << " examples" << endl;
    cout << "The average err was " << (err / examples) << "\n" << endl; 

    for(int i = 1; i < 100000; ++i){
        if(i % 100000 == 0){
            cout << "Trained on " << i << " examples" << endl; 
        }
        input[0] = ((double) rand() / RAND_MAX);
        output[0] = sin(input[0]) * cos(input[0]); 

        net.train_on_example(input, output);

        if(i < 100){
            net.update_weights(1); 
        }if(i < 400){
            net.update_weights(0.5); 
        }
        else if (i % 4 == 0){
            net.update_weights(0.2); 
        }
    }

    cout << "Trained!" << endl; 

    examples = 0;
    err = 0;

    for(int i = 0; i < 10000; ++i){
        input[0] = ((double) rand() / RAND_MAX); 
        output[0] = sin(input[0]) * cos(input[0]); 

        auto calced = net.forwardPass(input);

        ++examples; 
        err += abs( calced[0] - output[0] );
    }

    cout << "\nTested on " << examples << " examples" << endl;
    cout << "The average err was " << (err / examples) << "\n" << endl; 

    double user_val;

    while(1){
        cout << "Please enter a value: ";
        cin >> user_val;
        input[0] = user_val; 
        cout << "\n\nThe actual value of sin * cos (" << user_val << ") is " << sin(user_val) * cos(user_val) << endl;
        cout << "The Neural Network predicted: " << net.forwardPass(input)[0] << "\n" << endl; 
    }

}