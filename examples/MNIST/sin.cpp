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
#include "../../include/ff/activation.h"
#include "../../include/ff/identity.h"

using namespace std; 

int main(){

    int num_layers = 5;
    std::vector<int> neuron_counts = {1, 2, 3, 2, 1};
    std::vector<std::vector<std::vector< double >>> weights = {
                                                             {{}}, // No weights for layer 1
                                                             {{  }}, // Weights going into layer 2
                                                             {{ 0.5 }}}; // Weights going into layer 3
    std::vector< std::vector< double >> bias = { {},
                                                {0.5}, // Bias for neurons in layer 1
                                                {0.5}}; // Bias for neurons in layer 2

    NeuralNetworkFF net(num_layers, neuron_counts); 

    net.neurons.back()[0].setActivationBase(new Identity()); 

    vector< double > input = {0};
    vector< double > output = {0}; 

    int examples = 0;
    double err = 0;

    for(int i = 0; i < 10000; ++i){
        input[0] = ((double) rand() / RAND_MAX) * 2; 
        output[0] = sin(input[0]); 

        auto calced = net.forwardPass(input);

        ++examples; 
        err += abs( calced[0] - output[0] );
    }

    cout << "Tested on " << examples << " examples" << endl;
    cout << "The average err was " << (err / examples) << "\n" << endl; 

    for(int i = 0; i < 10000000; ++i){
        
        input[0] = ((double) rand() / RAND_MAX)* 2;
        output[0] = sin(input[0]) * 2; 

        net.train_on_example(input, output);


            if(i < 1000000 && i % 100 == 0)
            net.update_weights(0.07); 
            else if(i % 100 == 0)
            net.update_weights(0.005);
            //net.update_weights(1);
        
    }

    cout << "Trained!" << endl; 

    examples = 0;
    err = 0;

    for(int i = 0; i < 1000; ++i){
        input[0] = ((double) rand() / RAND_MAX) * 2; 
        output[0] = sin(input[0]) * 2; 

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
        cout << "\n\nThe actual value of sin(" << user_val << ") is " << sin(user_val) * 2 << endl;
        cout << "The Neural Network predicted: " << net.forwardPass(input)[0] << "\n" << endl; 
    }

}