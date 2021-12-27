/**
 * @file ff.cpp
 * 
 * @brief 
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "../../include/ff/ff.h"
#include <stdlib.h>

NeuralNetworkFF::NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts){
    //Initializes first layer of Neural Network
    for(int i = 0; i < neuron_counts[0]; ++i){
        neurons[0][i] = Neuron();
    }

    for(int x = 1 ; x < neuron_counts.size(); ++x){
        for(int y = 0; y < neuron_counts[x]; ++y){
            
            neurons[x][y].bias = 
            // RETURNS RANDOM NUMBER [0, 1]
            static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

        }
    }
}


void NeuralNetworkFF::forwardPass(std::vector<double> & input, std::vector<double> & output){
    
    if(maxLayerSize == -1)
        findMaxLayerSize(); 

    std::vector< double > intermediate_result;
    intermediate_result.resize(maxLayerSize); 

    // Setup all the input values for the neural network
    for(int i = 0; i < neurons[0].size(); ++i){
        neurons[0][i].setOutput(input[i]); 
        intermediate_result[i] = input[i]; 
    }    

    
    // Compute the forward pass for the network
    for(int i = 1; i < neurons.size(); ++i){
        for(int j = 0; i < neurons[i].size(); ++j){
            neurons[i][j].computeInput(intermediate_result); 
        }

        for(int j = 0; j < neurons[i].size(); ++j){
            intermediate_result[j] = neurons[i][j].getOutput(); 
        }
    }

    for(int i = 0; i < neurons.back().size(); ++i){
        output.push_back(intermediate_result[i]); 
    }

}

std::vector<double> NeuralNetworkFF::forwardPass(std::vector<double> & input){
    std::vector<double> output;
    forwardPass(input, output);
    return output; 
}

void NeuralNetworkFF::findMaxLayerSize(){
    
    for(int i = 0; i < neurons.size(); ++i){
        if(maxLayerSize < neurons[i].size())
            maxLayerSize = neurons[i].size(); 
    }

}