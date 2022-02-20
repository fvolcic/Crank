/**
 * @file output_ff.cpp
 * @author your name (you@domain.com)
 * @brief Handles saving and constructing the external representation here
 * @version 0.1
 * @date 2022-02-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#ifndef OUTPUT_FF_CPP
#define OUTPUT_FF_CPP

#include "../../include/ff/ff.h"
#include "../../include/ff/neuron.h"
#include "../../include/ff/activation.h"
#include "../../include/utils.h"
#include <string> 
#include <vector> 
#include <iostream> 
#include <fstream>

using namespace std; 

void NeuralNetworkFF::to_external_repr(std::ostream & os){

    for(int layer = 0; layer < neurons.size(); ++layer){

        // new layer
        os << "def layer\n"; 
        os << "neurons " << neurons[layer].size() << "\n\n";

        if(!layer){
            os << "end layer\n\n";
            continue;
        }

        for(int neuron_index = 0; neuron_index < neurons[layer].size(); ++neuron_index){
            
            os << "neuron " << neuron_index << " bias " << neurons[layer][neuron_index].getBias() << "\n";
            os << "neuron " << neuron_index << " weights "; 

            for(auto weight : neurons[layer][neuron_index].weights){
                os << weight << " "; 
            }
            os << "\n"; 

            if(neurons[layer][neuron_index].getActivationFunction()->to_external_repr() != "Sigmoid"){
                os << "neuron " << neuron_index << " activation " 
                << neurons[layer][neuron_index].getActivationFunction()->to_external_repr()
                << "\n";
               
            }

        }
            os << "\nend layer\n\n";
    }

}

void NeuralNetworkFF::save_to_file(std::string filename){
    ofstream outfile; 
    outfile.open(filename); 

    if(!outfile){
        cerr << "Error: Could not open " << filename << endl; 
        exit(1); 
    }

    // If no error, output network to file
    to_external_repr(outfile);

}

#endif