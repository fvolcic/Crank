/**
 * @file read_ff.cpp
 * 
 * @brief Contains functionality for reading in a neural network from a stream
 * @version 0.1
 * @date 2022-02-12
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef READ_FF_CPP
#define READ_FF_CPP

#include "../../include/ff/ff.h"
#include "../../include/ff/neuron.h"
#include "../../include/ff/activation.h"
#include "../../include/utils.h"
#include <istream> 
#include <string> 
#include <vector> 
#include <iostream> 

using namespace std; 

vector<Neuron> parse_layer_from_is(istream & is, int prev_layer_size); 

NeuralNetworkFF::NeuralNetworkFF(std::istream & is){

    string line;
    is >> std::ws; // remove unwanted leading whitespace  

    // Run this loop while there is data left in the input stream 
    while( getline(is, line, '\n') && line.length() ){

        vector<string> split_str = split(line, ' '); 
        string command = split_str[0]; 

        // Skip line on comment 
        if(command[0] == '#'){
            continue; 
        }

        // Parse a new layer
        if(command == "def"){
            
            // Parse a layer
            if(split_str.size() > 1 && split_str[1] == "layer"){
                neurons.push_back(parse_layer_from_is(is, neurons.back().size() )); 
            }
            else{
                cerr << "Invalid definition. Try \"def layer\"" << endl;
                exit(1); 
            }
        }

    }

    

}

vector<Neuron> parse_layer_from_is(istream & is, int prev_layer_size){

    string line;
    is >> std::ws; // remove unwanted leading whitespace  

    bool size_set = false; 
    int layer_size = 0; 

    vector<Neuron> neurons; 

    while(getline(is, line, '\n') && line.length()){

        vector<string> split_str = split(line, ' '); 
        
        if(split_str[0] == "def"){
            cerr << "Error: Unclosed layer before new layer definition." << endl; 
            exit(1);
        }
        
        else if(split_str[0] == "neurons"){
            if(size_set){
                cerr << "Size already set" << endl;
                exit(1); 
            }

            size_set = true; 
            layer_size = stoi(split_str[1]); 

            neurons.resize(layer_size); 

        }

        else if(split_str[0] == "neuron"){
            int index = stoi(split_str[1]); 
            
            if(split_str[2] == "bias"){
                neurons[index].setBias(stod(split_str[3])); 
            }

            if(split_str[2] == "weights"){
                if(split_str.size() != 3 + prev_layer_size){
                    cerr << "Invalid number of weights in neuron defintion\nLine: " << endl;
                    cerr << line;
                    exit(1);
                }

                vector<double> weights; 

                for(int i = 0; i < prev_layer_size; ++i){
                    weights.push_back(stod(split_str[3 + i])); 
                }

                neurons[index].setWeights(weights); 

            }

            if(split_str[2] == "weight"){
                int weight_index = stoi(split_str[3]);
                neurons[index].setWeight(weight_index, stod(split_str[4])); 
            }

            if(split_str[2] == "activation"){
                if(split_str[3] == "linear"){
                    neurons[index].setActivationBase(new Linear(stod(split_str[4])));
                }
            }

            

        }

        else if(split_str[0] == "end"){
            if(split_str[1] == "layer"){
                return neurons; 
            }

            cerr << "Error: Invalid close" << endl;
            cerr << "Line: " << line << endl;
            exit(1); 
        }

    }

    cerr << "Error: Invalid file. Unfinished layer definition" << endl;
    exit(1);
    return {}; 

}

#endif