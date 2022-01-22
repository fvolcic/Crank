/**
 * @file mnist_nn.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-01-22
 * 
 * @copyright Copyright (c) 2022
 * 
 * @note To compile:                            
 *      g++ examples/MNIST/mnist_nn.cpp src/mnist/* src/ff/* -o bin/mnist_nn_example
 *       To run:
 *      ./bin/mnist_nn_example
 */
#include "../../include/ff/ff.h"
#include "../../include/mnist/mnist.h"
#include <vector>
#include <iostream>

int main(){

    MNIST_DATASET * dataset = read_dataset(); // Read the dataset from memory

    int num_layers = 4;
    std::vector<int> neuron_counts = {784, 20, 15, 10};
    NeuralNetworkFF net(num_layers, neuron_counts); 

    for(int i = 0; i < dataset->test_labels.size(); ++i){

        std::vector<double> input; 
        for(int j = 0; j < dataset->training_images[i].size(); ++j){ 
            input.push_back(dataset->training_images[i][j]);
        }
        std::vector<double> output(10, 0);
        output[dataset->test_labels[i]] = 1; 
        net.train_on_example(input, output); 

        if(i % 10 == 0){
            net.update_weights(0.1); 
        }

        if(i % 100 == 0){
            std::cout << "Trained on " << i << " examples" << std::endl;
            
        }

    }

    std::cout << "Testing Phase!" << std::endl; 

    int examples = 0;
    int correct = 0;

    for(int i = 0; i < 10; ++i){
        std::vector<double> input; 
        for(int j = 0; j < dataset->test_images[i].size(); ++j){ 
            input.push_back(dataset->test_images[i][j]);
        }
        std::vector<double> output; 
        net.forwardPass(input, output); 

        double max = -1;
        int prediction = 0; 
        for(int j = 0; j < 10; ++j){
            if(output[j] > max){
                prediction = j;
                max = output[j];
            }
        }

        std::cout << "Predicted " << prediction << " | Correct " << (int) dataset->test_labels[i] << std::endl; 

        ++examples;

        if(prediction == (int)dataset->test_labels[i]){
            ++correct;
        }

    }

    std::cout << "Correct Rate " << ((double) correct / (double) examples) << std::endl; 

}