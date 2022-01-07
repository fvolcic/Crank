
#include "../../include/ff/ff.h"
#include "../../include/mnist/mnist.h"
#include <vector>

int main(){

    MNIST_DATASET * dataset = read_dataset(); // Read the dataset from memory

    int num_layers = 4;
    std::vector<int> neuron_counts = {784, 15, 13, 10};

    
}