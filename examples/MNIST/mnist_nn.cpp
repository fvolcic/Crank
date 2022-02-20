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
 *      g++ examples/MNIST/mnist_nn.cpp -o bin/mnist_nn_example
 *       To run:
 *      ./bin/mnist_nn_example
 */
#include "../../include/ff/ff.h"
#include "../../include/mnist/mnist.h"
#include <vector>
#include <iostream>
#include <random>
#include <cmath>

double random_nn2()
{
    static std::uniform_real_distribution<> distr(0, 1);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return distr(gen);
}

double random_range2(double a, double b)
{
    return random_nn2() * (b - a) + a;
}

int main()
{

    MNIST_DATASET *dataset = read_dataset(); // Read the dataset from memory

    int num_layers = 3;
    std::vector<int> neuron_counts = {784, 300, 10};
    NeuralNetworkFF net(num_layers, neuron_counts);

    // net.to_external_repr(std::cout);
    net.save_to_file("out_prev.net");
    for (int q = 0; q <1; ++q)
    {
        for (int jk = 0; jk < dataset->training_images.size(); ++jk)
        {

            // int i = (int) floor( dataset->test_labels.size() * random_range2(0, 1) );
            int i = jk;
            // int i = 0;
            std::vector<double> input;
            for (int j = 0; j < dataset->training_images[i].size(); ++j)
            {
                input.push_back(dataset->training_images[i][j]);
                // if (dataset->training_images[i][j])
                // {
                //     input.push_back(1);
                // }
                // else
                // {
                //     input.push_back(0);
                // }
            }
            std::vector<double> output(10, 0);
            output[dataset->training_labels[i]] = 1;
            net.train_on_example(input, output);

            if (jk % 500 == 0)
            {
                net.update_weights(0.25);
            }

            if (jk % 3000 == 0)
            {
                std::cout << "Trained on " << jk << " examples | " << dataset->training_images.size() - jk << " Remaining " << std::endl;
            }
        }
    }

    cout << "\n\n\n##################AFTER####################\n\n\n";

    // net.to_external_repr(std::cout);

    std::cout << "Testing Phase!" << std::endl;

    int examples = 0;
    int correct = 0;

    std::vector<double> input;
    for (int j = 0; j < dataset->training_images[0].size(); ++j)
    {
        input.push_back(dataset->training_images[0][j]); 
        // if (dataset->training_images[0][j])
        // {
        //     input.push_back(1);
        // }
        // else
        // {
        //     input.push_back(0);
        // }
    }
    std::vector<double> output;
    net.forwardPass(input, output);

    for (auto number : output)
        std::cout << number << " ";
    std::cout << std::endl
              << std::endl;

    std::cout << (int)dataset->training_labels[0] << std::endl;

    for (int i = 0; i < 400; ++i)
    {
        std::vector<double> input;
        for (int j = 0; j < dataset->test_images[i].size(); ++j)
        {

            //if (dataset->training_images[i][j])
            //    input.push_back(1);
            //else
            //    input.push_back(0);

            input.push_back(dataset->test_images[i][j]);
        }

        std::vector<double> output;
        net.forwardPass(input, output);

        double max = -1;
        int prediction = 0;
        for (int j = 0; j < 10; ++j)
        {
            if (output[j] > max)
            {
                prediction = j;
                max = output[j];
            }
        }

        std::cout << "Predicted " << prediction << " | Correct " << (int)dataset->test_labels[i] << std::endl;
        //std::cout << "Output: ";

        // for (auto number : output)
        //     std::cout << number << " ";
        // std::cout << std::endl
        //           << std::endl;

        ++examples;

        if (prediction == (int)dataset->test_labels[i])
        {
            ++correct;
        }else{
            dataset->display_image(i, dataset->test_images); 
        }
    }

    std::cout << "Correct Rate " << ((double)correct / (double)examples) << std::endl;

    net.save_to_file("examples/MNIST/out.net"); 

}