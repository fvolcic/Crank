/**
 * @file ff.h
 * 
 * @brief 
 * @version 0.1
 * @date 2021-12-27
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "activation.h"
#include "learning_functions.h"
#include "neuron.h"
#include "sigmoid.h"
#include <algorithm>
#include <iterator>
#include <vector>
#include <sstream>

/**
 * @brief A class that represents the possible activation functions one can use
 * 
 */
enum class ActivationFunctions
{
   Sigmoid
};

class NeuralNetworkFF
{
public:
   /**
         * @brief Create a new neural network with default randomized weights and bias'
         *       : The default activation function is the sigmoid
         * @param num_layers - The number of layers in the neural network
         * @param neuron_counts - The number of neurons in each layer of the neural network 
         */
   NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts);

   /**
     * @brief Construct the neural network using predefined weights and bias'
     * 
     * @param num_layers - Number of layers in the neural network
     * @param neuron_counts - The number of neurons in each layer of the neural network
     * @param weights - Weights of each neuron
     * @param bias - A bias term for each neuron in the neural network
     */
   NeuralNetworkFF(int num_layers, std::vector<int> &neuron_counts, const std::vector<std::vector<std::vector<double>>> &weights, const std::vector<std::vector<double>> &bias);

   /**
         * @brief Construct a new Neural Network object
         * 
         * @param network - Neural Network that you want copied 
         */
   NeuralNetworkFF(const NeuralNetworkFF &network);

   /**
         * @brief Destroy the Neural Network object
         * 
         */
   ~NeuralNetworkFF();

   /**
         * @brief Compute a forward pass of a neural network given the values of the input layer
         *      - This method copies the output to a output vector
         * 
         * @param input - Input layer of Neural Network
         * @param output - Last Layer of Neural Network 
         */
   void forwardPass(std::vector<double> &input, std::vector<double> &output);

   /**
         * @brief Compute a forward pass of a neural network given the values of the input layer
         *       -
         * @param input Input layer of the Neural Network
         * @return std::vector<int> - Return a copy of the output vector
         */
   std::vector<double> forwardPass(std::vector<double> &input);

   // Training Functions defined below

   /**
     * @brief This takes an input example and the expected output,
     *        and then comptues the derivative of weights and bias'.
     *        After that, the partial derivative is added to the average partial derivative
     *        for each weight and bias.
     * 
     * @param input 
     * @param expected_output 
     */
   void train_on_example(std::vector<double> &input, std::vector<double> &expected_output);

   /**
     * @brief Update the weights and bias' based on the gradients computed in backprop
     * 
     * @param learning_rate - the learning rate to use to update the weights 
     */
   void update_weights(double learning_rate, bool reset = true);

   // TODO Add Move Constructor
   // TODO Implement move semantics for the network

   /**
      * @brief Block the default construction of a Neural Network
      * 
      */
   NeuralNetworkFF() = delete; // No default constructed neural network

   /**
     * @brief Extra configuration settings for the train function
     * 
     */
   struct TrainConfig
   {
      int batch_size = 1;
      int skip_rate = 0;
      int num_training_examples = -1;
      bool verbose = false;
      int verbose_count = 100;

      LearningRateFunctionBase *learning_function = nullptr;
   };

   /**
    * @brief This method provides a training interface where all the training will happen based on the passed parameters. 
    * 
    * @tparam ExamplesIterator - An iterator type for the examples. 
    * @tparam ExpectIterator - an iterator type for the expected output of the network.
    * @param examples_iter - The examples iterator object.
    * @param examples_end - end examples iterator object.
    * @param expect_iter - The expect iterator object.
    * @param expect_end - the expect iterator end.
    * @param config - The training configuration struct.
    */
   template <typename ExamplesIterator, typename ExpectIterator>
   void train(ExamplesIterator examples_iter, ExamplesIterator examples_end,
              ExpectIterator expect_iter, ExpectIterator expect_end,
              TrainConfig *config = nullptr);

   /**
   * @brief Extra config settings for testing the network
   * 
   */
   struct TestConfig
   {
      int max_examples = -1;
   };

   /**
   * @brief The test results from the network test function
   * 
   */
   struct TestResults
   {
      size_t correct;
      size_t incorrect;
      double correct_rate;

      size_t num_examples;

      friend std::ostream& operator<<(std::ostream& os, const TestResults & results);

   };

   /**
    * @brief This function is a a skeleton function for testing the neural network. This function takes a number of easily testable
    *        parameters to make the actual testing of the network as seamless as possible.
    * 
    * @tparam ExamplesIterator - An iterator type for the examples. 
    * @tparam ExpectIterator - an iterator type for the expected output of the network.
    * @tparam OutputCmp - class type with operator()() defined which takes two std::vector<double>'s and then compares if they are "equivalent" (What the network was supposed to output)
    * @param examples_iter - The examples iterator object.
    * @param examples_end - end examples iterator object.
    * @param expect_iter - The expect iterator object.
    * @param expect_end - the expect iterator end.
    * @param output_cmp - the output compare object.
    * @param config - the test net config struct.
    * @return TestResults - the test results from testing the network. 
    */
   template <typename ExamplesIterator, typename ExpectIterator, typename OutputCmp>
   TestResults test(ExamplesIterator examples_iter, ExamplesIterator examples_end,
                    ExpectIterator expect_iter, ExpectIterator expect_end, OutputCmp output_cmp,
                    TestConfig *config = nullptr);

#ifndef NN_DEBUG
protected:
#endif
   /**
      * @brief This goes through all the layers and sets the maxLayer variable
      * 
      */
   void findMaxLayerSize();

#ifndef NN_DEBUG
private:
#endif

   /**
      * @brief Run backprop on the network starting at layer layer
      * 
      * @param layer - the starting layer for back prop
      */
   void back_propagation(int layer);

   /**
      * @brief Compute the loss with respect to the activation for a given neuron
      * 
      * @param neuron - the neuron to compute the function on
      * @param layer - the layer which the neuron resides
      */
   void calculate_dLoss_dActivation(Neuron &neuron, int layer, int index);

   /**
      * @brief 
      * 
      * @param neuron - neuron to compute the function on
      * @param layer - the layer which the neuron resides
      */
   void calculate_dActivation_dInput(Neuron &neuron, int layer);

   /**
      * @brief 
      * 
      * @param neuron - neuron to compute the function on
      * @param layer - the layer which the neuron resides 
      */
   void calculate_dLoss_dWeight_and_dLoss_dBias(Neuron &neuron, int layer);

   /**
     * @brief Return the number of layers in the network
     * 
     * @return size_t 
     */
   size_t get_num_layers();

   // TODO add other members to access the network metadata

   std::vector<std::vector<Neuron>> neurons; // Where all the neurons are stored internally in the network

   int maxLayerSize = -1; // The layer in the network with the most neurons
};