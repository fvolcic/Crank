/**
 * @file learning_functions.h
 * 
 * @brief File contains a number of different learning 
 * @version 0.1
 * @date 2022-01-25
 * 
 * @copyright Copyright (c) 2022
 * 
 */

/**
 * @brief This file contains functions that allow the user to specify the learning rate of the neural network as a function of different inputs. 
 *        A neural network learning function is constructed first by the user, then passed to functions that need them.
 */

class LearningRateFunctionBase
{
public:
    /**
     * @brief Get the learning rate object
     * 
     * @return double 
     */
    virtual double get_learning_rate() = 0;
};

/**
 * @brief Class for the first learning function
 * 
 *  */
class ConstantLearningFunction : public LearningRateFunctionBase
{

public:
    /**
     * @brief Construct a new Constant Learning Function object
     * 
     */
    ConstantLearningFunction();

    /**
     * @brief Get the learning rate of the instantiation
     * 
     * @return double 
     */
    double get_learning_rate() override;

private:
};