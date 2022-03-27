/**
 * @file snake.cpp
 *
 * @brief Neural network that learns to play snake
 * @version 0.1
 * @date 2022-03-26
 *
 * @copyright Copyright (c) 2022
 *
 */

using namespace std;

#include <iostream>
#include <vector>
#include <deque>
#include "../../include/crank.h"
#include <queue>
#include <utility>

class Snake
{

public:
    enum class Move
    {
        right,
        left,
        up,
        down,
        skip
    };

    Snake(int xSize, int ySize, int xPos, int yPos) : xSize(xSize), ySize(ySize),
                                                      xPos(xPos), yPos(yPos)
    {
        past_positions.push_front({xPos, yPos});
        update_food_location();
    }

    /**
     * @brief Returns false if the game has ended
     *
     * @param direction
     * @return true
     * @return false
     */
    bool run_iteration(Move direction)
    {
        ++moves; 
        ++moves_total;
        switch (direction)
        {
        case Move::left:
            --xPos;
            break;
        case Move::right:
            ++xPos;
            break;
        case Move::up:
            ++yPos;
            break;
        case Move::down:
            --yPos;
            break;
        default:
            return true;
        }

        if (check_scored())
        {
            score++;
            update_food_location();
            moves = 0; 
        }

        return game_over();
    }

    int get_score()
    {
        return 100 * score + (1 + moves_total);
    }

    void print_game(ostream &os)
    {
        cout << "!" << foodX << " " << foodY << " " << xPos << " " << yPos << endl;
        for (int col = 0; col < xSize + 2; ++col)
        {
            os << "*";
        }
        os << "\n";
        for (int row = 0; row < ySize; ++row)
        {
            os << "*";
            for (int col = 0; col < xSize; ++col)
            {
                if (row == yPos && col == xPos)
                {
                    os << "+";
                }
                else if (row == foodY && col == foodX)
                {
                    os << "o";
                }
                else
                {
                    os << " ";
                }
            }
            os << "*\n";
        }

        for (int col = 0; col < xSize + 2; ++col)
        {
            os << "*";
        }
        os << "\n";
    }

    int getX()
    {
        return xPos;
    }

    int getY()
    {
        return yPos;
    }

    bool check_scored()
    {
        return xPos == foodX && yPos == foodY;
    }

    bool game_over()
    {
        return (xPos < xSize && xPos >= 0 && yPos < ySize && yPos >= 0 && score < 50 && moves < 100);
    }

    void update_food_location()
    {
        foodX = round(random_range(0, xSize - 1));
        foodY = round(random_range(0, ySize - 1));
    }

    int xSize, ySize;
    int score = 0;
    int moves = 0; 
    int moves_total = 0; 
    int length = 1;

    int xPos, yPos;

    int foodX, foodY;

    deque<pair<int, int>> past_positions;
};

// int main()
// {

//     Snake snake(20, 20, 0, 0);

//     int move;

//     snake.print_game(cout);

//     while (1)
//     {
//         cin >> move;

//         switch (move)
//         {
//         case 1:
//             snake.run_iteration(Snake::Move::left);
//             break;
//         case 2:
//             snake.run_iteration(Snake::Move::right);
//             break;
//         case 3:
//             snake.run_iteration(Snake::Move::up);
//             break;
//         case 4:
//             snake.run_iteration(Snake::Move::down);
//             break;
//         default:
//             break;
//         }
//         snake.print_game(cout);
//     }

//     snake.print_game(cout);
//     snake.run_iteration(Snake::Move::right);
//     snake.print_game(cout);
// }

struct gen_offspring_config
{

    gen_offspring_config(int rate, double size) : mutation_rate(rate), mutation_size(size) {}

    int mutation_rate = 8;
    double mutation_size = 0.2;
};

/**
 * @brief take two neural networks, and return a new neural network that is the offspring of the parent
 *
 * @param parent1
 * @param parent2
 * @return NeuralNetworkFF*
 */
NeuralNetworkFF *generate_offspring(NeuralNetworkFF *parent1, NeuralNetworkFF *parent2, const gen_offspring_config *config)
{
    std::stringstream ss;
    parent1->to_external_repr(ss);
    NeuralNetworkFF *child_net = new NeuralNetworkFF(ss);

    for (int layer = 0; layer < child_net->neurons.size(); ++layer)
    {
        for (int neuron_index = 0; neuron_index < child_net->neurons[layer].size(); ++neuron_index)
        {
            auto &neuron = child_net->neurons[layer][neuron_index];

            auto &neuron_other = parent2->neurons[layer][neuron_index];

            for (int weight_index = 0; weight_index < neuron.weights.size(); ++weight_index)
            {
                neuron.weights[weight_index] = (neuron.weights[weight_index] + neuron_other.weights[weight_index]) / 2;

                if (random_range(0, 10) > config->mutation_rate)
                {
                    neuron.weights[weight_index] += random_range(-config->mutation_size, config->mutation_size);
                }
            }

            neuron.bias = (neuron.bias + neuron_other.bias) / 2;
            if (random_range(0, 10) > config->mutation_rate)
            {
                neuron.bias += random_range(-config->mutation_size, config->mutation_size);
            }
        }
    }

    return child_net;
}

class NetworkCompare
{

public:
    bool operator()(std::pair<double, NeuralNetworkFF *> first, std::pair<double, NeuralNetworkFF *> second)
    {
        return first.first > second.first;
    }
};

double measure_fitness(NeuralNetworkFF *network, bool print = false)
{

    Snake snake(3, 3, 2, 2);

    Snake::Move move = Snake::Move::skip;

    while (snake.run_iteration(move))
    {
        vector<double> input = {(double)snake.xPos, (double)snake.yPos, (double)snake.foodX, (double)snake.foodY};
        auto output = network->forwardPass(input); 

        double max = output[0];
        unsigned int max_index = 0; 

        for(int i = 1; i < output.size(); ++i){
            if(output[i] > max){
                max = output[i]; 
                max_index = i; 
            }
            
        }

        move = Snake::Move(max_index); 
        if(print)
        snake.print_game(cout);
    }


    return snake.get_score(); 
}

int main()
{

    int num_organisms = 100;
    int k_count = 10;
    int num_generations = 5000;

    int num_layers = 3;
    std::vector<int> neuron_counts = {4, 4, 4};

    std::vector<NeuralNetworkFF *> organisms;
    std::priority_queue<std::pair<double, NeuralNetworkFF *>, std::vector<std::pair<double, NeuralNetworkFF *>>, NetworkCompare> networks_by_fitness;

    // Initially load the organisms
    for (int i = 0; i < num_organisms; ++i)
    {
        organisms.push_back(new NeuralNetworkFF(num_layers, neuron_counts));
    }

    // Determine the fitness of the networks

    for (int generation = 0; generation < num_generations; ++generation)
    {
        networks_by_fitness = std::priority_queue<std::pair<double, NeuralNetworkFF *>, std::vector<std::pair<double, NeuralNetworkFF *>>, NetworkCompare>();

        for (auto organism : organisms)
        {
            networks_by_fitness.push({measure_fitness(organism), organism});
        }

        //std::cout << "Generation " << generation << " best Fitness: " << networks_by_fitness.top().first << std::endl;

        std::cout << "Generation " << generation << " best Fitness: " << measure_fitness(networks_by_fitness.top().second, false) << std::endl;

        // if(generation % 1 == 0){

        // string save_name = "genetics/network_save_generation_" + std::to_string(generation);
        // networks_by_fitness.top().second->save_to_file(save_name);

        //}

        std::vector<NeuralNetworkFF *> top_organisms;

        for (int i = 0; i < k_count; ++i)
        {
            top_organisms.push_back(networks_by_fitness.top().second);
            networks_by_fitness.pop();
        }

        while (!networks_by_fitness.empty())
        {
            delete networks_by_fitness.top().second;
            networks_by_fitness.pop();
        }

        organisms.clear();
        for (int i = 0; i < num_organisms; ++i)
        {
            int index_a = floor((double)i / top_organisms.size());

            if (index_a >= num_organisms)
            {
                index_a = (int)random_range(0, num_organisms - 1);
            }

            int index_b = i % top_organisms.size();

            gen_offspring_config config(0, 10);
            organisms.push_back(generate_offspring(top_organisms[index_a], top_organisms[index_b], &config));
        }

        for (int i = 0; i < num_organisms - organisms.size(); ++i)
        {
            organisms.push_back(new NeuralNetworkFF(num_layers, neuron_counts));
        }
    }
}