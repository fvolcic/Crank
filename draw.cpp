#include <xd/xd.hpp>
#include <vector>
#include "../include/crank.h"
#include "../include/mnist/mnist.h"
#include <string>

using namespace xd;

MNIST_DATASET * dataset; 

static const int cols = 600;
static const int rows = 600;

float mouseX;
float mouseY;

static bool pressed = false;

static int prediction = 0; 

std::vector<std::vector<bool>> mypixels;

NeuralNetworkFF net("trained.net");  

int make_prediction(std::vector<std::vector<bool>> & pix){

    std::vector<double> input(784); 
    int index = 0;
    for(int i = 0; i < 28; ++i){
        for(int j = 0; j < 28; ++j){
            input[index] = pix[i][j] ? 1 : 0;
            dataset->test_images[0][index] = pix[i][j] ? 100 : 0;
            ++index; 
        }
    }
    dataset->display_image(0, dataset->test_images); 
    auto result = net.forwardPass(input); 

    int max_index = 0;
    double max = result[0]; 

    for(int i = 0; i < 10; ++i){
        std::cout << result[i] << " "; 
        if(result[i] > max){
            max = result[i];
            max_index = i;
        }
    }
    std::cout << std::endl; 
    return max_index + 1; 

}

void onMouseMoved(float x, float y)
{
    mouseX = x;
    mouseY = y;
}

void onMousePressed(int button)
{
    pressed = true;
}

void onMouseReleased(int button)
{
    pressed = false;
}

void setup()
{
    dataset = read_dataset(); 
    mypixels = std::vector<std::vector<bool>>(28, std::vector<bool>(28, false));
    mouseReleased(onMouseReleased);
    mouseMoved(onMouseMoved);
    mousePressed(onMousePressed);
    size(cols, rows);
}

void draw()
{

    if (pressed)
    {
        int row = round((mouseY / rows) * 28);
        int col = round((mouseX / cols) * 28);

        mypixels[row][col] = true;
    }

    background(vec4(255));

    stroke(vec4(0, 0, 0, 100));
    strokeWeight(2);

    for (double i = 0; i < cols; i += (cols / 28.0))
    {
        line(i, 0, i, rows);
    }
    for (double i = 0; i < rows; i += (rows / 28.0))
    {
        line(0, i, cols, i);
    }

    fill(vec4(0, 0, 0, 255));
    for (int row = 0; row < mypixels.size(); ++row)
    {
        for (int col = 0; col < mypixels[0].size(); ++col)
        {
            if (mypixels[row][col])
            {
                rect(col * (cols / 28.0), row * (rows / 28.0), (cols / 28.0), (rows / 28.0));
            }
        }
    }

    prediction = make_prediction(mypixels); 

    fill(vec4(255));
    rect(0, 0, 50, 50); 
    fill(vec4(0, 0, 0, 100)); 
    text(std::to_string(prediction), 5.0f, 20.0f);

}

void destroy()
{
}
