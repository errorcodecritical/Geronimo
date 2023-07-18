#include <iostream>
#include <memory>
#include <vector>

#include <matplot/matplot.h> // Used for visualizing data.
#include "geronimo.h"

/*
    This is a simple example for training a network to solve the XOR problem.
 
    Given two inputs X, Y, the truth table for X ^ Y is as follows:

    | X | Y | X ^ Y |
    |---|---|-------|
    | 0 | 0 |     0 |
    | 0 | 1 |     1 |
    | 1 | 0 |     1 |
    | 1 | 1 |     0 |

    NOTE: To use matplot (after installing it) you'll need to update the Makefile CFLAGS:
    
    CFLAGS := -larmadillo -lmatplot

*/

int main() {
    using namespace geronimo;
    
    // Generate a simple 2 - 3 - 1 linear network to solve XOR;
    graph::sequential network;
    
    // Append a new 2-in/3-out sigmoid layer to the end of the network;
    network.add(new layer::sigmoid(2, 3));

    // Append a new 3-in/2-out, alpha=0.1, leaky relu layer to the end of the network;
    network.add(new layer::leaky(3, 1, 0.1));

    // Define the network's training data;
    std::vector<arma::mat> in = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<arma::mat> out = {{0}, {1}, {1}, {0}};

    // Log initial predictions for comparison;
    std::cout << "\nInference before training:" << std::endl;
    std::cout << "\nInputs:" << in[0] << "Outputs:" << network.predict(in[0]);
    std::cout << "\nInputs:" << in[1] << "Outputs:" << network.predict(in[1]);
    std::cout << "\nInputs:" << in[2] << "Outputs:" << network.predict(in[2]);
    std::cout << "\nInputs:" << in[3] << "Outputs:" << network.predict(in[3]);

    // Train the network for 10000 epochs, with a learning rate of 0.1 and return error samples for each epoch;
    std::vector<double> samples = network.sample_fit(in, out, 10000, 0.1);

    // Log predicted values for the problem;
    std::cout << "\nInference after training:" << std::endl;
    std::cout << "\nInputs:" << in[0] << "Outputs:" << network.predict(in[0]);
    std::cout << "\nInputs:" << in[1] << "Outputs:" << network.predict(in[1]);
    std::cout << "\nInputs:" << in[2] << "Outputs:" << network.predict(in[2]);
    std::cout << "\nInputs:" << in[3] << "Outputs:" << network.predict(in[3]);
    
    // View the evolution of the network's output error;
    matplot::figure();
    matplot::hold(true);
    matplot::grid(true);
    matplot::xlabel("Epoch (iterations)");
    matplot::ylabel("Error (mean squared)");
    matplot::plot(samples, "-")->color("blue");

    // View the mapped input-output spaces of the network;
    matplot::figure();
    matplot::hold(true);
    matplot::grid(true);
    matplot::xlabel("X");
    matplot::ylabel("Y");
    matplot::zlabel("Z = X ⊕ Y");
    
    auto [X, Y] = matplot::meshgrid(matplot::iota(0.0, 0.05, 1.0));
    auto Z = matplot::transform(X, Y, [&network](double x, double y) {
        return network.predict({x, y})(0, 0);
    });

    matplot::mesh(X, Y, Z)->edge_color("blue");
    matplot::show();

    return 0;
}
