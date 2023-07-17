#include <iostream>
#include <memory>
#include <vector>
#include <matplot/matplot.h> // Used solely for visualizing data.

#include "geronimo.h"

int main() {
    using namespace geronimo;
    
    arma::arma_rng::set_seed_random(); // Required for layer initialization;

    // Generate a simple 2 - 3 - 2 linear network to solve XOR;
    graph::sequential network;
    network.add(new layer::sigmoid(2, 3));
    network.add(new layer::leaky(3, 2, 0.1));

    // Define the network's training data;
    std::vector<arma::mat> in = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<arma::mat> out = {
        {1, 0},
        {0, 1},
        {0, 1},
        {1, 0}
    };

    // Train the network for 10000 epochs, with a learning rate of 0.1;
    std::vector<double> samples = network.sample_fit(in, out, 10000, 0.1);

    // Log predicted values for the problem;
    std::cout << network.predict({0, 0});
    std::cout << network.predict({0, 1});
    std::cout << network.predict({1, 0});
    std::cout << network.predict({1, 1});
    
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
    auto Z1 = matplot::transform(X, Y, [&network](double x, double y) {
        return network.predict({x, y})(0, 1);
    });
    auto Z2 = matplot::transform(X, Y, [&network](double x, double y) {
        return network.predict({x, y})(0, 0);
    });

    matplot::mesh(X, Y, Z1)->edge_color("blue");
    matplot::mesh(X, Y, Z2)->edge_color("red");
    matplot::show();

    return 0;
}
