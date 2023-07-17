#include <iostream>
#include <vector>
#include <matplot/matplot.h>

#include "model.h"

using namespace model;

int main() {
    graph::sequential network;
    // network.add(layer::relu(2, 3));
    network.add(2, 4, "sigmoid");
    network.add(4, 2, "sigmoid");
        
    std::vector<matrix> in = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<matrix> out = {
        {1, 0},
        {0, 1},
        {0, 1},
        {1, 0}
    };

    auto samples = network.fit(in, out, 50000, .5, 100);

    matplot::figure();
    matplot::hold(true);
    matplot::grid(true);
    matplot::xlabel("Epoch (iterations)");
    matplot::ylabel("Error (predicted - expected)");
    matplot::legend({"Class: False", "Class: True"});

    matplot::plot(samples.x, samples.y[0], "-")->color("blue");
    matplot::plot(samples.x, samples.y[1], "-")->color("red");

    auto [X, Y] = matplot::meshgrid(matplot::iota(0.0, 0.05, 1.0));
    auto Z1 = matplot::transform(X, Y, [&network](double x, double y) {
        return network.predict({x, y})(0, 1);
    });
    auto Z2 = matplot::transform(X, Y, [&network](double x, double y) {
        return network.predict({x, y})(0, 0);
    });

    matplot::figure();
    matplot::hold(true);
    matplot::grid(true);
    matplot::xlabel("X");
    matplot::ylabel("Y");
    matplot::zlabel("Z = xor(X, Y)");

    matplot::mesh(X, Y, Z1)->edge_color("blue");
    matplot::mesh(X, Y, Z2)->edge_color("red");
    matplot::legend({"Class: False", "Class: True"});

    matplot::show();

    return 0;
}