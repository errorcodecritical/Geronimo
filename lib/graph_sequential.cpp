#include "graph_sequential.h"

geronimo::graph::sequential::sequential() = default;

geronimo::graph::sequential::~sequential() = default;

void geronimo::graph::sequential::add(layer::base* new_layer) {
    m_layers.emplace_back(new_layer);
}

arma::mat& geronimo::graph::sequential::predict(arma::mat inputs) {
    arma::mat& result = inputs;

    for (int i = 0; i < m_layers.size(); i++) {
        result = m_layers[i]->activate(result);
    }

    return result;
}

void geronimo::graph::sequential::fit(std::vector<arma::mat>& inputs, std::vector<arma::mat>& outputs, size_t epochs, double learning_rate) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, inputs.size() - 1);

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        size_t n = dist(rng);
        arma::mat& predicted = predict(inputs[n]);
        arma::mat& expected = outputs[n]; 
        arma::mat errors = (predicted - expected);

        for (size_t depth = 1; depth <= m_layers.size(); depth++) {
            errors = m_layers[m_layers.size() - depth]->optimize(errors, learning_rate);
        }
    }
}

std::vector<double> geronimo::graph::sequential::sample_fit(std::vector<arma::mat>& inputs, std::vector<arma::mat>& outputs, size_t epochs, double learning_rate) {
    std::vector<double> samples(epochs);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, inputs.size() - 1);

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        size_t n = dist(rng);
        arma::mat& predicted = predict(inputs[n]);
        arma::mat& expected = outputs[n];
        arma::mat errors = (predicted - expected);
        
        samples[epoch] = arma::accu(errors % errors) / errors.size();

        for (size_t depth = 1; depth <= m_layers.size(); depth++) {
            errors = m_layers[m_layers.size() - depth]->optimize(errors, learning_rate);
        }
    }

    return samples;
}

