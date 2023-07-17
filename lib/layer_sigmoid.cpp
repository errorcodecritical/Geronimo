#include "layer_sigmoid.h"

geronimo::layer::sigmoid::sigmoid(size_t n_inputs, size_t n_outputs, double v_weights, double v_biases) {
    m_data.emplace_back(1, n_inputs);                                           // inputs
    m_data.emplace_back(n_inputs, n_outputs, arma::fill::randn) * v_weights;    // weights
    m_data.emplace_back(1, n_outputs, arma::fill::randn) * v_biases;            // biases
    m_data.emplace_back(1, n_outputs);                                          // outputs
    m_data.emplace_back(1, n_outputs);                                          // gradients
}

geronimo::layer::sigmoid::sigmoid(std::vector<arma::mat> import) {
    m_data = import;
}

geronimo::layer::sigmoid::~sigmoid() = default;

arma::mat& geronimo::layer::sigmoid::activate(arma::mat& inputs) {
    accessor* data = reinterpret_cast<accessor*>(this->m_data.data());

    data->inputs = inputs;
    data->outputs = 1.0 / (1.0 + exp(-(data->inputs * data->weights + data->biases)));
    data->gradients = data->outputs % (1.0 - data->outputs);

    return data->outputs;
}

arma::mat& geronimo::layer::sigmoid::optimize(arma::mat& errors, double learning_rate) {
    accessor* data = reinterpret_cast<accessor*>(this->m_data.data());

    arma::mat temp = errors % data->gradients;
    errors = temp * data->weights.t();

    data->weights -= data->inputs.t() * temp * learning_rate;
    data->biases -= temp * learning_rate;

    return errors;
}