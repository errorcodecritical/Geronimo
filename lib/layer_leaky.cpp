#include "layer_leaky.h"

geronimo::layer::leaky::leaky(size_t n_inputs, size_t n_outputs, double alpha, double v_weights, double v_biases) {
    m_data.emplace_back(1, n_inputs);                                           // inputs
    m_data.emplace_back(n_inputs, n_outputs, arma::fill::randn) * v_weights;    // weights
    m_data.emplace_back(1, n_outputs, arma::fill::randn) * v_biases;            // biases
    m_data.emplace_back(1, n_outputs);                                          // outputs
    m_data.emplace_back(1, n_outputs);                                          // gradients
    m_data.emplace_back(1, 1) = {alpha};                                        // alpha
}

geronimo::layer::leaky::leaky(std::vector<arma::mat> import) {
    m_data = import;
}

geronimo::layer::leaky::~leaky() = default;

arma::mat& geronimo::layer::leaky::activate(arma::mat& inputs) {
    accessor* data = reinterpret_cast<accessor*>(this->m_data.data());
    double alpha = data->alpha(0, 0);

    data->inputs = inputs;
    data->outputs = data->inputs * data->weights + data->biases;
    data->gradients = data->outputs;

    data->outputs.for_each([&alpha](double& value) { value = value > 0.0 ? value : value * alpha; });
    data->gradients.for_each([&alpha](double& value) { value = value > 0.0 ? 1.0 : alpha; });

    return data->outputs;
}

arma::mat& geronimo::layer::leaky::optimize(arma::mat& errors, double learning_rate) {
    accessor* data = reinterpret_cast<accessor*>(this->m_data.data());

    arma::mat temp = errors % data->gradients;
    errors = temp * data->weights.t();

    data->weights -= data->inputs.t() * temp * learning_rate;
    data->biases -= temp * learning_rate;

    return errors;
}