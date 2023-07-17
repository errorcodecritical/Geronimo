#ifndef GERONIMO_LAYER_RELU_H
#define GERONIMO_LAYER_RELU_H

#include "layer_base.h"

namespace geronimo::layer {
    class relu : public base {
    private:
        struct accessor {
            arma::mat inputs;
            arma::mat weights;
            arma::mat biases;
            arma::mat outputs;
            arma::mat gradients;
        };

    public:
        relu(size_t n_inputs, size_t n_outputs, double v_weights = 1.0, double v_biases = 1.0);
        relu(std::vector<arma::mat> import);
        ~relu();
        
        virtual arma::mat& activate(arma::mat& inputs) override;
        virtual arma::mat& optimize(arma::mat& errors, double learning_rate) override;
    };
};

#endif