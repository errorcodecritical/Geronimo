#ifndef GERONIMO_LAYER_SIGMOID_H
#define GERONIMO_LAYER_SIGMOID_H

#include "layer_base.h"

namespace geronimo::layer {
    class sigmoid : public base {
    private:
        struct accessor {
            arma::mat inputs;
            arma::mat weights;
            arma::mat biases;
            arma::mat outputs;
            arma::mat gradients;
        };

    public:
        sigmoid(size_t n_inputs, size_t n_outputs, double v_weights = 1.0, double v_biases = 1.0);
        sigmoid(std::vector<arma::mat> import);
        ~sigmoid();
        
        virtual arma::mat& activate(arma::mat& inputs) override;
        virtual arma::mat& optimize(arma::mat& errors, double learning_rate) override;
    };
};

#endif