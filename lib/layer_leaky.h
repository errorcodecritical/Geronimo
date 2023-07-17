#ifndef GERONIMO_LAYER_LEAKY_H
#define GERONIMO_LAYER_LEAKY_H

#include "layer_base.h"

namespace geronimo::layer {
    class leaky : public base {
    private:
        struct accessor {
            arma::mat inputs;
            arma::mat weights;
            arma::mat biases;
            arma::mat outputs;
            arma::mat gradients;
            arma::mat alpha;
        };

    public:
        leaky(size_t n_inputs, size_t n_outputs, double alpha = 0.3, double v_weights = 1.0, double v_biases = 1.0);
        leaky(std::vector<arma::mat> import);
        ~leaky();
        
        virtual arma::mat& activate(arma::mat& inputs) override;
        virtual arma::mat& optimize(arma::mat& errors, double learning_rate) override;
    };
};

#endif