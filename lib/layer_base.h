#ifndef GERONIMO_LAYER_BASE_H
#define GERONIMO_LAYER_BASE_H

#include <armadillo>

namespace geronimo::layer {
    /**
     * @brief Abstract base class for all layer objects.
     */
    class base {
    protected:
        std::vector<arma::mat> m_data;
    public:
        virtual arma::mat& activate(arma::mat& inputs) = 0;
        virtual arma::mat& optimize(arma::mat& errors, double learning_rate) = 0;
    };
}

#endif