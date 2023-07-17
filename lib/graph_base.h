#ifndef GERONIMO_GRAPH_BASE_H
#define GERONIMO_GRAPH_BASE_H

#include <vector>
#include <memory>

#include "layer_base.h"

namespace geronimo::graph {

    /**
     * @brief Abstract base class for all graph objects.
     */
    class base {
    protected:
        std::vector<std::unique_ptr<layer::base>> m_layers;
    public:
        virtual void add(layer::base* layer) = 0;
        virtual arma::mat& predict(arma::mat inputs) = 0;
        virtual void fit(std::vector<arma::mat>& inputs, std::vector<arma::mat>& outputs, size_t epochs, double learning_rate) = 0;
    };
}

#endif