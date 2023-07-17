#ifndef GERONIMO_GRAPH_SEQUENTIAL_H
#define GERONIMO_GRAPH_SEQUENTIAL_H

#include "graph_base.h"

namespace geronimo::graph {
    /**
     * @brief Linear stack of layers.
     */
    class sequential : public base {  
    public:
        sequential();
        ~sequential();

        /**
         * @brief Append a new layer to the end of the layer stack.
         * @param new_layer
         */
        virtual void add(layer::base* new_layer) override;

        /**
         * @brief Perform inference on a given set of inputs. Calls layer activation methods in order (feedforward).
         * @param inputs 
         * @return arma::mat& 
         */
        virtual arma::mat& predict(arma::mat inputs) override;

        /**
         * @brief Perform training given sets of inputs and expected outputs. Calls optimization methods in reverse order (backpropagation).
         * @param inputs Set of training inputs.
         * @param outputs Set of training outputs.
         * @param epochs Number of iterations to run optimization.
         * @param learning_rate Training multiplier.
         */
        virtual void fit(std::vector<arma::mat>& inputs, std::vector<arma::mat>& outputs, size_t epochs, double learning_rate) override;

        /**
         * @brief Same as sequential::fit(), but provides per-epoch error sample data;
         * @param inputs 
         * @param outputs 
         * @param epochs 
         * @param learning_rate 
         * @return error sample data
         */
        std::vector<double> sample_fit(std::vector<arma::mat>& inputs, std::vector<arma::mat>& outputs, size_t epochs, double learning_rate);
    };
}

#endif