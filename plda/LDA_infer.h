#include "common.h"
#include "model.h"
#include "sampler.h"

#ifndef LDA_INFER
#define LDA_INFER

namespace plda_namespace {
    class LDA_infer {
        private:
            learning_lda::LDASampler *sampler;
            learning_lda::LDAModel *model;
            map<string, int> word_index_map;
            int total_iterations;
            int burnin_iterations;

        public:
            LDA_infer(std::string, double, double, int, int, int);
            ~LDA_infer();
            std::string run(std::string);
    };
}

#endif
