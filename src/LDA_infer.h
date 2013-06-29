#include "../plda/common.h"
#include "../plda/model.h"
#include "../plda/sampler.h"

#ifndef LDA_INFER
#define LDA_INFER


namespace plda_namespace {
    class LDA_infer {
        public:
            LDA_infer(std::string, double, double, int, int, int);
            ~LDA_infer();
            std::vector<double> run(std::string);
            std::map<std::string, double> get_related_words(
                std::vector<double>, int);

        private:
            LDA_infer(const LDA_infer& that);
            LDA_infer& operator=(const LDA_infer&);
            learning_lda::LDASampler *sampler;
            learning_lda::LDAModel *model;
            std::map<string, int> word_index_map;
            std::vector<std::string> index_word_vec;
            int total_iterations;
            int burnin_iterations;
    };
}

#endif
