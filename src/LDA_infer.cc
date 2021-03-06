/*
    A wrapper class to pass to Python. 
    This class has the root from infer.cc of the original plda package.

    JF SEOmoz 04/18/2013
*/

#include <fstream>
#include <set>
#include <sstream>
#include <string>

#include "../plda/common.h"
#include "../plda/document.h"
#include "../plda/model.h"
#include "../plda/sampler.h"
#include "../plda/cmd_flags.h"
#include "LDA_infer.h"

using learning_lda::LDACorpus;
using learning_lda::LDAModel;
using learning_lda::LDAAccumulativeModel;
using learning_lda::LDASampler;
using learning_lda::LDADocument;
using learning_lda::LDACmdLineFlags;
using learning_lda::DocumentWordTopicsPB;
using learning_lda::RandInt;
using std::ifstream;
using std::ofstream;
using std::istringstream;

using namespace plda_namespace;

LDA_infer::LDA_infer(string model_file, double alpha, double beta, int total_iterations_, 
                      int burnin_iterations_, int seed) {
    /* Notice: if the seed < 0, then use the time call. else use the seed. This is 
       for testing purpose.
    */
    if(seed < 0)
        srand(time(NULL));
    else
        srand(seed);

    ifstream model_fin(model_file.c_str());
    model =  new LDAModel(model_fin, &word_index_map);
    sampler = new LDASampler(alpha, beta, model, NULL);
    total_iterations = total_iterations_;
    burnin_iterations = burnin_iterations_;
}

LDA_infer::~LDA_infer() {
    delete sampler;
    delete model;
}

std::vector<double> LDA_infer::run(string line) {
    std::stringstream out;

    // Keep the following untouched (original codes copied from infer.cc). 
    istringstream ss(line);
    DocumentWordTopicsPB document_topics;
    std::string word;
    int count;
    while (ss >> word >> count) {  // Load and init a document.
      std::vector<int32> topics;
      for (int i = 0; i < count; ++i) {
        topics.push_back(RandInt(model->num_topics()));
      }
      std::map<string, int>::const_iterator iter = word_index_map.find(word);
      if (iter != word_index_map.end()) {
        document_topics.add_wordtopics(word, iter->second, topics);
      }
    }
    LDADocument document(document_topics, model -> num_topics());
    TopicProbDistribution prob_dist(model -> num_topics(), 0);
    for (int iter = 0; iter < total_iterations; ++iter) {
      sampler->SampleNewTopicsForDocument(&document, false);
      // This line changed to use the class variables.
      if (iter >= burnin_iterations) {
        const vector<int64>& document_distribution =
            document.topic_distribution();
        for (int i = 0; i < document_distribution.size(); ++i) {
          prob_dist[i] += document_distribution[i];
        }
      }
    }

    // divide by number of iterations
    double running_iterations = total_iterations - burnin_iterations;
    for (int topic = 0; topic < prob_dist.size(); ++topic) {
        prob_dist[topic] /= running_iterations;
    }
    return prob_dist;
}

