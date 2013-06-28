/*
    A wrapper class to pass to Python. 
    This class has the root from infer.cc of the original plda package.

    JF SEOmoz 04/18/2013
    MP Moz 06/18/2013 - updated to return related words
*/

#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <algorithm>
#include <assert.h>

#include "../plda/common.h"
#include "../plda/document.h"
#include "../plda/model.h"
#include "../plda/sampler.h"
#include "../plda/cmd_flags.h"
#include "LDA_infer.h"

using learning_lda::LDASampler;
using learning_lda::LDADocument;
using learning_lda::DocumentWordTopicsPB;
using learning_lda::RandInt;
using learning_lda::LDAModel;
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

    // make the index -> word map
    for (std::map<std::string, int>::const_iterator iter = word_index_map.begin();
        iter != word_index_map.end(); ++iter) {
        index_word_map.push_back(iter->first);
    }
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

// compare function for heap
// returns True if first is larger then second
bool compare_word_scores(std::pair<std::string, double> a,
    std::pair<std::string, double> b) {
    return a.second > b.second;
}


std::map<std::string, double> LDA_infer::get_related_words(
    std::vector<double> topics, int N) {

    int ntopics = model->num_topics();
    assert (topics.size() == ntopics);

    // process: maintain a heap of the top N words
    // read through the model, compute each word score and return
    // the top N
    std::vector< std::pair<std::string, double> > topN_words;
    std::make_heap(topN_words.begin(), topN_words.end(), compare_word_scores);

    int nwords_seen = 0;
    for (LDAModel::Iterator iter(model); !iter.Done(); iter.Next()) {

        // compute the score for this word
        double this_word_score = 0.0;
        TopicCountDistribution topic_distribution = iter.Distribution();
        for (int k=0; k < ntopics; ++k) {
            this_word_score += topic_distribution[k] * topics[k];
        }

        // update the heap
        nwords_seen += 1;
        topN_words.push_back(std::make_pair(index_word_map[iter.Word()], this_word_score));
        std::push_heap(
            topN_words.begin(), topN_words.end(), compare_word_scores);
        if (nwords_seen > N) {
            // pop the smallest value
            std::pop_heap(
                topN_words.begin(), topN_words.end(), compare_word_scores);
            topN_words.pop_back();
        }
    }

    // make return vector
    std::map<std::string, double> ret;
    for (int k=0; k < topN_words.size(); ++k) {
        ret[topN_words[k].first] = topN_words[k].second;
    }
    return ret;
}

