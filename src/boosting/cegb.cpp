#include "cegb.hpp"

#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/utils/common.h>

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <ctime>

#include <chrono>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace LightGBM {

void CEGB::Init(const BoostingConfig *config, const Dataset *train_data,
                const ObjectiveFunction *objective_function,
                const std::vector<const Metric *> &training_metrics) {
  GBDT::Init(config, train_data, objective_function, training_metrics);
}

void CEGB::ResetTrainingData(
    const BoostingConfig *config, const Dataset *train_data,
    const ObjectiveFunction *objective_function,
    const std::vector<const Metric *> &training_metrics) {
  InitTreeLearner(config);
  GBDT::ResetTrainingData(config, train_data, objective_function,
                          training_metrics);
  ResetFeatureTracking();
  allow_train = true;
}

void CEGB::RollbackOneIter() {
  Log::Fatal("CEGB mode does not support rollback.");
}

bool CEGB::TrainOneIter(const score_t *gradient, const score_t *hessian,
                        bool is_eval) {
  if (!allow_train) {
    Log::Fatal(
        "CEGB mode does not support further training after loading a model.");
    return false;
  }

  if (gbdt_config_->cegb_config.independent_branches)
    iter_features_used.clear();

  bool res = GBDT::TrainOneIter(gradient, hessian, is_eval);

  if (gbdt_config_->cegb_config.independent_branches) {
    for (int i_feature : iter_features_used)
      coupled_feature_used[i_feature] = true;
  }

  return res;
}

void CEGB::ResetFeatureTracking() {
  lazy_feature_used.clear();
  lazy_feature_used.resize(train_data_->num_total_features() *
                           train_data_->num_data());
  coupled_feature_used.clear();
  coupled_feature_used.resize(train_data_->num_total_features());
}

void CEGB::InitTreeLearner(const BoostingConfig *config) {
  if (config->device_type != std::string("cpu"))
    Log::Fatal(
        "CEGB currently only supports CPU tree learner, '%s' is unsupported.",
        config->device_type);
  if (config->tree_learner_type != std::string("serial"))
    Log::Fatal("CEGB currently only supports serial tree learner, '%s' is "
               "unsupported.",
               config->tree_learner_type);

  if (tree_learner_ != nullptr)
    return;

  tree_learner_ =
      std::unique_ptr<TreeLearner>((TreeLearner *)new CEGBTreeLearner(
          &config->tree_config, &config->cegb_config, lazy_feature_used,
          coupled_feature_used, iter_features_used));
}

bool CEGB::LoadModelFromString(const std::string &model_str) {
  allow_train = false;
  return GBDT::LoadModelFromString(model_str);
}

void CEGB::PredictCost(const double *features, double *output) const {
  *output = 0;
}

void CEGB::PredictMulti(const double *features, double *output_raw,
                        double *output, double *leaf, double *cost,
                        bool all_iterations) const {
  if (output_raw)
    PredictRaw(features, output_raw);
  if (output)
    Predict(features, output);
  if (leaf)
    PredictLeafIndex(features, leaf);
  if (cost)
    PredictCost(features, cost);
}

} // namespace LightGBM