#include "cegb.hpp"

#include <LightGBM/utils/openmp_wrapper.h>

#include <LightGBM/utils/common.h>

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <ctime>

#include <chrono>
#include <set>
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

  Log::Debug("CEGB: Training finished for iter %d, total cost %f.", iter_,
             cost);
  return res;
}

void CEGB::ResetFeatureTracking() {
  lazy_feature_used.clear();
  coupled_feature_used.clear();

  if (train_data_ != nullptr) {
    lazy_feature_used.resize(train_data_->num_total_features() *
                             train_data_->num_data());
    coupled_feature_used.resize(train_data_->num_total_features());
  }
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
          coupled_feature_used, iter_features_used, bag_data_indices_));
}

bool CEGB::LoadModelFromString(const std::string &model_str) {
  allow_train = false;
  return GBDT::LoadModelFromString(model_str);
}

static double find_cost_or_zero(std::map<int, double> &m, int feature) {
  auto res = m.find(feature);
  if (res == m.end())
    return 0;
  else
    return res->second;
}

inline void CEGB::InitPredict(int num_iteration) {
  GBDT::InitPredict(num_iteration);

  if (num_tree_per_iteration_ > 1)
    Log::Fatal(
        "CEGB::InitPredict not implemented for num_tree_per_iteration_ > 1.");

  models_costinfo.clear();
  models_costinfo.resize(num_tree_per_iteration_);

  for (int i = 0; i < num_tree_per_iteration_; ++i)
    models_costinfo[i].resize(models_[i]->num_leaves());

  for (int i = 0; i < num_tree_per_iteration_; ++i) {
    auto &model = models_[i];
    int n_leafs = model->num_leaves();

    for (int i_leaf = 0; i < n_leafs; ++i) {
      detail::CEGB_CostInfo &cinfo = models_costinfo[i][i_leaf];

      cinfo.n_splits = 0;
      cinfo.features.clear();

      std::vector<int> path = model->GetPathToLeaf(i_leaf);

      for (int i_split_node : path) {
        cinfo.features.insert(model->split_feature(i_split_node));
        cinfo.n_splits++;
      }
    }
  }
}

void CEGB::PredictMulti(const double *features, double *output_raw,
                        double *output, double *leaf, double *cost) const {

  std::set<int> features_used;
  double i_cost = 0;
  double i_pred = 0;

  if (num_tree_per_iteration_ > 1)
    Log::Fatal(
        "CEGB::PredictMulti not implemented for num_tree_per_iteration_ > 1.");

  for (int i = 0; i < num_iteration_for_pred_; ++i) {
    auto &model = models_[i];

    int i_leaf = model->PredictLeafIndex(features);

    const detail::CEGB_CostInfo &cinfo = models_costinfo[i][i_leaf];

    // feature penalty
    for (int i_feature : cinfo.features) {
      if (features_used.find(i_feature) == features_used.end())
        continue;

      i_cost += find_cost_or_zero(
          gbdt_config_->cegb_config.penalty_feature_lazy, i_feature);
      i_cost += find_cost_or_zero(
          gbdt_config_->cegb_config.penalty_feature_coupled, i_feature);
      features_used.insert(i_feature);
    }

    // split penalty
    i_cost += gbdt_config_->cegb_config.penalty_split * cinfo.n_splits;

    // prediction
    i_pred += model->LeafOutput(i_leaf);

    if (leaf != nullptr)
      leaf[i] = i_leaf;

    if (cost != nullptr)
      cost[i] = i_cost;

    if (output_raw != nullptr)
      output_raw[i] = i_pred;

    if (output != nullptr) {
      output[i] = i_pred;

      if (objective_function_ != nullptr)
        objective_function_->ConvertOutput(output, output);
    }
  }
}

void CEGB::MyAddPredictionToScore(const Tree *tree,
                                  const data_size_t *data_indices,
                                  data_size_t data_cnt, int cur_tree_id) {

  if (data_indices == nullptr || data_cnt == 0) {
    tree->AddPredictionToScoreGetFeatures(
        train_data_, train_data_->num_data(),
        (double *)train_score_updater_->score() +
            cur_tree_id * train_score_updater_->num_data(),
        lazy_feature_used, train_data_->num_data());

  } else {
    tree->AddPredictionToScoreGetFeatures(
        train_data_, data_indices, data_cnt,
        (double *)train_score_updater_->score() +
            cur_tree_id * train_score_updater_->num_data(),
        lazy_feature_used, train_data_->num_data());
  }
}

void CEGB::UpdateScore(const Tree *tree, const int cur_tree_id) {
  // update training score
  if (!is_use_subset_) {
    train_score_updater_->AddScore(tree_learner_.get(), tree, cur_tree_id);
  } else {
    MyAddPredictionToScore(tree, nullptr, 0, cur_tree_id);
  }
  // update validation score
  for (auto &score_updater : valid_score_updater_) {
    score_updater->AddScore(tree, cur_tree_id);
  }
}

void CEGB::UpdateScoreOutOfBag(const Tree *tree, const int cur_tree_id) {
  // we need to predict out-of-bag scores of data for boosting
  if (num_data_ - bag_data_cnt_ > 0 && !is_use_subset_) {
    MyAddPredictionToScore(tree, bag_data_indices_.data() + bag_data_cnt_,
                           num_data_ - bag_data_cnt_, cur_tree_id);
  }
}

} // namespace LightGBM
