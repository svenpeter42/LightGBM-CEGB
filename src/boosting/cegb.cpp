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

void CEGB::Init(const BoostingConfig *config, const Dataset *train_data, const ObjectiveFunction *objective_function,
                const std::vector<const Metric *> &training_metrics) {
  GBDT::Init(config, train_data, objective_function, training_metrics);
}

void CEGB::ResetTrainingData(const BoostingConfig *config, const Dataset *train_data, const ObjectiveFunction *objective_function,
                             const std::vector<const Metric *> &training_metrics) {
  InitTreeLearner(config);
  GBDT::ResetTrainingData(config, train_data, objective_function, training_metrics);
  ResetFeatureTracking();

  allow_train = true;
}

void CEGB::RollbackOneIter() { Log::Fatal("CEGB mode does not support rollback."); }

bool CEGB::TrainOneIter(const score_t *gradient, const score_t *hessian, bool is_eval) {
  if (!allow_train) {
    Log::Fatal("CEGB mode does not support further training after loading a model.");
    return false;
  }

  if (gbdt_config_->cegb_config.independent_branches)
    iter_features_used.clear();

  bool res = GBDT::TrainOneIter(gradient, hessian, is_eval);

#if 0
  cost = 0;

  for (int j = 0; j < train_data_->num_total_features(); ++j) {
    auto res = gbdt_config_->cegb_config.penalty_feature_lazy.find(j);
    if (res == gbdt_config_->cegb_config.penalty_feature_lazy.end())
      continue;
    double penalty = res->second;
    for (int i = 0; i < train_data_->num_data(); ++i) {
      if (lazy_feature_used[train_data_->num_data() * j + i])
        cost += penalty;
    }
  }

  cost /= train_data_->num_data();
#endif

  if (gbdt_config_->cegb_config.independent_branches) {
    for (int i_feature : iter_features_used)
      coupled_feature_used[i_feature] = true;
  }

  Log::Debug("CEGB: Training finished for iter %d", iter_);
  return res;
}

void CEGB::ResetFeatureTracking() {
  lazy_feature_used.clear();
  coupled_feature_used.clear();

  if (train_data_ != nullptr) {
    lazy_feature_used.resize(train_data_->num_total_features() * train_data_->num_data());
    coupled_feature_used.resize(train_data_->num_total_features());
  }
}

void CEGB::InitTreeLearner(const BoostingConfig *config) {
  if (config->device_type != std::string("cpu"))
    Log::Fatal("CEGB currently only supports CPU tree learner, '%s' is unsupported.", config->device_type);
  if (config->tree_learner_type != std::string("serial"))
    Log::Fatal("CEGB currently only supports serial tree learner, '%s' is unsupported.", config->tree_learner_type);

  if (tree_learner_ != nullptr)
    return;

  tree_learner_ = std::unique_ptr<TreeLearner>((TreeLearner *)new CEGBTreeLearner(
      &config->tree_config, &config->cegb_config, lazy_feature_used, coupled_feature_used, iter_features_used, bag_data_indices_));
}

static double find_cost_or_zero(const std::map<int, double> &m, int feature) {
  auto res = m.find(feature);
  if (res == m.end())
    return 0;
  else
    return res->second;
}

inline void CEGB::InitPredict(int num_iteration) {
  GBDT::InitPredict(num_iteration);

  if (num_tree_per_iteration_ > 1)
    Log::Fatal("CEGB::InitPredict not implemented for num_tree_per_iteration_ > 1.");

  if (models_costinfo.size() == num_iteration_for_pred_)
    return;

  models_costinfo.clear();
  models_costinfo.resize(num_iteration_for_pred_);

  for (int i = 0; i < num_iteration_for_pred_; ++i)
    models_costinfo[i].resize(models_[i]->num_leaves());

  for (int i = 0; i < num_iteration_for_pred_; ++i) {
    auto &model = models_[i];
    int n_leafs = model->num_leaves();

    std::vector<std::vector<int>> paths = model->GetPathToLeafs();

    for (int i_leaf = 0; i_leaf < n_leafs; ++i_leaf) {
      detail::CEGB_CostInfo &cinfo = models_costinfo[i][i_leaf];

      cinfo.n_splits = 0;
      cinfo.features.clear();

      std::vector<int> &path = paths[i_leaf];

      for (int i_split_node : path) {
        cinfo.features.insert(model->split_feature(i_split_node));
        cinfo.n_splits++;
      }
    }
  }
}

void CEGB::InitPredict(int num_iteration, const BoostingConfig *config) {
  CEGB::InitPredict(num_iteration);

  if (config != nullptr)
    predict_penalty_split = config->cegb_config.penalty_split;
}

void CEGB::PredictMulti(const double *features, double *output_raw, double *output, double *leaf, double *cost) const {

  std::set<int> features_used;
  double i_cost = 0;
  double i_pred = 0;

  if (num_tree_per_iteration_ > 1)
    Log::Fatal("CEGB::PredictMulti not implemented for num_tree_per_iteration_ > 1.");

  for (int i = 0; i < num_iteration_for_pred_; ++i) {
    auto &model = models_[i];

    int i_leaf = model->PredictLeafIndex(features);

    const detail::CEGB_CostInfo &cinfo = models_costinfo[i][i_leaf];

    // feature penalty
    for (int i_feature : cinfo.features) {
      if (features_used.find(i_feature) != features_used.end())
        continue;

      i_cost += find_cost_or_zero(predict_penalty_feature_lazy, i_feature);
      i_cost += find_cost_or_zero(predict_penalty_feature_coupled, i_feature);
      features_used.insert(i_feature);
    }

    // split penalty
    i_cost += predict_penalty_split * cinfo.n_splits;

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

void CEGB::MyAddPredictionToScore(const Tree *tree, const data_size_t *data_indices, data_size_t data_cnt, int cur_tree_id) {

  if (data_indices == nullptr || data_cnt == 0) {
    tree->AddPredictionToScoreGetFeatures(train_data_, train_data_->num_data(),
                                          (double *)train_score_updater_->score() + cur_tree_id * train_score_updater_->num_data(),
                                          lazy_feature_used, train_data_->num_data());

  } else {
    tree->AddPredictionToScoreGetFeatures(train_data_, data_indices, data_cnt,
                                          (double *)train_score_updater_->score() + cur_tree_id * train_score_updater_->num_data(),
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
    MyAddPredictionToScore(tree, bag_data_indices_.data() + bag_data_cnt_, num_data_ - bag_data_cnt_, cur_tree_id);
  }
}

static void export_feature_penalties(const std::map<int, double> *m, std::stringstream &ss) {
  bool first = true;
  for (auto &i : *m) {
    if (!first)
      ss << ",";
    first = false;

    ss << i.first << ":" << i.second;
  }

  ss << "\n";
}

static void import_feature_penalties(std::map<int, double> &m, std::string v) {
  m.clear();

  if (v.size() == 0)
    return;

  std::vector<std::string> penalties = Common::Split(v.c_str(), ',');

  for (auto &penalty : penalties) {
    std::vector<std::string> tmp = Common::Split(penalty.c_str(), ':');

    if (tmp.size() == 0)
      continue;

    if (tmp.size() != 2) {
      Log::Warning("Unknown feature penalty: \"%s\"", penalty.c_str());
      continue;
    }

    std::pair<int, double> i_penalty;
    if (!Common::AtoiAndCheck(tmp[0].c_str(), &i_penalty.first)) {
      Log::Warning("Feature value should be of type int, got \"%s\"", tmp[0].c_str());
      continue;
    }
    if (!Common::AtofAndCheck(tmp[1].c_str(), &i_penalty.second)) {
      Log::Warning("Feature penalty should be of type double, got \"%s\"", tmp[1].c_str());
      continue;
    }

    if (std::isinf(i_penalty.second) || std::isnan(i_penalty.second) || std::signbit(i_penalty.second)) {
      Log::Warning("invalid i_penalty.second, ignoring!", i_penalty.second);
      continue;
    }

    if (i_penalty.first < 0) {
      Log::Warning("invalid i_penalty.first, ignoring!", i_penalty.first);
      continue;
    }

    if (i_penalty.second > 0.0f)
      m.insert(i_penalty);
  }
}

std::string CEGB::SaveModelToString(int num_iteration) const {
  std::stringstream ss;
  ss << GBDT::SaveModelToString(num_iteration);

  double tradeoff = 0.0f;
  bool independent_branches = false;
  double penalty_split = 0.0f;
  const std::map<int, double> *penalty_feature_lazy = nullptr;
  const std::map<int, double> *penalty_feature_coupled = nullptr;

  if (gbdt_config_ != nullptr) {
    penalty_split = gbdt_config_->cegb_config.penalty_split;
    penalty_feature_lazy = &gbdt_config_->cegb_config.penalty_feature_lazy;
    penalty_feature_coupled = &gbdt_config_->cegb_config.penalty_feature_coupled;
    tradeoff = gbdt_config_->cegb_config.tradeoff;
    independent_branches = gbdt_config_->cegb_config.independent_branches;
  } else {
    penalty_split = predict_penalty_split;
    penalty_feature_lazy = &predict_penalty_feature_lazy;
    penalty_feature_coupled = &predict_penalty_feature_coupled;
    tradeoff = predict_tradeoff;
    independent_branches = predict_independent_branches;
  }

  ss << std::setprecision(std::numeric_limits<double>::digits10 + 2);
  ss << "cegb_penalty_split=" << penalty_split << "\n";
  ss << "cegb_tradeoff=" << tradeoff << "\n";

  if (independent_branches)
    ss << "cegb_independent_branches=true\n";
  else
    ss << "cegb_independent_branches=false\n";

  ss << "cegb_penalty_feature_lazy=";
  export_feature_penalties(penalty_feature_lazy, ss);

  ss << "cegb_penalty_feature_coupled=";
  export_feature_penalties(penalty_feature_coupled, ss);

  return ss.str();
}

bool CEGB::LoadModelFromString(const std::string &model_str) {
  allow_train = false;
  bool ret = GBDT::LoadModelFromString(model_str);
  if (!ret)
    return ret;

  std::vector<std::string> lines = Common::Split(model_str.c_str(), '\n');

  auto line = Common::FindFromLines(lines, "cegb_penalty_split=");
  if (line.size() > 0) {
    Common::Atof(Common::Split(line.c_str(), '=')[1].c_str(), &predict_penalty_split);
  } else {
    Log::Warning("Model file doesn't specify cegb_penalty_split, assuming 0.");
    predict_penalty_split = 0.0f;
  }

  line = Common::FindFromLines(lines, "cegb_tradeoff=");
  if (line.size() > 0) {
    Common::Atof(Common::Split(line.c_str(), '=')[1].c_str(), &predict_tradeoff);
  } else {
    Log::Warning("Model file doesn't specify cegb_tradeoff, assuming 0.");
    predict_penalty_split = 0.0f;
  }

  line = Common::FindFromLines(lines, "cegb_independent_branches=");
  if (line.size() > 0) {
    auto value = Common::Split(line.c_str(), '=')[1];
    std::transform(value.begin(), value.end(), value.begin(), Common::tolower);
    if (value == std::string("false") || value == std::string("-") || value == std::string("0")) {
      predict_independent_branches = false;
    } else if (value == std::string("true") || value == std::string("+") || value == std::string("1")) {
      predict_independent_branches = true;
    } else {
      Log::Warning("Model file specifies invalid cegb_independent_branches (%s), assuming true.", value);
      predict_independent_branches = true;
    }
  } else {
    Log::Warning("Model file doesn't specify cegb_independent_branches, assuming true.");
    predict_independent_branches = true;
  }

  line = Common::FindFromLines(lines, "cegb_penalty_feature_lazy=");
  if (line.size() > 0) {
    import_feature_penalties(predict_penalty_feature_lazy, Common::Split(line.c_str(), '=')[1]);
  } else {
    Log::Warning("Model file doesn't specify cegb_penalty_feature_lazy, assuming none.");
    predict_independent_branches = true;
  }

  line = Common::FindFromLines(lines, "cegb_penalty_feature_coupled=");
  if (line.size() > 0) {
    import_feature_penalties(predict_penalty_feature_coupled, Common::Split(line.c_str(), '=')[1]);
  } else {
    Log::Warning("Model file doesn't specify cegb_penalty_feature_coupled, assuming none.");
    predict_independent_branches = true;
  }

  return true;
}

} // namespace LightGBM
