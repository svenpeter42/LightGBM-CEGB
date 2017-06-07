#ifndef LIGHTGBM_BOOSTING_CEGB_H_
#define LIGHTGBM_BOOSTING_CEGB_H_

#include <LightGBM/boosting.h>
#include <LightGBM/tree_learner.h>

// FIXME:
#include "../treelearner/cegb_tree_learner.h"
#include "gbdt.h"
#include "score_updater.hpp"

#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace LightGBM {

namespace detail {
struct CEGB_CostInfo {
  size_t n_splits;
  std::set<int> features;
};
} // namespace detail

/*!
* \brief CEGB algorithm implementation. including Training, prediction.
*/
class CEGB : public GBDT {
public:
  /*!
  * \brief Constructor
  */
  CEGB() : GBDT(), allow_train(true), cost(0.0f) {}
  /*!
  * \brief Destructor
  */
  ~CEGB() {}
  /*!
  * \brief Initialization logic
  * \param config Config for boosting
  * \param train_data Training data
  * \param objective_function Training objective function
  * \param training_metrics Training metrics
  * \param output_model_filename Filename of output model
  */
  void Init(const BoostingConfig *config, const Dataset *train_data,
            const ObjectiveFunction *objective_function,
            const std::vector<const Metric *> &training_metrics) override;
  void ResetTrainingData(
      const BoostingConfig *config, const Dataset *train_data,
      const ObjectiveFunction *objective_function,
      const std::vector<const Metric *> &training_metrics) override;

  /*!
  * \brief one training iteration
  */
  bool TrainOneIter(const score_t *gradient, const score_t *hessian,
                    bool is_eval) override;

  void RollbackOneIter();

  /*!
  * \brief Restore from a serialized string
  */
  bool LoadModelFromString(const std::string &model_str) override;

  void PredictMulti(const double *features, double *output_raw, double *output,
                    double *leaf, double *cost) const override;

  virtual const char *SubModelName() const override { return "cegb_tree"; }

  void UpdateScore(const Tree *tree, const int cur_tree_id) override;
  void UpdateScoreOutOfBag(const Tree *tree, const int cur_tree_id) override;

  inline void InitPredict(int num_iteration) override;

private:
  std::vector<bool> lazy_feature_used;
  std::vector<bool> coupled_feature_used;
  std::vector<int> iter_features_used;

  std::vector<std::vector<detail::CEGB_CostInfo>> models_costinfo;

  bool allow_train;

  double cost;

  void ResetFeatureTracking();
  void InitTreeLearner(const BoostingConfig *);

  void MyAddPredictionToScore(const Tree *tree, const data_size_t *data_indices,
                              data_size_t data_cnt, int cur_tree_id);
};

} // namespace LightGBM
#endif // LightGBM_BOOSTING_CEGB_H_
