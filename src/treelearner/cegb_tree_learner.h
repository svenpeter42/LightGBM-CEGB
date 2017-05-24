#ifndef LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_

#include <LightGBM/utils/array_args.h>
#include <LightGBM/utils/random.h>

#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>
#include <LightGBM/tree_learner.h>

#include "data_partition.hpp"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"
#include "serial_tree_learner.h"
#include "split_info.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

namespace LightGBM {

class CEGBTreeLearner : public SerialTreeLearner {
public:
  CEGBTreeLearner(const TreeConfig *tree_config, const CEGBConfig *cegb_config_,
                  std::vector<bool> &lazy_features_used_,
                  std::vector<bool> &coupled_features_used_,
                  std::vector<int> &new_features_used_)
      : SerialTreeLearner(tree_config), cegb_config(cegb_config_),
        lazy_features_used(lazy_features_used_),
        coupled_features_used(coupled_features_used_),
        new_features_used(new_features_used_) {
    independent_branches = false;

    // GreedyMiser mode -> treat branches as independent even when using coupled
    // feature penalties
    if (cegb_config->independent_branches == true)
      independent_branches = true;

    // no coupled feature penalties -> all branches are independent since they
    // share no training instances
    if (cegb_config->penalty_feature_coupled.size() == 0)
      independent_branches = true;

    // no prediction cost penalty -> all branches are independent since they
    // share no training instances
    if (cegb_config->tradeoff == 0)
      independent_branches = true;

    used_new_coupled_feature = true;
  }

  ~CEGBTreeLearner() {}

  void FindBestSplitsForLeaves();
  void FindBestThresholds();
  void Split(Tree *, int, int *, int *);

private:
  const CEGBConfig *cegb_config;
  std::vector<bool> &lazy_features_used;
  std::vector<bool> &coupled_features_used;
  std::vector<int> &new_features_used;

  bool independent_branches;
  bool used_new_coupled_feature;

  /*! \brief stores best thresholds for all feature for all leaves */
  std::map<int, std::vector<SplitInfo>> leaf_feature_splits;
  std::map<int, std::vector<double>> leaf_feature_penalty;

  double CalculateOndemandCosts(int, int);
  void FindBestSplitForLeaf(int);
};

} // namespace LightGBM

#endif // LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_