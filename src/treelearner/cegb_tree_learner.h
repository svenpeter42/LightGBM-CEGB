#ifndef LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/array_args.h>

#include <LightGBM/tree_learner.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>

#include "feature_histogram.hpp"
#include "split_info.hpp"
#include "data_partition.hpp"
#include "leaf_splits.hpp"
#include "serial_tree_learner.h"

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <memory>


namespace LightGBM {

class CEGBTreeLearner: public SerialTreeLearner {
public:
  CEGBTreeLearner(const TreeConfig* tree_config, const CEGBConfig* cegb_config_, std::vector<bool> &lazy_features_used_, std::vector<bool> &coupled_features_used_) : SerialTreeLearner(tree_config), cegb_config(cegb_config_), lazy_features_used(lazy_features_used_), coupled_features_used(coupled_features_used_)
  {
  }

  ~CEGBTreeLearner()  { }

private:
	const CEGBConfig* cegb_config;
	std::vector<bool> &lazy_features_used;
	std::vector<bool> &coupled_features_used;
};

}  // namespace LightGBM

#endif   // LIGHTGBM_TREELEARNER_CEGB_TREE_LEARNER_H_