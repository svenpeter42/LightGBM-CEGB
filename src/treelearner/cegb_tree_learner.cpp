#include "cegb_tree_learner.h"

#include <LightGBM/utils/array_args.h>

#include <algorithm>
#include <vector>
namespace LightGBM {

template <typename K, typename V>
static inline void insert_or_assign(std::map<K, V> &m, K k, V v) {
  auto result = m.find(k);

  if (result != m.end())
    m.erase(result);

  m.insert(std::make_pair(k, v));
}

inline int CEGBTreeLearner::GetRealDataIndex(int idx) {
  if (bag_data_indices.size() == 0)
    return idx;

  return bag_data_indices[idx];
}

// mostly taken from SerialTreeLearner; only changed to store SplitInfo for all
// features
void CEGBTreeLearner::FindBestThresholds() {
  std::vector<int8_t> is_feature_used(num_features_, 0);
#pragma omp parallel for schedule(static, 1024) if (num_features_ >= 2048)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    if (!is_feature_used_[feature_index])
      continue;
    if (parent_leaf_histogram_array_ != nullptr &&
        !parent_leaf_histogram_array_[feature_index].is_splittable()) {
      smaller_leaf_histogram_array_[feature_index].set_is_splittable(false);
      continue;
    }
    is_feature_used[feature_index] = 1;
  }

  bool use_subtract = true;
  if (parent_leaf_histogram_array_ == nullptr) {
    use_subtract = false;
  }
  ConstructHistograms(is_feature_used, use_subtract);

  std::vector<SplitInfo> smaller_all(num_features_);
  std::vector<SplitInfo> larger_all(num_features_);

  int leaf_smaller = smaller_leaf_splits_->LeafIndex();
  int leaf_larger = -1;

  if (larger_leaf_splits_ != nullptr)
    leaf_larger = larger_leaf_splits_->LeafIndex();

  if (need_lazy_features) {
    leaf_feature_penalty[leaf_smaller].resize(num_features_);

    if (leaf_larger >= 0)
      leaf_feature_penalty[leaf_larger].resize(num_features_);
  }

  OMP_INIT_EX();
// find splits
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_used[feature_index])
      continue;

    // const int tid = omp_get_thread_num();
    SplitInfo smaller_split;
    train_data_->FixHistogram(
        feature_index, smaller_leaf_splits_->sum_gradients(),
        smaller_leaf_splits_->sum_hessians(),
        smaller_leaf_splits_->num_data_in_leaf(),
        smaller_leaf_histogram_array_[feature_index].RawData());
    int real_fidx = train_data_->RealFeatureIndex(feature_index);

    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
        smaller_leaf_splits_->sum_gradients(),
        smaller_leaf_splits_->sum_hessians(),
        smaller_leaf_splits_->num_data_in_leaf(), &smaller_split);
    smaller_all[feature_index] = smaller_split;
    smaller_all[feature_index].feature = real_fidx;

    if (need_lazy_features)
      leaf_feature_penalty[leaf_smaller][feature_index] =
          CalculateOndemandCosts(real_fidx, leaf_smaller);

    // only has root leaf
    if (leaf_larger < 0)
      continue;

    if (need_lazy_features)
      leaf_feature_penalty[leaf_larger][feature_index] =
          CalculateOndemandCosts(real_fidx, leaf_larger);

    if (use_subtract)
      larger_leaf_histogram_array_[feature_index].Subtract(
          smaller_leaf_histogram_array_[feature_index]);
    else
      train_data_->FixHistogram(
          feature_index, larger_leaf_splits_->sum_gradients(),
          larger_leaf_splits_->sum_hessians(),
          larger_leaf_splits_->num_data_in_leaf(),
          larger_leaf_histogram_array_[feature_index].RawData());

    SplitInfo larger_split;
    // find best threshold for larger child
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(
        larger_leaf_splits_->sum_gradients(),
        larger_leaf_splits_->sum_hessians(),
        larger_leaf_splits_->num_data_in_leaf(), &larger_split);

    larger_all[feature_index] = larger_split;
    larger_all[feature_index].feature = real_fidx;
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  insert_or_assign(leaf_feature_splits, smaller_leaf_splits_->LeafIndex(),
                   smaller_all);

  if (leaf_larger >= 0)
    insert_or_assign(leaf_feature_splits, leaf_larger, larger_all);
}

double CEGBTreeLearner::CalculateOndemandCosts(int feature_index,
                                               int leaf_index) {
  if (!need_lazy_features)
    return 0.0f;

  double penalty = 0.0f;
  auto res = cegb_config->penalty_feature_lazy.find(feature_index);
  if (res != cegb_config->penalty_feature_lazy.end())
    penalty = res->second;

  if (penalty <= 0.0f)
    return 0.0f;

  double total = 0.0f;
  data_size_t cnt_leaf_data = 0;
  auto tmp_idx = data_partition_->GetIndexOnLeaf(leaf_index, &cnt_leaf_data);

  for (data_size_t i_input = 0; i_input < cnt_leaf_data; ++i_input) {
    int real_idx = GetRealDataIndex(tmp_idx[i_input]);
    if (lazy_features_used[train_data_->num_data() * feature_index + real_idx])
      continue;
    total += penalty;
  }

  return total;
}

void CEGBTreeLearner::FindBestSplitForLeaf(int leaf) {
  std::vector<double> gain(num_features_);

  for (int i_feature = 0; i_feature < num_features_; i_feature++) {
    double i_gain = leaf_feature_splits[leaf][i_feature].gain;
    double i_penalty_lazy = 0.0f;
    double i_penalty_coupled = 0.0f;

    if (!coupled_features_used[i_feature]) {
      auto res = cegb_config->penalty_feature_coupled.find(i_feature);
      if (res != cegb_config->penalty_feature_coupled.end())
        i_penalty_coupled = res->second;
    }
    if (need_lazy_features)
      i_penalty_lazy = leaf_feature_penalty[leaf][i_feature];

    // FIXME:
    // a "good" split can have positive i_gain but negative i_gain - tradeoff *
    // i_penalty
    // a split with negative i_gain is always bad though.
    // pathological case would be a gain <= 0.0f split with no penalty and
    // another split with gain just above 0.0f but a penalty just high enough to
    // make it worse than the first split.
    // we don't change SplitInfo.gain though so that SerialTreeLearner::Train
    // should catch this as long we we make sure to not select this split here
    // as
    // the best for this leaf
    if (i_gain < 0.0f)
      gain[i_feature] = -INFINITY;
    else
      gain[i_feature] =
          i_gain - cegb_config->tradeoff * (i_penalty_lazy + i_penalty_coupled);
  }

  auto best_idx = ArrayArgs<double>::ArgMax(gain);
  best_split_per_leaf_[leaf] = leaf_feature_splits[leaf][best_idx];
  best_split_per_leaf_[leaf].gain = gain[best_idx];
}

void CEGBTreeLearner::FindBestSplitsForLeaves() {
  // always need to update best split for left and right leaf
  FindBestSplitForLeaf(smaller_leaf_splits_->LeafIndex());

  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0)
    FindBestSplitForLeaf(larger_leaf_splits_->LeafIndex());

  // need to update all leaves if branches are coupled and a new feature was
  // used
  if (independent_branches == true || used_new_coupled_feature == false)
    return;

  OMP_INIT_EX();
#pragma omp parallel for schedule(static)
  for (int leaf = 0; leaf < (int)leaf_feature_splits.size(); ++leaf) {
    OMP_LOOP_EX_BEGIN();
    if (leaf == smaller_leaf_splits_->LeafIndex())
      continue;
    if (leaf == larger_leaf_splits_->LeafIndex())
      continue;

    FindBestSplitForLeaf(leaf);
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();
}

void CEGBTreeLearner::Split(Tree *tree, int best_leaf, int *left_leaf,
                            int *right_leaf) {
  const SplitInfo &best_split_info = best_split_per_leaf_[best_leaf];
  const int inner_feature_index =
      train_data_->InnerFeatureIndex(best_split_info.feature);

  data_size_t cnt_leaf_data = 0;
  auto tmp_idx = data_partition_->GetIndexOnLeaf(best_leaf, &cnt_leaf_data);
  for (data_size_t i_input = 0; i_input < cnt_leaf_data; ++i_input) {
    int real_idx = GetRealDataIndex(tmp_idx[i_input]);
    lazy_features_used[train_data_->num_data() * inner_feature_index +
                       real_idx] = true;
  }

  if (independent_branches == true) {
    used_new_coupled_feature = false;
    if (!coupled_features_used[inner_feature_index])
      new_features_used.push_back(inner_feature_index);
  } else {
    if (coupled_features_used[inner_feature_index]) {
      used_new_coupled_feature = false;
    } else {
      coupled_features_used[inner_feature_index] = true;
      used_new_coupled_feature = true;
    }
  }

  SerialTreeLearner::Split(tree, best_leaf, left_leaf, right_leaf);
  leaf_feature_splits.erase(best_leaf);
}

} // namespace LightGBM
