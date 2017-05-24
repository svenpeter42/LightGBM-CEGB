#include "cegb_tree_learner.h"

#include <LightGBM/utils/array_args.h>

#include <algorithm>
#include <vector>
namespace LightGBM {

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
  #ifdef TIMETAG
  auto start_time = std::chrono::steady_clock::now();
  #endif
  // std::vector<SplitInfo> smaller_best(num_threads_);
  // std::vector<SplitInfo> larger_best(num_threads_);

  std::vector<SplitInfo> smaller_all(num_features_);
  std::vector<SplitInfo> larger_all(num_features_);

  int leaf = smaller_leaf_splits_->LeafIndex();
  leaf_feature_penalty[leaf].resize(num_features_);

  leaf = larger_leaf_splits_->LeafIndex();
  if (larger_leaf_splits_ != nullptr && leaf >= 0)
    leaf_feature_penalty[leaf].resize(num_features_);

  OMP_INIT_EX();
  // find splits
  #pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_used[feature_index]) {
      continue;
    }
    // const int tid = omp_get_thread_num();
    SplitInfo smaller_split;
    train_data_->FixHistogram(
        feature_index, smaller_leaf_splits_->sum_gradients(),
        smaller_leaf_splits_->sum_hessians(),
        smaller_leaf_splits_->num_data_in_leaf(),
        smaller_leaf_histogram_array_[feature_index].RawData());

    smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
        smaller_leaf_splits_->sum_gradients(),
        smaller_leaf_splits_->sum_hessians(),
        smaller_leaf_splits_->num_data_in_leaf(), &smaller_split);
    smaller_all[feature_index] = smaller_split;
    smaller_all[feature_index].feature =
        train_data_->RealFeatureIndex(feature_index);

    leaf = smaller_leaf_splits_->LeafIndex();
    leaf_feature_penalty[leaf][feature_index] = CalculateOndemandCosts(
        train_data_->RealFeatureIndex(feature_index), leaf);

    // if (smaller_split.gain > smaller_best[tid].gain) {
    //  smaller_best[tid] = smaller_split;
    //  smaller_best[tid].feature =
    //  train_data_->RealFeatureIndex(feature_index);
    //}

    // only has root leaf
    if (larger_leaf_splits_ == nullptr ||
        larger_leaf_splits_->LeafIndex() < 0) {
      continue;
    }

    leaf = larger_leaf_splits_->LeafIndex();
    leaf_feature_penalty[leaf][feature_index] = CalculateOndemandCosts(
        train_data_->RealFeatureIndex(feature_index), leaf);

    if (use_subtract) {
      larger_leaf_histogram_array_[feature_index].Subtract(
          smaller_leaf_histogram_array_[feature_index]);
    } else {
      train_data_->FixHistogram(
          feature_index, larger_leaf_splits_->sum_gradients(),
          larger_leaf_splits_->sum_hessians(),
          larger_leaf_splits_->num_data_in_leaf(),
          larger_leaf_histogram_array_[feature_index].RawData());
    }
    SplitInfo larger_split;
    // find best threshold for larger child
    larger_leaf_histogram_array_[feature_index].FindBestThreshold(
        larger_leaf_splits_->sum_gradients(),
        larger_leaf_splits_->sum_hessians(),
        larger_leaf_splits_->num_data_in_leaf(), &larger_split);

    larger_all[feature_index] = larger_split;
    larger_all[feature_index].feature =
        train_data_->RealFeatureIndex(feature_index);
    // if (larger_split.gain > larger_best[tid].gain) {
    //  larger_best[tid] = larger_split;
    //  larger_best[tid].feature = train_data_->RealFeatureIndex(feature_index);
    // }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  leaf = smaller_leaf_splits_->LeafIndex();
  leaf_feature_splits[leaf] = smaller_all;

  if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0) {
    leaf = larger_leaf_splits_->LeafIndex();
    leaf_feature_splits[leaf] = larger_all;
  }

  // auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_best);
  // int leaf = smaller_leaf_splits_->LeafIndex();
  // best_split_per_leaf_[leaf] = smaller_best[smaller_best_idx]; 

  // if (larger_leaf_splits_ != nullptr && larger_leaf_splits_->LeafIndex() >= 0)
  // {
  //  leaf = larger_leaf_splits_->LeafIndex();
  //  auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_best);
  //  best_split_per_leaf_[leaf] = larger_best[larger_best_idx];
  //}
  #ifdef TIMETAG
  find_split_time += std::chrono::steady_clock::now() - start_time;
  #endif
}

double CEGBTreeLearner::CalculateOndemandCosts(int feature_index,
                                               int leaf_index) {
  double penalty = cegb_config->penalty_feature_lazy.at(feature_index);
  if (penalty <= 0.0f)
    return 0.0f;

  double total = 0.0f;
  data_size_t cnt_leaf_data = 0;
  auto tmp_idx = data_partition_->GetIndexOnLeaf(leaf_index, &cnt_leaf_data);

  for (data_size_t i_input = 0; i_input < cnt_leaf_data; ++i_input) {
    if (coupled_features_used[train_data_->num_data() * feature_index +
                              tmp_idx[i_input]])
      continue;
    total += penalty;
  }

  return total;
}

void CEGBTreeLearner::FindBestSplitForLeaf(int leaf) {
  std::vector<double> gain;

  gain.resize(num_features_);
  for (int i_feature = 0; i_feature < num_features_; i_feature++) {
    double i_gain = leaf_feature_splits[leaf][i_feature].gain;
    double i_penalty_lazy = leaf_feature_penalty[leaf][i_feature];
    double i_penalty_coupled = 0.0f;
    if (!coupled_features_used[i_feature])
      i_penalty_coupled = cegb_config->penalty_feature_coupled.at(i_feature);

    // FIXME:
    // a "good" split can have positive i_gain but negative i_gain - tradeoff *
    // i_penalty
    // a split with negative i_gain is always bad though
    if (i_gain < 0.0f)
      gain[i_feature] = -INFINITY;
    else
      gain[i_feature] =
          i_gain - cegb_config->tradeoff * (i_penalty_lazy + i_penalty_coupled);
  }

  auto best_idx = ArrayArgs<double>::ArgMax(gain);
  best_split_per_leaf_[leaf] = leaf_feature_splits[leaf][best_idx];
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
  for (int leaf = 0; leaf < (int)best_split_per_leaf_.size(); ++leaf) {
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
  for (data_size_t i_input = 0; i_input < cnt_leaf_data; ++i_input)
    coupled_features_used[train_data_->num_data() * inner_feature_index +
                          tmp_idx[i_input]] = true;

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
}

} // namespace LightGBM
