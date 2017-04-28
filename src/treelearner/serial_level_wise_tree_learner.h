#ifndef LIGHTGBM_TREELEARNER_SERIAL_LEVEL_WISE_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_SERIAL_LEVEL_WISE_TREE_LEARNER_H_

#include <LightGBM/utils/random.h>
#include <LightGBM/utils/array_args.h>

#include <LightGBM/tree_learner.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree.h>

#include "feature_histogram.hpp"
#include "split_info.hpp"
#include "data_partition.hpp"
#include "leaf_splits.hpp"

#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <memory>


namespace LightGBM {

/*!
* \brief Efficient Level Wise tree learner
*/
class SerialLevelWiseTreeLearner: public TreeLearner {
public:
  explicit SerialLevelWiseTreeLearner(const TreeConfig* tree_config);

  ~SerialLevelWiseTreeLearner();

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ResetTrainingData(const Dataset* train_data) override;

  void ResetConfig(const TreeConfig* tree_config) override;

  Tree* Train(const score_t* gradients, const score_t *hessians, bool is_constant_hessian) override;

  Tree* FitByExistingTree(const Tree* old_tree, const score_t* gradients, const score_t* hessians) const override;

  void SetBaggingData(const data_size_t* used_indices, data_size_t num_data) override {
    data_partition_->SetUsedDataIndices(used_indices, num_data);
  }

  void AddPredictionToScore(const Tree* tree, double* out_score) const override {
    if (tree->num_leaves() <= 1) { return; }
    CHECK(tree->num_leaves() <= data_partition_->num_leaves());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < tree->num_leaves(); ++i) {
      double output = static_cast<double>(tree->LeafOutput(i));
      data_size_t cnt_leaf_data = 0;
      auto tmp_idx = data_partition_->GetIndexOnLeaf(i, &cnt_leaf_data);
      for (data_size_t j = 0; j < cnt_leaf_data; ++j) {
        out_score[tmp_idx[j]] += output;
      }
    }
  }

protected:

  virtual void BeforeTrain();

  virtual bool BeforeFindBestSplit(const Tree* tree, int left_leaf, int right_leaf);

  virtual void ConstructHistograms(const std::vector<int8_t>& is_feature_used, bool use_subtract);

  virtual void FindBestThresholds();

  virtual void FindBestSplitsForLeaves();

  virtual void Split(Tree* tree, int best_leaf, int* left_leaf, int* right_leaf);

  inline virtual data_size_t GetGlobalDataCountInLeaf(int leaf_idx) const;
  /*! \brief number of data */
  data_size_t num_data_;
  /*! \brief number of features */
  int num_features_;
  /*! \brief training data */
  const Dataset* train_data_;
  /*! \brief gradients of current iteration */
  const score_t* gradients_;
  /*! \brief hessians of current iteration */
  const score_t* hessians_;

  std::vector<int> row_idx_2_leaves_idx_;
  /*! \brief training data partition on leaves */
  std::unique_ptr<DataPartition> data_partition_;
  /*! \brief used for generate used features */
  Random random_;
  /*! \brief used for sub feature training, is_feature_used_[i] = false means don't used feature i */
  std::vector<int8_t> is_feature_used_;
  /*! \brief pointer to histograms array of parent of current leaves */
  FeatureHistogram* parent_leaf_histogram_array_;
  /*! \brief pointer to histograms array of smaller leaf */
  FeatureHistogram* smaller_leaf_histogram_array_;
  /*! \brief pointer to histograms array of larger leaf */
  FeatureHistogram* larger_leaf_histogram_array_;

  std::vector<int> cur_working_leaves_;
  /*! \brief store best split points for all leaves */
  std::vector<SplitInfo> best_split_per_leaf_;

  /*! \brief stores best thresholds for all feature for smaller leaf */
  std::unique_ptr<LeafSplits> smaller_leaf_splits_;
  /*! \brief stores best thresholds for all feature for larger leaf */
  std::unique_ptr<LeafSplits> larger_leaf_splits_;

  /*! \brief used to cache historical histogram to speed up*/
  HistogramPool histogram_pool_;
  /*! \brief config of tree learner*/
  const TreeConfig* tree_config_;
  int num_threads_;
  bool is_constant_hessian_;
};

inline data_size_t SerialLevelWiseTreeLearner::GetGlobalDataCountInLeaf(int leafIdx) const {
  if (leafIdx >= 0) {
    return data_partition_->leaf_count(leafIdx);
  } else {
    return 0;
  }
}

}  // namespace LightGBM
#endif   // LIGHTGBM_TREELEARNER_SERIAL_LEVEL_WISE_TREE_LEARNER_H_
