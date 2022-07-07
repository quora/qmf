/*
 * Copyright 2016 Quora, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <fstream>

#include <qmf/bpr/BPREngine.h>
#include <qmf/DatasetReader.h>
#include <qmf/metrics/MetricsEngine.h>
#include <qmf/utils/Util.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

// model arguments
DEFINE_uint64(nepochs, 10, "number of epochs for SGD");
DEFINE_uint64(nfactors, 30, "dimension of learned factors");
DEFINE_double(init_learning_rate, 0.05, "initial learning rate");
DEFINE_double(bias_lambda, 1.0, "regularization on biases");
DEFINE_double(user_lambda, 0.025, "regularization on user factors");
DEFINE_double(item_lambda, 0.0025, "regularization on item factors");
DEFINE_double(decay_rate, 0.9, "decay rate on learning rate");
DEFINE_bool(use_biases, false, "use bias term");
DEFINE_double(init_distribution_bound, 0.01, "init distirbution bound");
DEFINE_uint64(num_negative_samples, 3, "number of negative items to sample for each positive item");
DEFINE_uint64(num_hogwild_threads, 1, "number of parallel threads for hogwild");
DEFINE_bool(shuffle_training_set, true, "shuffle training set after each epoch");

// settings
DEFINE_uint64(eval_num_neg, 3, "number of negatives generated per positive in evaluation");
DEFINE_int32(eval_seed, 42, "random seed for generating evaluation set and test users");
DEFINE_uint64(nthreads, 16, "number of threads for parallel execution");

// datasets
DEFINE_string(train_dataset, "", "filename of training dataset");
DEFINE_string(test_dataset, "", "filename of test dataset");

// metrics
DEFINE_string(test_avg_metrics, "", "comma-separated list of test metrics (averaged per-user)");
DEFINE_uint64(num_test_users, 0, "# users to use for computing test avg metrics (0 = all users)");
DEFINE_bool(test_always, false, "whether to compute test avg metrics after "
                                "each epoch (if false, only computes at the "
                                "end)");

// model output
DEFINE_string(user_factors, "", "filename of user factors");
DEFINE_string(item_factors, "", "filename of item factors");

int main(int argc, char** argv) {
  gflags::SetUsageMessage("bpr");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  // make glog to log to stderr
  FLAGS_logtostderr = 1;

  if (FLAGS_user_factors.empty() || FLAGS_item_factors.empty()) {
    LOG(WARNING)
      << "warning: missing model output filenames! (use options --{user,item}_factors)";
  }

  qmf::BPRConfig config{FLAGS_nepochs,
                        FLAGS_nfactors,
                        FLAGS_init_learning_rate,
                        FLAGS_bias_lambda,
                        FLAGS_user_lambda,
                        FLAGS_item_lambda,
                        FLAGS_decay_rate,
                        FLAGS_use_biases,
                        FLAGS_init_distribution_bound,
                        FLAGS_num_negative_samples,
                        FLAGS_num_hogwild_threads,
                        FLAGS_shuffle_training_set};

  qmf::MetricsConfig metricsConfig{
    FLAGS_num_test_users, FLAGS_test_always, FLAGS_eval_seed};
  const auto metricsEngine =
    std::make_unique<qmf::MetricsEngine>(metricsConfig);

  // test average metrics
  const auto metrics = qmf::split(FLAGS_test_avg_metrics, ',');
  if (!metrics.empty()) {
    for (const auto& metric : metrics) {
      CHECK(metricsEngine->addTestAvgMetric(metric)) << "metric " << metric
                                                     << " is not available";
    }
  }

  qmf::BPREngine engine(
    config, metricsEngine, FLAGS_eval_num_neg, FLAGS_eval_seed, FLAGS_nthreads);

  LOG(INFO) << "loading training data";
  qmf::DatasetReader trainReader(FLAGS_train_dataset);
  engine.init(trainReader.readAll());

  if (!FLAGS_test_dataset.empty()) {
    LOG(INFO) << "loading test data";
    qmf::DatasetReader testReader(FLAGS_test_dataset);
    engine.initTest(testReader.readAll());
  }

  LOG(INFO) << "training";
  engine.optimize();

  if (!FLAGS_user_factors.empty() && !FLAGS_item_factors.empty()) {
    LOG(INFO) << "saving model output";
    engine.saveUserFactors(FLAGS_user_factors);
    engine.saveItemFactors(FLAGS_item_factors);
  }

  return 0;
}
