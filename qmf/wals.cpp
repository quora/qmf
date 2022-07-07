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

#include <qmf/wals/WALSEngine.h>
#include <qmf/DatasetReader.h>
#include <qmf/metrics/MetricsEngine.h>
#include <qmf/utils/Util.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

// model arguments
DEFINE_uint64(nepochs, 10, "number of epochs for ALS");
DEFINE_uint64(nfactors, 30, "dimension of learned factors");
DEFINE_double(regularization_lambda, 0.05, "regularization param");
DEFINE_double(confidence_weight, 40, "confidence weight");
DEFINE_double(init_distribution_bound, 0.01, "init distirbution bound");

// settings
DEFINE_int32(nthreads, 16, "number of threads for parallel execution");

// datasets
DEFINE_string(train_dataset, "", "filename of training dataset");
DEFINE_string(test_dataset, "", "filename of test dataset");

// metrics
DEFINE_string(test_avg_metrics, "", "comma-separated list of test metrics (averaged per-user)");
DEFINE_int32(eval_seed, 42, "random seed for picking test users");
DEFINE_uint64(num_test_users, 0, "# users to use for computing test avg metrics (0 = all users)");
DEFINE_bool(test_always, false, "whether to compute test avg metrics after "
                                "each epoch (if false, only computes at the "
                                "end)");

// model output
DEFINE_string(user_factors, "", "filename of user factors");
DEFINE_string(item_factors, "", "filename of item factors");

int main(int argc, char** argv) {
  gflags::SetUsageMessage("wals");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  // make glog to log to stderr
  FLAGS_logtostderr = 1;

  if (FLAGS_user_factors.empty() || FLAGS_item_factors.empty()) {
    LOG(WARNING)
      << "warning: missing model output filenames! (use options --{user,item}_factors)";
  }

  qmf::WALSConfig config{FLAGS_nepochs,
                         FLAGS_nfactors,
                         FLAGS_regularization_lambda,
                         FLAGS_confidence_weight,
                         FLAGS_init_distribution_bound};

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

  qmf::WALSEngine engine(config, metricsEngine, FLAGS_nthreads);

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
