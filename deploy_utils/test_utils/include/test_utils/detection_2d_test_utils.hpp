#pragma once

#include "deploy_core/base_detection.hpp"

namespace easy_deploy {

void test_detection_2d_algorithm_correctness(const std::shared_ptr<BaseDetectionModel> &model,
                                             const std::string &test_image_path,
                                             float              conf_threshold,
                                             size_t             expected_obj_num,
                                             const std::string &test_visual_result_save_path = "");

void test_detection_2d_algorithm_async_correctness(
    const std::shared_ptr<BaseDetectionModel> &model,
    const std::string                         &test_image_path,
    float                                      conf_threshold,
    size_t                                     expected_obj_num,
    const std::string                         &test_visual_result_save_path = "");

} // namespace easy_deploy
