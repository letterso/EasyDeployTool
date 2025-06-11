#include "trt_core/trt_core.hpp"

namespace easy_deploy {

struct TrtInferCoreParams {
  std::string                                            model_path;
  std::unordered_map<std::string, std::vector<uint64_t>> input_blobs_shape;
  std::unordered_map<std::string, std::vector<uint64_t>> output_blobs_shape;
  int                                                    mem_buf_size;
};

class TrtInferCoreFactory : public BaseInferCoreFactory {
public:
  TrtInferCoreFactory(TrtInferCoreParams params) : params_(params)
  {}

  std::shared_ptr<BaseInferCore> Create() override
  {
    return CreateTrtInferCore(params_.model_path, params_.input_blobs_shape,
                              params_.output_blobs_shape, params_.mem_buf_size);
  }

private:
  TrtInferCoreParams params_;
};

std::shared_ptr<BaseInferCoreFactory> CreateTrtInferCoreFactory(
    std::string                                                   model_path,
    const std::unordered_map<std::string, std::vector<uint64_t>> &input_blobs_shape,
    const std::unordered_map<std::string, std::vector<uint64_t>> &output_blobs_shape,
    const int                                                     mem_buf_size)
{
  TrtInferCoreParams params;
  params.model_path         = model_path;
  params.input_blobs_shape  = input_blobs_shape;
  params.output_blobs_shape = output_blobs_shape;
  params.mem_buf_size       = mem_buf_size;

  return std::make_shared<TrtInferCoreFactory>(params);
}

} // namespace easy_deploy
