#pragma once

#include <functional>
#include <future>
#include <vector>

#include "common_utils/block_queue.hpp"

#include "deploy_core/common.hpp"

namespace easy_deploy {

/**
 * @brief Async Pipeline Block
 *
 * @tparam ParsingType
 */
template <typename ParsingType>
class AsyncPipelineBlock {
public:
  AsyncPipelineBlock() = default;
  AsyncPipelineBlock(const AsyncPipelineBlock &block)
      : func_(block.func_), block_name_(block.block_name_)
  {}

  AsyncPipelineBlock &operator=(const AsyncPipelineBlock &block)
  {
    func_       = block.func_;
    block_name_ = block.block_name_;
    return *this;
  }

  AsyncPipelineBlock(const std::function<bool(ParsingType)> &func) : func_(func)
  {}

  AsyncPipelineBlock(const std::function<bool(ParsingType)> &func, const std::string &block_name)
      : func_(func), block_name_(block_name)
  {}

  const std::string &GetName() const
  {
    return block_name_;
  }

  bool operator()(const ParsingType &pipeline_unit) const
  {
    return func_(pipeline_unit);
  }

private:
  std::function<bool(ParsingType)> func_;
  std::string                      block_name_;
};

/**
 * @brief Async Pipeline Context
 *
 * @tparam ParsingType
 */
template <typename ParsingType>
class AsyncPipelineContext {
  using Block_t   = AsyncPipelineBlock<ParsingType>;
  using Context_t = AsyncPipelineContext<ParsingType>;

public:
  AsyncPipelineContext() = default;

  AsyncPipelineContext(const Block_t &block) : blocks_({block})
  {}

  AsyncPipelineContext(const std::vector<Block_t> &block_vec)
  {
    for (const auto &block : block_vec)
    {
      blocks_.push_back(block);
    }
  }

  AsyncPipelineContext &operator=(const std::vector<Block_t> &block_vec)
  {
    for (const auto &block : block_vec)
    {
      blocks_.push_back(block);
    }
    return *this;
  }

  AsyncPipelineContext(const Context_t &context) : blocks_(context.blocks_)
  {}

  AsyncPipelineContext(const std::vector<Context_t> &context_vec)
  {
    for (const auto &context : context_vec)
    {
      for (const auto &block : context.blocks_)
      {
        blocks_.push_back(block);
      }
    }
  }

  AsyncPipelineContext &operator=(const std::vector<Context_t> &context_vec)
  {
    for (const auto &context : context_vec)
    {
      for (const auto &block : context.blocks_)
      {
        blocks_.push_back(block);
      }
    }
    return *this;
  }

  AsyncPipelineContext &operator=(const Context_t &context)
  {
    for (const auto &block : context.blocks_)
    {
      blocks_.push_back(block);
    }
    return *this;
  }

public:
  std::vector<Block_t> blocks_;
};

/**
 * @brief Async Pipeline Processing Instance
 *
 * @tparam ParsingType
 */
template <typename ParsingType>
class PipelineInstance {
  using Block_t    = AsyncPipelineBlock<ParsingType>;
  using Context_t  = AsyncPipelineContext<ParsingType>;
  using Callback_t = std::function<bool(const ParsingType &)>;

  // for inner processing
  struct _InnerPackage {
    ParsingType package;
    Callback_t  callback;
  };
  using InnerParsingType = std::shared_ptr<_InnerPackage>;
  using InnerBlock_t     = AsyncPipelineBlock<InnerParsingType>;
  using InnerContext_t   = AsyncPipelineContext<InnerParsingType>;

public:
  PipelineInstance() = default;

  PipelineInstance(const std::vector<Context_t> &block_list) : context_(block_list)
  {
    // initialize inner context
    std::vector<InnerBlock_t> inner_block_list;
    for (const auto &block : context_.blocks_)
    {
      auto         func = [&](InnerParsingType p) -> bool { return block(p->package); };
      InnerBlock_t inner_block(func, block.GetName());
      inner_block_list.push_back(inner_block);
    }
    inner_context_ = InnerContext_t(inner_block_list);
  }

  ~PipelineInstance()
  {
    ClosePipeline();
  }

  void Init(int bq_max_size = 100)
  {
    // 1. for `n` blocks, construct `n+1` block queues
    const auto blocks = inner_context_.blocks_;
    const int  n      = blocks.size();
    LOG_DEBUG("[AsyncPipelineInstance] Total {%d} Pipeline Blocks", n);
    for (int i = 0; i < n + 1; ++i)
    {
      block_queue_.emplace_back(std::make_shared<BlockQueue<InnerParsingType>>(bq_max_size));
    }
    pipeline_close_flag_.store(false);

    async_futures_.resize(n + 1);
    // 2. open `n` async threads to execute blocks
    for (int i = 0; i < n; ++i)
    {
      async_futures_[i] = std::async(&PipelineInstance::ThreadExcuteEntry, this, block_queue_[i],
                                     block_queue_[i + 1], blocks[i]);
    }
    // 3. open output threads to execute callback
    async_futures_[n] = std::async(&PipelineInstance::ThreadOutputEntry, this, block_queue_[n]);

    pipeline_initialized_.store(true);
  }

  void ClosePipeline()
  {
    if (pipeline_initialized_)
    {
      LOG_DEBUG("[AsyncPipelineInstance] Closing pipeline ...");
      for (const auto &bq : block_queue_)
      {
        bq->DisableAndClear();
      }
      LOG_DEBUG("[AsyncPipelineInstance] Disabled all block queue ...");
      pipeline_close_flag_.store(true);

      for (auto &future : async_futures_)
      {
        auto res = future.get();
      }
      LOG_DEBUG("[AsyncPipelineInstance] Join all block queue ...");
      block_queue_.clear();
      LOG_DEBUG("[AsyncPipelineInstance] Async pipeline is released successfully!!");
      pipeline_initialized_ = false;
      pipeline_close_flag_.store(true);
      pipeline_no_more_input_.store(true);
    }
  }

  void StopPipeline()
  {
    if (pipeline_initialized_)
    {
      pipeline_no_more_input_.store(true);
      block_queue_[0]->SetNoMoreInput();
    }
  }

  bool IsInitialized() const
  {
    return pipeline_initialized_;
  }

  const Context_t &GetContext() const
  {
    return context_;
  }

  void PushPipeline(const ParsingType &obj, const Callback_t &callback)
  {
    auto inner_pack      = std::make_shared<_InnerPackage>();
    inner_pack->package  = obj;
    inner_pack->callback = callback;

    block_queue_[0]->BlockPush(inner_pack);
  }

private:
  bool ThreadExcuteEntry(std::shared_ptr<BlockQueue<InnerParsingType>> bq_input,
                         std::shared_ptr<BlockQueue<InnerParsingType>> bq_output,
                         const InnerBlock_t                           &pipeline_block)
  {
    LOG_DEBUG("[AsyncPipelineInstance] {%s} thread start!", pipeline_block.GetName().c_str());
    while (!pipeline_close_flag_)
    {
      auto data = bq_input->Take();
      if (!data.has_value())
      {
        if (pipeline_no_more_input_)
        {
          LOG_DEBUG("[AsyncPipelineInstance] {%s} set no more output ...",
                    pipeline_block.GetName().c_str());
          bq_output->SetNoMoreInput();
          break;
        } else
        {
          continue;
        }
      }

      try
      {
        auto start = std::chrono::high_resolution_clock::now();
        pipeline_block(data.value());
        auto end = std::chrono::high_resolution_clock::now();
        LOG_DEBUG("[AsyncPipelineInstance] Block name: {%s}, cost(us): %ld",
                  pipeline_block.GetName().c_str(),
                  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
      } catch (const std::exception &e)
      {
        LOG_ERROR(
            "[AsyncPipelineInstance] {%s}, excute block function failed! Got exception : %s, Drop "
            "package.",
            pipeline_block.GetName().c_str(), e.what());
        continue;
      }

      bq_output->BlockPush(data.value());
    }
    LOG_DEBUG("[AsyncPipelineInstance] {%s} thread quit!", pipeline_block.GetName().c_str());
    return true;
  }

  bool ThreadOutputEntry(std::shared_ptr<BlockQueue<InnerParsingType>> bq_input)
  {
    LOG_DEBUG("[AsyncPipelineInstance] {Output} thread start!");
    while (!pipeline_close_flag_)
    {
      auto data = bq_input->Take();
      if (!data.has_value())
      {
        if (pipeline_no_more_input_)
        {
          LOG_DEBUG("[AsyncPipelineInstance] {Output} set no more output ...");
          break;
        } else
        {
          continue;
        }
      }
      const auto &inner_pack = data.value();
      if (inner_pack != nullptr && inner_pack->callback != nullptr)
      {
        inner_pack->callback(inner_pack->package);
      } else
      {
        LOG_WARN(
            "[AsyncPipelineInstance] {Output} package without valid callback will be dropped!!!");
      }
    }
    LOG_DEBUG("[AsyncPipelineInstance] {Output} thread quit!");

    return true;
  }

private:
  Context_t context_;

  InnerContext_t inner_context_;

  std::vector<std::shared_ptr<BlockQueue<InnerParsingType>>> block_queue_;
  std::vector<std::future<bool>>                             async_futures_;

  std::atomic<bool> pipeline_close_flag_{true};
  std::atomic<bool> pipeline_no_more_input_{true};
  std::atomic<bool> pipeline_initialized_{false};
};

} // namespace easy_deploy
