sudo docker exec -it fth_mamba /bin/bash

mamba activate vidur 

pip install debugpy -i https://mirrors.aliyun.com/pypi/simple/


```sh
python -m vidur.main  \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Meta-Llama-3-8B \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 1 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 512  \
--length_generator_config_type trace \
--trace_request_length_generator_config_max_tokens 16384 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 6.45 \
--replica_scheduler_config_type sarathi  \
--sarathi_scheduler_config_batch_size_cap 512  \
--sarathi_scheduler_config_chunk_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384

```

python -m vidur.main  \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Meta-Llama-3-8B \
--cluster_config_num_replicas 2 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 1 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 512  \
--length_generator_config_type trace \
--trace_request_length_generator_config_max_tokens 16384 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 6.45 \
--replica_scheduler_config_type sarathi  \
--sarathi_scheduler_config_batch_size_cap 512  \
--sarathi_scheduler_config_chunk_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384


python -m vidur.main > test_fth/output_250113_1424.log \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Meta-Llama-3-70B \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 4 \
--replica_config_num_pipeline_stages 2 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 512  \
--length_generator_config_type trace \
--trace_request_length_generator_config_max_tokens 16384 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 6.45 \
--replica_scheduler_config_type sarathi  \
--sarathi_scheduler_config_batch_size_cap 512  \
--sarathi_scheduler_config_chunk_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384


python -m vidur.main > test_fth/output_250113_1424.log \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Meta-Llama-3-8B \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 8 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 512  \
--length_generator_config_type trace \
--trace_request_length_generator_config_max_tokens 16384 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 6.45 \
--replica_scheduler_config_type sarathi  \
--sarathi_scheduler_config_batch_size_cap 512  \
--sarathi_scheduler_config_chunk_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384




python -m vidur.main \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Meta-Llama-3-8B \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 2 \
--replica_config_num_pipeline_stages 1 \
--request_generator_config_type synthetic \
--synthetic_request_generator_config_num_requests 512  \
--length_generator_config_type trace \
--trace_request_length_generator_config_max_tokens 16384 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
--interval_generator_config_type poisson \
--poisson_request_interval_generator_config_qps 6.45 \
--replica_scheduler_config_type sarathi  \
--sarathi_scheduler_config_batch_size_cap 512  \
--sarathi_scheduler_config_chunk_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size 16384 \
--random_forrest_execution_time_predictor_config_prediction_max_batch_size 512 \
--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request 16384


https://mp.weixin.qq.com/s/LrgVvjq5i43a82WlHESvJA

pdb.set_trace()

import pdb

/disk1/futianhao


vidur-search可以运行的命令

python -m vidur.config_optimizer.config_explorer.main --output-dir  /app/software1/vidur/test_fth --config-path /app/software1/vidur/vidur/config_optimizer/config_explorer/config/config.yml


python -m vidur.config_optimizer.config_explorer.main --output-dir  /app/software1/vidur/test_fth --config-path /app/software1/vidur/vidur/config_optimizer/config_explorer/config/config_short_fth.yml

python -m vidur.config_optimizer.config_explorer.main >test_fth/output_search_250122_1011.log --output-dir  /app/software1/vidur/test_fth --config-path /app/software1/vidur/vidur/config_optimizer/config_explorer/config/config.yml

python -m vidur.config_optimizer.config_explorer.main >vidur/config_optimizer/config_explorer/search_output_fth/output_search_250122_1011.log --output-dir  vidur/config_optimizer/config_explorer/search_output_fth --config-path /app/software1/vidur/vidur/config_optimizer/config_explorer/config/config.yml

python -m vidur.config_optimizer.config_explorer.main >vidur/config_optimizer/config_explorer/search_output_fth/output_search_250122_1011.log \
--output-dir  vidur/config_optimizer/config_explorer/search_output_fth \
--config-path /app/software1/vidur/vidur/config_optimizer/config_explorer/config/config.yml
# --trace_request_length_generator_config_trace_file ./data/processed_traces/lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv

python -m vidur.config_optimizer.config_explorer.main \
--output-dir  vidur/config_optimizer/config_explorer/search_output_fth \
--config-path /app/software1/vidur/vidur/config_optimizer/config_explorer/config/config.yml

pdb.set_trace()

/disk1/futianhao/software1/vidur/vidur/config_optimizer/config_explorer/search_output_fth

python -m vidur.main --replica_config_model_name codellama/CodeLlama-34b-Instruct-hf \
--request_generator_config_type synthetic --length_generator_config_type trace \
--interval_generator_config_type poisson --synthetic_request_generator_config_max_tokens 4096 \
--trace_request_length_generator_config_max_tokens 4096 --zipf_request_length_generator_config_max_tokens 4096 \
--uniform_request_length_generator_config_max_tokens 4096 --fixed_request_length_generator_config_max_tokens 4096 \
--trace_request_generator_config_max_tokens 4096 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv \
--trace_request_length_generator_config_prefill_scale_factor 1 --trace_request_length_generator_config_decode_scale_factor 1 \
--synthetic_request_generator_config_num_requests 16000 --vllm_scheduler_config_max_tokens_in_batch 4096 --replica_config_device h100 \
--replica_scheduler_config_type vllm --replica_config_tensor_parallel_size 2 --replica_config_num_pipeline_stages 1 \
--vllm_scheduler_config_batch_size_cap 64 --lightllm_scheduler_config_batch_size_cap 64 --orca_scheduler_config_batch_size_cap 64 \
--faster_transformer_scheduler_config_batch_size_cap 64 --sarathi_scheduler_config_batch_size_cap 64 --cluster_config_num_replicas 8 \
--metrics_config_output_dir vidur/config_optimizer/config_explorer/search_output_fth/runs/0c9b9e8b/32.0 --metrics_config_cache_dir ./cache_tmpfs \
--poisson_request_interval_generator_config_qps 32.0 --gamma_request_interval_generator_config_qps 32.0 --time_limit 1800 \
--no-metrics_config_save_table_to_wandb --no-metrics_config_store_plots --no-metrics_config_store_operation_metrics \
--no-metrics_config_store_token_completion_metrics --no-metrics_config_enable_chrome_trace \
--linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling



nice -n 1  python -m vidur.main --replica_config_model_name meta-llama/Llama-2-7b-hf --request_generator_config_type synthetic --length_generator_config_type trace --interval_generator_config_type poisson --synthetic_request_generator_config_max_tokens 4096 --trace_request_length_generator_config_max_tokens 4096 --zipf_request_length_generator_config_max_tokens 4096 --uniform_request_length_generator_config_max_tokens 4096 --fixed_request_length_generator_config_max_tokens 4096 --trace_request_generator_config_max_tokens 4096 --trace_request_length_generator_config_trace_file ./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv --trace_request_length_generator_config_prefill_scale_factor 1 --trace_request_length_generator_config_decode_scale_factor 1 --synthetic_request_generator_config_num_requests 32 --vllm_scheduler_config_max_tokens_in_batch 4096 --replica_config_device h100 --replica_scheduler_config_type vllm --replica_config_tensor_parallel_size 1 --replica_config_num_pipeline_stages 2 --vllm_scheduler_config_batch_size_cap 64 --lightllm_scheduler_config_batch_size_cap 64 --orca_scheduler_config_batch_size_cap 64 --faster_transformer_scheduler_config_batch_size_cap 64 --sarathi_scheduler_config_batch_size_cap 64 --cluster_config_num_replicas 8 --metrics_config_output_dir vidur/config_optimizer/config_explorer/search_output_fth/runs/7003b794/16.0 --metrics_config_cache_dir ./cache_tmpfs --poisson_request_interval_generator_config_qps 16.0 --gamma_request_interval_generator_config_qps 16.0 --time_limit 1800 --no-metrics_config_save_table_to_wandb --no-metrics_config_store_plots --no-metrics_config_store_operation_metrics --no-metrics_config_store_token_completion_metrics --no-metrics_config_enable_chrome_trace --linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling 


nice -n 1  python -m vidur.main --replica_config_model_name meta-llama/Llama-2-7b-hf --request_generator_config_type synthetic --length_generator_config_type trace \
--interval_generator_config_type poisson --synthetic_request_generator_config_max_tokens 4096 --trace_request_length_generator_config_max_tokens 4096 \
--zipf_request_length_generator_config_max_tokens 4096 --uniform_request_length_generator_config_max_tokens 4096 --fixed_request_length_generator_config_max_tokens 4096 \
--trace_request_generator_config_max_tokens 4096 --trace_request_length_generator_config_trace_file ./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv \
--trace_request_length_generator_config_prefill_scale_factor 1 --trace_request_length_generator_config_decode_scale_factor 1 --synthetic_request_generator_config_num_requests 32 \
--vllm_scheduler_config_max_tokens_in_batch 4096 --replica_config_device h100 --replica_scheduler_config_type vllm --replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 2 --vllm_scheduler_config_batch_size_cap 64 --lightllm_scheduler_config_batch_size_cap 64 --orca_scheduler_config_batch_size_cap 64 \
--faster_transformer_scheduler_config_batch_size_cap 64 --sarathi_scheduler_config_batch_size_cap 64 --cluster_config_num_replicas 8 \
--metrics_config_output_dir vidur/config_optimizer/config_explorer/search_output_fth/runs/7003b794/16.0 --metrics_config_cache_dir ./cache_tmpfs \
--poisson_request_interval_generator_config_qps 16.0 --gamma_request_interval_generator_config_qps 16.0 --time_limit 1800 --no-metrics_config_save_table_to_wandb \
--no-metrics_config_store_plots --no-metrics_config_store_operation_metrics --no-metrics_config_store_token_completion_metrics --no-metrics_config_enable_chrome_trace \
--linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling 


nice -n 1  python -m vidur.main --replica_config_model_name meta-llama/Llama-2-7b-hf --request_generator_config_type synthetic --length_generator_config_type trace \
--trace_request_length_generator_config_trace_file ./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv \
--interval_generator_config_type poisson --synthetic_request_generator_config_max_tokens 4096 --trace_request_length_generator_config_max_tokens 4096 \
--zipf_request_length_generator_config_max_tokens 4096 --uniform_request_length_generator_config_max_tokens 4096 --fixed_request_length_generator_config_max_tokens 4096 \
--trace_request_generator_config_max_tokens 4096 \
--trace_request_length_generator_config_prefill_scale_factor 1 --trace_request_length_generator_config_decode_scale_factor 1 --synthetic_request_generator_config_num_requests 32 \
--vllm_scheduler_config_max_tokens_in_batch 4096 --replica_config_device h100 --replica_scheduler_config_type vllm --replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 2 --vllm_scheduler_config_batch_size_cap 64 --lightllm_scheduler_config_batch_size_cap 64 --orca_scheduler_config_batch_size_cap 64 \
--faster_transformer_scheduler_config_batch_size_cap 64 --sarathi_scheduler_config_batch_size_cap 64 --cluster_config_num_replicas 8 \
--metrics_config_output_dir vidur/config_optimizer/config_explorer/search_output_fth/runs/7003b794/16.0 --metrics_config_cache_dir ./cache_tmpfs \
--poisson_request_interval_generator_config_qps 16.0 --gamma_request_interval_generator_config_qps 16.0 --time_limit 1800 --no-metrics_config_save_table_to_wandb \
--no-metrics_config_store_plots --no-metrics_config_store_operation_metrics --no-metrics_config_store_token_completion_metrics --no-metrics_config_enable_chrome_trace \
--linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling 

