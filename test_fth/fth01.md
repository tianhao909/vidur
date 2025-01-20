(vidur) root@x08j01357:/app/software1/vidur# python -m vidur.main -h
usage: main.py [-h] [--seed SEED] [--log_level LOG_LEVEL] [--time_limit TIME_LIMIT]
               [--cluster_config_num_replicas CLUSTER_CONFIG_NUM_REPLICAS]
               [--replica_config_model_name REPLICA_CONFIG_MODEL_NAME]
               [--replica_config_memory_margin_fraction REPLICA_CONFIG_MEMORY_MARGIN_FRACTION]
               [--replica_config_num_pipeline_stages REPLICA_CONFIG_NUM_PIPELINE_STAGES]
               [--replica_config_tensor_parallel_size REPLICA_CONFIG_TENSOR_PARALLEL_SIZE]
               [--replica_config_device REPLICA_CONFIG_DEVICE]
               [--replica_config_network_device REPLICA_CONFIG_NETWORK_DEVICE]
               [--global_scheduler_config_type GLOBAL_SCHEDULER_CONFIG_TYPE]
               [--replica_scheduler_config_type REPLICA_SCHEDULER_CONFIG_TYPE]
               [--vllm_scheduler_config_batch_size_cap VLLM_SCHEDULER_CONFIG_BATCH_SIZE_CAP]
               [--vllm_scheduler_config_block_size VLLM_SCHEDULER_CONFIG_BLOCK_SIZE]
               [--vllm_scheduler_config_watermark_blocks_fraction VLLM_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION]
               [--vllm_scheduler_config_num_blocks VLLM_SCHEDULER_CONFIG_NUM_BLOCKS]
               [--vllm_scheduler_config_max_tokens_in_batch VLLM_SCHEDULER_CONFIG_MAX_TOKENS_IN_BATCH]
               [--lightllm_scheduler_config_batch_size_cap LIGHTLLM_SCHEDULER_CONFIG_BATCH_SIZE_CAP]
               [--lightllm_scheduler_config_block_size LIGHTLLM_SCHEDULER_CONFIG_BLOCK_SIZE]
               [--lightllm_scheduler_config_watermark_blocks_fraction LIGHTLLM_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION]
               [--lightllm_scheduler_config_num_blocks LIGHTLLM_SCHEDULER_CONFIG_NUM_BLOCKS]
               [--lightllm_scheduler_config_max_tokens_in_batch LIGHTLLM_SCHEDULER_CONFIG_MAX_TOKENS_IN_BATCH]
               [--lightllm_scheduler_config_max_waiting_iters LIGHTLLM_SCHEDULER_CONFIG_MAX_WAITING_ITERS]
               [--orca_scheduler_config_batch_size_cap ORCA_SCHEDULER_CONFIG_BATCH_SIZE_CAP]
               [--orca_scheduler_config_block_size ORCA_SCHEDULER_CONFIG_BLOCK_SIZE]
               [--orca_scheduler_config_watermark_blocks_fraction ORCA_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION]
               [--orca_scheduler_config_num_blocks ORCA_SCHEDULER_CONFIG_NUM_BLOCKS]
               [--faster_transformer_scheduler_config_batch_size_cap FASTER_TRANSFORMER_SCHEDULER_CONFIG_BATCH_SIZE_CAP]
               [--faster_transformer_scheduler_config_block_size FASTER_TRANSFORMER_SCHEDULER_CONFIG_BLOCK_SIZE]
               [--faster_transformer_scheduler_config_watermark_blocks_fraction FASTER_TRANSFORMER_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION]
               [--faster_transformer_scheduler_config_num_blocks FASTER_TRANSFORMER_SCHEDULER_CONFIG_NUM_BLOCKS]
               [--sarathi_scheduler_config_batch_size_cap SARATHI_SCHEDULER_CONFIG_BATCH_SIZE_CAP]
               [--sarathi_scheduler_config_block_size SARATHI_SCHEDULER_CONFIG_BLOCK_SIZE]
               [--sarathi_scheduler_config_watermark_blocks_fraction SARATHI_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION]
               [--sarathi_scheduler_config_num_blocks SARATHI_SCHEDULER_CONFIG_NUM_BLOCKS]
               [--sarathi_scheduler_config_chunk_size SARATHI_SCHEDULER_CONFIG_CHUNK_SIZE]
               [--request_generator_config_type REQUEST_GENERATOR_CONFIG_TYPE]
               [--synthetic_request_generator_config_seed SYNTHETIC_REQUEST_GENERATOR_CONFIG_SEED]
               [--length_generator_config_type LENGTH_GENERATOR_CONFIG_TYPE]
               [--trace_request_length_generator_config_seed TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_SEED]
               [--trace_request_length_generator_config_max_tokens TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_MAX_TOKENS]
               [--trace_request_length_generator_config_trace_file TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_TRACE_FILE]
               [--trace_request_length_generator_config_prefill_scale_factor TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_PREFILL_SCALE_FACTOR]
               [--trace_request_length_generator_config_decode_scale_factor TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_DECODE_SCALE_FACTOR]
               [--zipf_request_length_generator_config_seed ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_SEED]
               [--zipf_request_length_generator_config_max_tokens ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_MAX_TOKENS]
               [--zipf_request_length_generator_config_theta ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_THETA]
               [--zipf_request_length_generator_config_scramble | --no-zipf_request_length_generator_config_scramble]
               [--zipf_request_length_generator_config_min_tokens ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_MIN_TOKENS]
               [--zipf_request_length_generator_config_prefill_to_decode_ratio ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_PREFILL_TO_DECODE_RATIO]
               [--uniform_request_length_generator_config_seed UNIFORM_REQUEST_LENGTH_GENERATOR_CONFIG_SEED]
               [--uniform_request_length_generator_config_max_tokens UNIFORM_REQUEST_LENGTH_GENERATOR_CONFIG_MAX_TOKENS]
               [--uniform_request_length_generator_config_min_tokens UNIFORM_REQUEST_LENGTH_GENERATOR_CONFIG_MIN_TOKENS]
               [--uniform_request_length_generator_config_prefill_to_decode_ratio UNIFORM_REQUEST_LENGTH_GENERATOR_CONFIG_PREFILL_TO_DECODE_RATIO]
               [--fixed_request_length_generator_config_seed FIXED_REQUEST_LENGTH_GENERATOR_CONFIG_SEED]
               [--fixed_request_length_generator_config_max_tokens FIXED_REQUEST_LENGTH_GENERATOR_CONFIG_MAX_TOKENS]
               [--fixed_request_length_generator_config_prefill_tokens FIXED_REQUEST_LENGTH_GENERATOR_CONFIG_PREFILL_TOKENS]
               [--fixed_request_length_generator_config_decode_tokens FIXED_REQUEST_LENGTH_GENERATOR_CONFIG_DECODE_TOKENS]
               [--interval_generator_config_type INTERVAL_GENERATOR_CONFIG_TYPE]
               [--trace_request_interval_generator_config_seed TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_SEED]
               [--trace_request_interval_generator_config_trace_file TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_TRACE_FILE]
               [--trace_request_interval_generator_config_start_time TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_START_TIME]
               [--trace_request_interval_generator_config_end_time TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_END_TIME]
               [--trace_request_interval_generator_config_time_scale_factor TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_TIME_SCALE_FACTOR]
               [--poisson_request_interval_generator_config_seed POISSON_REQUEST_INTERVAL_GENERATOR_CONFIG_SEED]
               [--poisson_request_interval_generator_config_qps POISSON_REQUEST_INTERVAL_GENERATOR_CONFIG_QPS]
               [--gamma_request_interval_generator_config_seed GAMMA_REQUEST_INTERVAL_GENERATOR_CONFIG_SEED]
               [--gamma_request_interval_generator_config_qps GAMMA_REQUEST_INTERVAL_GENERATOR_CONFIG_QPS]
               [--gamma_request_interval_generator_config_cv GAMMA_REQUEST_INTERVAL_GENERATOR_CONFIG_CV]
               [--static_request_interval_generator_config_seed STATIC_REQUEST_INTERVAL_GENERATOR_CONFIG_SEED]
               [--synthetic_request_generator_config_num_requests SYNTHETIC_REQUEST_GENERATOR_CONFIG_NUM_REQUESTS]
               [--synthetic_request_generator_config_duration SYNTHETIC_REQUEST_GENERATOR_CONFIG_DURATION]
               [--trace_request_generator_config_seed TRACE_REQUEST_GENERATOR_CONFIG_SEED]
               [--trace_request_generator_config_trace_file TRACE_REQUEST_GENERATOR_CONFIG_TRACE_FILE]
               [--trace_request_generator_config_prefill_scale_factor TRACE_REQUEST_GENERATOR_CONFIG_PREFILL_SCALE_FACTOR]
               [--trace_request_generator_config_decode_scale_factor TRACE_REQUEST_GENERATOR_CONFIG_DECODE_SCALE_FACTOR]
               [--trace_request_generator_config_time_scale_factor TRACE_REQUEST_GENERATOR_CONFIG_TIME_SCALE_FACTOR]
               [--trace_request_generator_config_max_tokens TRACE_REQUEST_GENERATOR_CONFIG_MAX_TOKENS]
               [--execution_time_predictor_config_type EXECUTION_TIME_PREDICTOR_CONFIG_TYPE]
               [--linear_regression_execution_time_predictor_config_compute_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_COMPUTE_INPUT_FILE]
               [--linear_regression_execution_time_predictor_config_attention_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_INPUT_FILE]
               [--linear_regression_execution_time_predictor_config_all_reduce_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_ALL_REDUCE_INPUT_FILE]
               [--linear_regression_execution_time_predictor_config_send_recv_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_SEND_RECV_INPUT_FILE]
               [--linear_regression_execution_time_predictor_config_cpu_overhead_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_CPU_OVERHEAD_INPUT_FILE]
               [--linear_regression_execution_time_predictor_config_k_fold_cv_splits LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_K_FOLD_CV_SPLITS]
               [--linear_regression_execution_time_predictor_config_no_cache | --no-linear_regression_execution_time_predictor_config_no_cache]
               [--linear_regression_execution_time_predictor_config_kv_cache_prediction_granularity LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_KV_CACHE_PREDICTION_GRANULARITY]
               [--linear_regression_execution_time_predictor_config_prediction_max_prefill_chunk_size LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_PREFILL_CHUNK_SIZE]
               [--linear_regression_execution_time_predictor_config_prediction_max_batch_size LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_BATCH_SIZE]
               [--linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_TOKENS_PER_REQUEST]
               [--linear_regression_execution_time_predictor_config_attention_decode_batching_overhead_fraction LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_DECODE_BATCHING_OVERHEAD_FRACTION]
               [--linear_regression_execution_time_predictor_config_attention_prefill_batching_overhead_fraction LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_PREFILL_BATCHING_OVERHEAD_FRACTION]
               [--linear_regression_execution_time_predictor_config_nccl_cpu_launch_overhead_ms LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_NCCL_CPU_LAUNCH_OVERHEAD_MS]
               [--linear_regression_execution_time_predictor_config_nccl_cpu_skew_overhead_per_device_ms LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_NCCL_CPU_SKEW_OVERHEAD_PER_DEVICE_MS]
               [--linear_regression_execution_time_predictor_config_num_training_job_threads LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_NUM_TRAINING_JOB_THREADS]
               [--linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling | --no-linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling]
               [--linear_regression_execution_time_predictor_config_polynomial_degree LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_DEGREE [LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_DEGREE ...]]
               [--linear_regression_execution_time_predictor_config_polynomial_include_bias LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_INCLUDE_BIAS [LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_INCLUDE_BIAS ...]]
               [--linear_regression_execution_time_predictor_config_polynomial_interaction_only LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_INTERACTION_ONLY [LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_INTERACTION_ONLY ...]]
               [--linear_regression_execution_time_predictor_config_fit_intercept LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_FIT_INTERCEPT [LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_FIT_INTERCEPT ...]]
               [--random_forrest_execution_time_predictor_config_compute_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_COMPUTE_INPUT_FILE]
               [--random_forrest_execution_time_predictor_config_attention_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_INPUT_FILE]
               [--random_forrest_execution_time_predictor_config_all_reduce_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_ALL_REDUCE_INPUT_FILE]
               [--random_forrest_execution_time_predictor_config_send_recv_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_SEND_RECV_INPUT_FILE]
               [--random_forrest_execution_time_predictor_config_cpu_overhead_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_CPU_OVERHEAD_INPUT_FILE]
               [--random_forrest_execution_time_predictor_config_k_fold_cv_splits RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_K_FOLD_CV_SPLITS]
               [--random_forrest_execution_time_predictor_config_no_cache | --no-random_forrest_execution_time_predictor_config_no_cache]
               [--random_forrest_execution_time_predictor_config_kv_cache_prediction_granularity RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_KV_CACHE_PREDICTION_GRANULARITY]
               [--random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_PREFILL_CHUNK_SIZE]
               [--random_forrest_execution_time_predictor_config_prediction_max_batch_size RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_BATCH_SIZE]
               [--random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_TOKENS_PER_REQUEST]
               [--random_forrest_execution_time_predictor_config_attention_decode_batching_overhead_fraction RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_DECODE_BATCHING_OVERHEAD_FRACTION]
               [--random_forrest_execution_time_predictor_config_attention_prefill_batching_overhead_fraction RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_PREFILL_BATCHING_OVERHEAD_FRACTION]
               [--random_forrest_execution_time_predictor_config_nccl_cpu_launch_overhead_ms RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NCCL_CPU_LAUNCH_OVERHEAD_MS]
               [--random_forrest_execution_time_predictor_config_nccl_cpu_skew_overhead_per_device_ms RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NCCL_CPU_SKEW_OVERHEAD_PER_DEVICE_MS]
               [--random_forrest_execution_time_predictor_config_num_training_job_threads RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NUM_TRAINING_JOB_THREADS]
               [--random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling | --no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling]
               [--random_forrest_execution_time_predictor_config_num_estimators RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NUM_ESTIMATORS [RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NUM_ESTIMATORS ...]]
               [--random_forrest_execution_time_predictor_config_max_depth RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_MAX_DEPTH [RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_MAX_DEPTH ...]]
               [--random_forrest_execution_time_predictor_config_min_samples_split RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_MIN_SAMPLES_SPLIT [RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_MIN_SAMPLES_SPLIT ...]]
               [--metrics_config_write_metrics | --no-metrics_config_write_metrics]
               [--metrics_config_write_json_trace | --no-metrics_config_write_json_trace]
               [--metrics_config_wandb_project METRICS_CONFIG_WANDB_PROJECT]
               [--metrics_config_wandb_group METRICS_CONFIG_WANDB_GROUP]
               [--metrics_config_wandb_run_name METRICS_CONFIG_WANDB_RUN_NAME]
               [--metrics_config_wandb_sweep_id METRICS_CONFIG_WANDB_SWEEP_ID]
               [--metrics_config_wandb_run_id METRICS_CONFIG_WANDB_RUN_ID]
               [--metrics_config_enable_chrome_trace | --no-metrics_config_enable_chrome_trace]
               [--metrics_config_save_table_to_wandb | --no-metrics_config_save_table_to_wandb]
               [--metrics_config_store_plots | --no-metrics_config_store_plots]
               [--metrics_config_store_operation_metrics | --no-metrics_config_store_operation_metrics]
               [--metrics_config_store_token_completion_metrics | --no-metrics_config_store_token_completion_metrics]
               [--metrics_config_store_request_metrics | --no-metrics_config_store_request_metrics]
               [--metrics_config_store_batch_metrics | --no-metrics_config_store_batch_metrics]
               [--metrics_config_store_utilization_metrics | --no-metrics_config_store_utilization_metrics]
               [--metrics_config_keep_individual_batch_metrics | --no-metrics_config_keep_individual_batch_metrics]
               [--metrics_config_subsamples METRICS_CONFIG_SUBSAMPLES]
               [--metrics_config_min_batch_index METRICS_CONFIG_MIN_BATCH_INDEX]
               [--metrics_config_max_batch_index METRICS_CONFIG_MAX_BATCH_INDEX]
               [--metrics_config_output_dir METRICS_CONFIG_OUTPUT_DIR] [--metrics_config_cache_dir METRICS_CONFIG_CACHE_DIR]

options:
  -h, --help            show this help message and exit
  --seed SEED           Seed for the random number generator. (default: 42)
  --log_level LOG_LEVEL
                        Logging level. (default: info)
  --time_limit TIME_LIMIT
                        Time limit for simulation in seconds. 0 means no limit. (default: 0)
  --cluster_config_num_replicas CLUSTER_CONFIG_NUM_REPLICAS
                        Number of replicas. (default: 1)
  --replica_config_model_name REPLICA_CONFIG_MODEL_NAME
                        Model name. (default: meta-llama/Llama-2-7b-hf)
  --replica_config_memory_margin_fraction REPLICA_CONFIG_MEMORY_MARGIN_FRACTION
                        Memory margin fraction. (default: 0.1)
  --replica_config_num_pipeline_stages REPLICA_CONFIG_NUM_PIPELINE_STAGES
                        Number of pipeline stages. (default: 1)
  --replica_config_tensor_parallel_size REPLICA_CONFIG_TENSOR_PARALLEL_SIZE
                        Tensor parallel size. (default: 1)
  --replica_config_device REPLICA_CONFIG_DEVICE
                        Device. (default: a100)
  --replica_config_network_device REPLICA_CONFIG_NETWORK_DEVICE
                        Network device. (default: a100_pairwise_nvlink)
  --global_scheduler_config_type GLOBAL_SCHEDULER_CONFIG_TYPE
                        Global scheduler config. (default: round_robin)
  --replica_scheduler_config_type REPLICA_SCHEDULER_CONFIG_TYPE
                        Replica scheduler config. (default: sarathi)
  --vllm_scheduler_config_batch_size_cap VLLM_SCHEDULER_CONFIG_BATCH_SIZE_CAP
                        Maximum batch size cap. (default: 128)
  --vllm_scheduler_config_block_size VLLM_SCHEDULER_CONFIG_BLOCK_SIZE
                        Block size. (default: 16)
  --vllm_scheduler_config_watermark_blocks_fraction VLLM_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION
                        Watermark blocks fraction. (default: 0.01)
  --vllm_scheduler_config_num_blocks VLLM_SCHEDULER_CONFIG_NUM_BLOCKS
                        Number of blocks. (default: None)
  --vllm_scheduler_config_max_tokens_in_batch VLLM_SCHEDULER_CONFIG_MAX_TOKENS_IN_BATCH
                        Maximum tokens in batch for vLLM. (default: 4096)
  --lightllm_scheduler_config_batch_size_cap LIGHTLLM_SCHEDULER_CONFIG_BATCH_SIZE_CAP
                        Maximum batch size cap. (default: 128)
  --lightllm_scheduler_config_block_size LIGHTLLM_SCHEDULER_CONFIG_BLOCK_SIZE
                        Block size. (default: 16)
  --lightllm_scheduler_config_watermark_blocks_fraction LIGHTLLM_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION
                        Watermark blocks fraction. (default: 0.01)
  --lightllm_scheduler_config_num_blocks LIGHTLLM_SCHEDULER_CONFIG_NUM_BLOCKS
                        Number of blocks. (default: None)
  --lightllm_scheduler_config_max_tokens_in_batch LIGHTLLM_SCHEDULER_CONFIG_MAX_TOKENS_IN_BATCH
                        Maximum tokens in batch for LightLLM. (default: 4096)
  --lightllm_scheduler_config_max_waiting_iters LIGHTLLM_SCHEDULER_CONFIG_MAX_WAITING_ITERS
                        Maximum waiting iterations for LightLLM. (default: 10)
  --orca_scheduler_config_batch_size_cap ORCA_SCHEDULER_CONFIG_BATCH_SIZE_CAP
                        Maximum batch size cap. (default: 128)
  --orca_scheduler_config_block_size ORCA_SCHEDULER_CONFIG_BLOCK_SIZE
                        Block size. (default: 16)
  --orca_scheduler_config_watermark_blocks_fraction ORCA_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION
                        Watermark blocks fraction. (default: 0.01)
  --orca_scheduler_config_num_blocks ORCA_SCHEDULER_CONFIG_NUM_BLOCKS
                        Number of blocks. (default: None)
  --faster_transformer_scheduler_config_batch_size_cap FASTER_TRANSFORMER_SCHEDULER_CONFIG_BATCH_SIZE_CAP
                        Maximum batch size cap. (default: 128)
  --faster_transformer_scheduler_config_block_size FASTER_TRANSFORMER_SCHEDULER_CONFIG_BLOCK_SIZE
                        Block size. (default: 16)
  --faster_transformer_scheduler_config_watermark_blocks_fraction FASTER_TRANSFORMER_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION
                        Watermark blocks fraction. (default: 0.01)
  --faster_transformer_scheduler_config_num_blocks FASTER_TRANSFORMER_SCHEDULER_CONFIG_NUM_BLOCKS
                        Number of blocks. (default: None)
  --sarathi_scheduler_config_batch_size_cap SARATHI_SCHEDULER_CONFIG_BATCH_SIZE_CAP
                        Maximum batch size cap. (default: 128)
  --sarathi_scheduler_config_block_size SARATHI_SCHEDULER_CONFIG_BLOCK_SIZE
                        Block size. (default: 16)
  --sarathi_scheduler_config_watermark_blocks_fraction SARATHI_SCHEDULER_CONFIG_WATERMARK_BLOCKS_FRACTION
                        Watermark blocks fraction. (default: 0.01)
  --sarathi_scheduler_config_num_blocks SARATHI_SCHEDULER_CONFIG_NUM_BLOCKS
                        Number of blocks. (default: None)
  --sarathi_scheduler_config_chunk_size SARATHI_SCHEDULER_CONFIG_CHUNK_SIZE
                        Chunk size for Sarathi. (default: 512)
  --request_generator_config_type REQUEST_GENERATOR_CONFIG_TYPE
                        Request generator config. (default: synthetic)
  --synthetic_request_generator_config_seed SYNTHETIC_REQUEST_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --length_generator_config_type LENGTH_GENERATOR_CONFIG_TYPE
                        Length generator config for Synthetic Request Generator. (default: fixed)
  --trace_request_length_generator_config_seed TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --trace_request_length_generator_config_max_tokens TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_MAX_TOKENS
                        Maximum tokens. (default: 4096)
  --trace_request_length_generator_config_trace_file TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_TRACE_FILE
                        Path to the trace request length generator file. (default:
                        data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv)
  --trace_request_length_generator_config_prefill_scale_factor TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_PREFILL_SCALE_FACTOR
                        Prefill scale factor for the trace request length generator. (default: 1)
  --trace_request_length_generator_config_decode_scale_factor TRACE_REQUEST_LENGTH_GENERATOR_CONFIG_DECODE_SCALE_FACTOR
                        Decode scale factor for the trace request length generator. (default: 1)
  --zipf_request_length_generator_config_seed ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --zipf_request_length_generator_config_max_tokens ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_MAX_TOKENS
                        Maximum tokens. (default: 4096)
  --zipf_request_length_generator_config_theta ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_THETA
                        Theta for Zipf Request Length Generator. (default: 0.6)
  --zipf_request_length_generator_config_scramble, --no-zipf_request_length_generator_config_scramble
                        Scramble for Zipf Request Length Generator. (default: False)
  --zipf_request_length_generator_config_min_tokens ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_MIN_TOKENS
                        Minimum tokens for Zipf Request Length Generator. (default: 1024)
  --zipf_request_length_generator_config_prefill_to_decode_ratio ZIPF_REQUEST_LENGTH_GENERATOR_CONFIG_PREFILL_TO_DECODE_RATIO
                        Prefill to decode ratio for Zipf Request Length Generator. (default: 20.0)
  --uniform_request_length_generator_config_seed UNIFORM_REQUEST_LENGTH_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --uniform_request_length_generator_config_max_tokens UNIFORM_REQUEST_LENGTH_GENERATOR_CONFIG_MAX_TOKENS
                        Maximum tokens. (default: 4096)
  --uniform_request_length_generator_config_min_tokens UNIFORM_REQUEST_LENGTH_GENERATOR_CONFIG_MIN_TOKENS
                        Minimum tokens for Uniform Request Length Generator. (default: 1024)
  --uniform_request_length_generator_config_prefill_to_decode_ratio UNIFORM_REQUEST_LENGTH_GENERATOR_CONFIG_PREFILL_TO_DECODE_RATIO
                        Prefill to decode ratio for Uniform Request Length Generator. (default: 20.0)
  --fixed_request_length_generator_config_seed FIXED_REQUEST_LENGTH_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --fixed_request_length_generator_config_max_tokens FIXED_REQUEST_LENGTH_GENERATOR_CONFIG_MAX_TOKENS
                        Maximum tokens. (default: 4096)
  --fixed_request_length_generator_config_prefill_tokens FIXED_REQUEST_LENGTH_GENERATOR_CONFIG_PREFILL_TOKENS
                        Prefill tokens for Fixed Request Length Generator. (default: 2048)
  --fixed_request_length_generator_config_decode_tokens FIXED_REQUEST_LENGTH_GENERATOR_CONFIG_DECODE_TOKENS
                        Decode tokens for Fixed Request Length Generator. (default: 512)
  --interval_generator_config_type INTERVAL_GENERATOR_CONFIG_TYPE
                        Interval generator config for Synthetic Request Generator. (default: poisson)
  --trace_request_interval_generator_config_seed TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --trace_request_interval_generator_config_trace_file TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_TRACE_FILE
                        Path to the trace request interval generator file. (default:
                        data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv)
  --trace_request_interval_generator_config_start_time TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_START_TIME
                        Start time of the trace request interval generator. (default: 1970-01-04 12:00:00)
  --trace_request_interval_generator_config_end_time TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_END_TIME
                        End time of the trace request interval generator. (default: 1970-01-04 15:00:00)
  --trace_request_interval_generator_config_time_scale_factor TRACE_REQUEST_INTERVAL_GENERATOR_CONFIG_TIME_SCALE_FACTOR
                        Time scale factor for the trace request interval generator. (default: 1.0)
  --poisson_request_interval_generator_config_seed POISSON_REQUEST_INTERVAL_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --poisson_request_interval_generator_config_qps POISSON_REQUEST_INTERVAL_GENERATOR_CONFIG_QPS
                        Queries per second for Poisson Request Interval Generator. (default: 0.5)
  --gamma_request_interval_generator_config_seed GAMMA_REQUEST_INTERVAL_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --gamma_request_interval_generator_config_qps GAMMA_REQUEST_INTERVAL_GENERATOR_CONFIG_QPS
                        Queries per second for Gamma Request Interval Generator. (default: 0.2)
  --gamma_request_interval_generator_config_cv GAMMA_REQUEST_INTERVAL_GENERATOR_CONFIG_CV
                        Coefficient of variation for Gamma Request Interval Generator. (default: 0.5)
  --static_request_interval_generator_config_seed STATIC_REQUEST_INTERVAL_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --synthetic_request_generator_config_num_requests SYNTHETIC_REQUEST_GENERATOR_CONFIG_NUM_REQUESTS
                        Number of requests for Synthetic Request Generator. (default: 128)
  --synthetic_request_generator_config_duration SYNTHETIC_REQUEST_GENERATOR_CONFIG_DURATION
                        Duration of the synthetic request generator. (default: None)
  --trace_request_generator_config_seed TRACE_REQUEST_GENERATOR_CONFIG_SEED
                        Seed for the random number generator. (default: 42)
  --trace_request_generator_config_trace_file TRACE_REQUEST_GENERATOR_CONFIG_TRACE_FILE
                        Path to the trace request generator file. (default: data/processed_traces/splitwise_conv.csv)
  --trace_request_generator_config_prefill_scale_factor TRACE_REQUEST_GENERATOR_CONFIG_PREFILL_SCALE_FACTOR
                        Prefill scale factor for the trace request generator. (default: 1.0)
  --trace_request_generator_config_decode_scale_factor TRACE_REQUEST_GENERATOR_CONFIG_DECODE_SCALE_FACTOR
                        Decode scale factor for the trace request generator. (default: 1.0)
  --trace_request_generator_config_time_scale_factor TRACE_REQUEST_GENERATOR_CONFIG_TIME_SCALE_FACTOR
                        Time scale factor for the trace request generator. (default: 1.0)
  --trace_request_generator_config_max_tokens TRACE_REQUEST_GENERATOR_CONFIG_MAX_TOKENS
                        Maximum tokens for the trace request generator. (default: 4096)
  --execution_time_predictor_config_type EXECUTION_TIME_PREDICTOR_CONFIG_TYPE
                        Execution time predictor config. (default: random_forrest)
  --linear_regression_execution_time_predictor_config_compute_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_COMPUTE_INPUT_FILE
                        Path to the compute input file. (default: ./data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv)
  --linear_regression_execution_time_predictor_config_attention_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_INPUT_FILE
                        Path to the attention input file. (default: ./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv)
  --linear_regression_execution_time_predictor_config_all_reduce_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_ALL_REDUCE_INPUT_FILE
                        Path to the all reduce input file. (default:
                        ./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv)
  --linear_regression_execution_time_predictor_config_send_recv_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_SEND_RECV_INPUT_FILE
                        Path to the send recv input file. (default: ./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv)
  --linear_regression_execution_time_predictor_config_cpu_overhead_input_file LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_CPU_OVERHEAD_INPUT_FILE
                        Path to the cpu overhead input file. (default:
                        ./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv)
  --linear_regression_execution_time_predictor_config_k_fold_cv_splits LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_K_FOLD_CV_SPLITS
                        Number of k fold cross validation splits. (default: 10)
  --linear_regression_execution_time_predictor_config_no_cache, --no-linear_regression_execution_time_predictor_config_no_cache
                        Whether to cache prediction models. (default: False)
  --linear_regression_execution_time_predictor_config_kv_cache_prediction_granularity LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_KV_CACHE_PREDICTION_GRANULARITY
                        KV cache prediction granularity. (default: 64)
  --linear_regression_execution_time_predictor_config_prediction_max_prefill_chunk_size LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_PREFILL_CHUNK_SIZE
                        Max prefill chunk size for prediction. (default: 4096)
  --linear_regression_execution_time_predictor_config_prediction_max_batch_size LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_BATCH_SIZE
                        Max batch size for prediction. (default: 128)
  --linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_TOKENS_PER_REQUEST
                        Max tokens per request for prediction. (default: 4096)
  --linear_regression_execution_time_predictor_config_attention_decode_batching_overhead_fraction LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_DECODE_BATCHING_OVERHEAD_FRACTION
                        Attention decode batching overhead fraction. (default: 0.1)
  --linear_regression_execution_time_predictor_config_attention_prefill_batching_overhead_fraction LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_PREFILL_BATCHING_OVERHEAD_FRACTION
                        Attention prefill batching overhead fraction. (default: 0.1)
  --linear_regression_execution_time_predictor_config_nccl_cpu_launch_overhead_ms LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_NCCL_CPU_LAUNCH_OVERHEAD_MS
                        NCCL CPU launch overhead in ms. (default: 0.02)
  --linear_regression_execution_time_predictor_config_nccl_cpu_skew_overhead_per_device_ms LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_NCCL_CPU_SKEW_OVERHEAD_PER_DEVICE_MS
                        NCCL CPU skew overhead per device in ms. (default: 0.0)
  --linear_regression_execution_time_predictor_config_num_training_job_threads LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_NUM_TRAINING_JOB_THREADS
                        Number of training job threads. (default: -1)
  --linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling, --no-linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling
                        Whether to skip CPU overhead modeling. (default: True)
  --linear_regression_execution_time_predictor_config_polynomial_degree LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_DEGREE [LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_DEGREE ...]
                        Polynomial degree for linear regression. (default: [1, 2, 3, 4, 5])
  --linear_regression_execution_time_predictor_config_polynomial_include_bias LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_INCLUDE_BIAS [LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_INCLUDE_BIAS ...]
                        Polynomial include bias for linear regression. (default: [True, False])
  --linear_regression_execution_time_predictor_config_polynomial_interaction_only LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_INTERACTION_ONLY [LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_POLYNOMIAL_INTERACTION_ONLY ...]
                        Polynomial interaction only for linear regression. (default: [True, False])
  --linear_regression_execution_time_predictor_config_fit_intercept LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_FIT_INTERCEPT [LINEAR_REGRESSION_EXECUTION_TIME_PREDICTOR_CONFIG_FIT_INTERCEPT ...]
                        Fit intercept for linear regression. (default: [True, False])
  --random_forrest_execution_time_predictor_config_compute_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_COMPUTE_INPUT_FILE
                        Path to the compute input file. (default: ./data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv)
  --random_forrest_execution_time_predictor_config_attention_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_INPUT_FILE
                        Path to the attention input file. (default: ./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv)
  --random_forrest_execution_time_predictor_config_all_reduce_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_ALL_REDUCE_INPUT_FILE
                        Path to the all reduce input file. (default:
                        ./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv)
  --random_forrest_execution_time_predictor_config_send_recv_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_SEND_RECV_INPUT_FILE
                        Path to the send recv input file. (default: ./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv)
  --random_forrest_execution_time_predictor_config_cpu_overhead_input_file RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_CPU_OVERHEAD_INPUT_FILE
                        Path to the cpu overhead input file. (default:
                        ./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv)
  --random_forrest_execution_time_predictor_config_k_fold_cv_splits RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_K_FOLD_CV_SPLITS
                        Number of k fold cross validation splits. (default: 10)
  --random_forrest_execution_time_predictor_config_no_cache, --no-random_forrest_execution_time_predictor_config_no_cache
                        Whether to cache prediction models. (default: False)
  --random_forrest_execution_time_predictor_config_kv_cache_prediction_granularity RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_KV_CACHE_PREDICTION_GRANULARITY
                        KV cache prediction granularity. (default: 64)
  --random_forrest_execution_time_predictor_config_prediction_max_prefill_chunk_size RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_PREFILL_CHUNK_SIZE
                        Max prefill chunk size for prediction. (default: 4096)
  --random_forrest_execution_time_predictor_config_prediction_max_batch_size RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_BATCH_SIZE
                        Max batch size for prediction. (default: 128)
  --random_forrest_execution_time_predictor_config_prediction_max_tokens_per_request RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_PREDICTION_MAX_TOKENS_PER_REQUEST
                        Max tokens per request for prediction. (default: 4096)
  --random_forrest_execution_time_predictor_config_attention_decode_batching_overhead_fraction RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_DECODE_BATCHING_OVERHEAD_FRACTION
                        Attention decode batching overhead fraction. (default: 0.1)
  --random_forrest_execution_time_predictor_config_attention_prefill_batching_overhead_fraction RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_ATTENTION_PREFILL_BATCHING_OVERHEAD_FRACTION
                        Attention prefill batching overhead fraction. (default: 0.1)
  --random_forrest_execution_time_predictor_config_nccl_cpu_launch_overhead_ms RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NCCL_CPU_LAUNCH_OVERHEAD_MS
                        NCCL CPU launch overhead in ms. (default: 0.02)
  --random_forrest_execution_time_predictor_config_nccl_cpu_skew_overhead_per_device_ms RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NCCL_CPU_SKEW_OVERHEAD_PER_DEVICE_MS
                        NCCL CPU skew overhead per device in ms. (default: 0.0)
  --random_forrest_execution_time_predictor_config_num_training_job_threads RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NUM_TRAINING_JOB_THREADS
                        Number of training job threads. (default: -1)
  --random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling, --no-random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling
                        Whether to skip CPU overhead modeling. (default: True)
  --random_forrest_execution_time_predictor_config_num_estimators RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NUM_ESTIMATORS [RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_NUM_ESTIMATORS ...]
                        Number of estimators for random forest. (default: [250, 500, 750])
  --random_forrest_execution_time_predictor_config_max_depth RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_MAX_DEPTH [RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_MAX_DEPTH ...]
                        Maximum depth for random forest. (default: [8, 16, 32])
  --random_forrest_execution_time_predictor_config_min_samples_split RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_MIN_SAMPLES_SPLIT [RANDOM_FORREST_EXECUTION_TIME_PREDICTOR_CONFIG_MIN_SAMPLES_SPLIT ...]
                        Minimum samples split for random forest. (default: [2, 5, 10])
  --metrics_config_write_metrics, --no-metrics_config_write_metrics
                        Whether to write metrics. (default: True)
  --metrics_config_write_json_trace, --no-metrics_config_write_json_trace
                        Whether to write json trace. (default: False)
  --metrics_config_wandb_project METRICS_CONFIG_WANDB_PROJECT
                        Weights & Biases project name. (default: None)
  --metrics_config_wandb_group METRICS_CONFIG_WANDB_GROUP
                        Weights & Biases group name. (default: None)
  --metrics_config_wandb_run_name METRICS_CONFIG_WANDB_RUN_NAME
                        Weights & Biases run name. (default: None)
  --metrics_config_wandb_sweep_id METRICS_CONFIG_WANDB_SWEEP_ID
                        Weights & Biases sweep id. (default: None)
  --metrics_config_wandb_run_id METRICS_CONFIG_WANDB_RUN_ID
                        Weights & Biases run id. (default: None)
  --metrics_config_enable_chrome_trace, --no-metrics_config_enable_chrome_trace
                        Enable Chrome tracing. (default: True)
  --metrics_config_save_table_to_wandb, --no-metrics_config_save_table_to_wandb
                        Whether to save table to wandb. (default: False)
  --metrics_config_store_plots, --no-metrics_config_store_plots
                        Whether to store plots. (default: True)
  --metrics_config_store_operation_metrics, --no-metrics_config_store_operation_metrics
                        Whether to store operation metrics. (default: False)
  --metrics_config_store_token_completion_metrics, --no-metrics_config_store_token_completion_metrics
                        Whether to store token completion metrics. (default: False)
  --metrics_config_store_request_metrics, --no-metrics_config_store_request_metrics
                        Whether to store request metrics. (default: True)
  --metrics_config_store_batch_metrics, --no-metrics_config_store_batch_metrics
                        Whether to store batch metrics. (default: True)
  --metrics_config_store_utilization_metrics, --no-metrics_config_store_utilization_metrics
                        Whether to store utilization metrics. (default: True)
  --metrics_config_keep_individual_batch_metrics, --no-metrics_config_keep_individual_batch_metrics
                        Whether to keep individual batch metrics. (default: False)
  --metrics_config_subsamples METRICS_CONFIG_SUBSAMPLES
                        Subsamples. (default: None)
  --metrics_config_min_batch_index METRICS_CONFIG_MIN_BATCH_INDEX
                        Minimum batch index. (default: None)
  --metrics_config_max_batch_index METRICS_CONFIG_MAX_BATCH_INDEX
                        Maximum batch index. (default: None)
  --metrics_config_output_dir METRICS_CONFIG_OUTPUT_DIR
                        Output directory. (default: simulator_output)
  --metrics_config_cache_dir METRICS_CONFIG_CACHE_DIR
                        Cache directory. (default: cache)
(vidur) root@x08j01357:/app/software1/vidur# 