# Understanding metrics logged by the simulator
理解模拟器记录的指标

## Preliminaries

For every request, we define the following key metrics:对于每个请求，我们定义了以下关键指标：

1. Request arrival time ($a_r$): the time at which a request enters the system
2. Request schedule time ($s_r$): the time at which a given request is scheduled for the first time (irrespective of subsequent restarts).
3. Request completion time ($c_r$): the time at which a request completes.
4. Request prefill completion time ($f_r$): the time at which prefill completes and first output token is produced.
5. Request execution time ($e_r$): the total amount of time a request spends actually executing on GPUs (across all attempts) - excluding the time request is allocated on a replica but not executing due to pipeline-bubbles etc.
6. Request preemption time ($p_r$): the total amount of time a request spends request is allocated on a replica but not executing due to pipeline-bubbles, scheduling preemptions, time between restarts, etc (aggregate across all attempts).
7. Request scheduling delay ($d_r$): the total amount for which the request is waiting before getting scheduled ($s_r - a_r$).

Note that arrival, schedule and completion time refer to a specific point in time, where as, execution, preemption time, scheduling delay refer to period of time.

初步说明
对于每个请求，我们定义了以下关键指标：

请求到达时间（$a_r$）：请求进入系统的时间。
请求调度时间（$s_r$）：给定请求第一次被调度的时间（不考虑后续的重启）。
请求完成时间（$c_r$）：请求完成的时间。
请求预填充完成时间（$f_r$）：预填充完成并产生第一个输出令牌的时间。
请求执行时间（$e_r$）：请求在 GPU 上实际执行的总时间（跨越所有尝试） - 不包括请求分配到副本上但不执行的时间，由于管道泡沫等原因。
请求抢占时间（$p_r$）：请求分配到副本上但不执行的总时间，由于管道泡沫、调度抢占、重启之间的时间等（跨所有尝试累积）。
请求调度延迟（$d_r$）：请求在被调度前等待的总时间（$s_r - a_r$）。
请注意，到达、调度和完成时间指的是特定时间点，而执行、抢占时间和调度延迟指的是时间段。

## Logged Metics

1. `request_inter_arrival_delay_histogram`: Histogram of difference between arrival times of adjacent requests ($a_{r+1} - a_r$).
2. `request_num_tokens_histogram`: Histogram of number of tokens (prefill + decode) across all requests.
3. `request_num_restarts_histogram`: Histogram of number of restarts for a given request. Note that this is expected to be a non-zero entity only when using vLLM or dSararthi schedulers - which restart requests in case a replica runs out of memory.
4. `request_e2e_time_cdf`: CDF of end-to-end request latency ($c_r - a_r$).
5. `request_e2e_time_normalised_cdf`: CDF of end-to-end request latency normalised by number of output tokens.
6. `request_execution_plus_preemption_times_cdf`: CDF of total time a request spends in the system excluding initial scheduling delay ($c_r - s_r$).
7. `request_scheduling_delay_cdf`: CDF of request scheduling delay ($s_r - a_r$).
8. `request_execution_time_cdf`: CDF of request execution time ($e_r$).
9. `request_preempted_time_cdf`: CDF of request preemption time ($p_r$).
10. `decode_token_execution_plus_preemption_times`: CDF of per decode token execution time and preemption time - i.e. inter-token delay observed by the user.
11. `batch_num_tokens_cdf`: CDF of total number of tokens to be processed in a batch (sum of prefill tokens + one per decode request). This distribution is useful towards understanding how the compute load is distributed across batches. Note that with iteration level scheduling a batch is formed at every iteration.
12. `batch_sizes_cdf`: CDF of batch sizes - usually larger batch sizes imply higher throughput.
13. `prefill_time_e2e_cdf`: CDF of end-to-end latency to the first output token (time-to-first-byte), i.e, time elapsed since the request arrival to the point where first output is generated ($f_r - a_r$).
14. `prefill_time_execution_plus_preemption_cdf`: CDF of total prefill process time excluding the initial scheduling delay ($f_r - s_r$). This metric is useful for tracking the prefill efficiency.
15. `prefill_time_execution_plus_preemption_normalized_cdf`: Similar to `prefill_time_execution_plus_preemption_cdf`, but normalized by the number of prefill tokens. This provides distribution independent of request prefill length, and thus, easier to analyze.
16. `decode_time_execution_plus_preemption_normalized_cdf`: CDF of total time spent processing decodes ($c_r - f_r$) normalized by the number of decode tokens. This provides an indicator similar to `decode_token_execution_plus_preemption_times`, however, this metric is presents an averaged over all decode tokens in the request.
17. `request_completions_time_series`: Time series of request completion times - this provides an indicator for makespan and helps in identifying the request processing rate (requests per second) by analyzing the slope of the curve.
18. `prefill_completions_time_series`: Time series of prefill token completion times - helps in identifying the prefill processing rate (prefill tokens per second) by analyzing the slope of the curve.
19. `decode_completions_time_series`: Time series of decode  completion times - helps in identifying the decode processing rate (decode tokens per second) by analyzing the slope of the curve.
20. `replica_{replica_id}_memory_usage_weighted_mean`: Memory usage statistics per replica-level - tracks the mean utilization value across entire execution time.
21. `replica_{replica_id}_stage_{stage_id}_busy_time_percent_weighted_mean`: Percentage of time a given replica stage is executing something on device - i.e. not waiting due to scheduling issues or pipeline bubbles.
22. `replica_{replica_id}_stage_{stage_id}_mfu_weighted_mean`: Model FLOPS Utilization (MFU) at a per replica stage level - it tell how much value we are able to extract from the hardware. MFU increases with batch size, reduced bubble time, higher prefill tokens, etc.
23. `request_arrivals_time_series`: Time series of request arrival timestamps.


记录的指标

1. `request_inter_arrival_delay_histogram`：相邻请求到达时间差（$a_{r+1} - a_r$）的直方图。
2. `request_num_tokens_histogram`：所有请求中令牌数量（预填充 + 解码）的直方图。
3. `request_num_restarts_histogram`：给定请求重启次数的直方图。注意，这只有在使用 vLLM 或 dSararthi 调度器时才会是非零实体 - 这些调度器在副本内存不足时会重启请求。
4. `request_e2e_time_cdf`：端到端请求延迟（$c_r - a_r$）的累积分布函数（CDF）。
5. `request_e2e_time_normalised_cdf`：按输出令牌数量归一化的端到端请求延迟的累积分布函数（CDF）。
6. `request_execution_plus_preemption_times_cdf`：请求在系统中花费的总时间，不包括初始调度延迟（$c_r - s_r$）的累积分布函数（CDF）。
7. `request_scheduling_delay_cdf`：请求调度延迟（$s_r - a_r$）的累积分布函数（CDF）。
8. `request_execution_time_cdf`：请求执行时间（$e_r$）的累积分布函数（CDF）。
9. `request_preempted_time_cdf`：请求抢占时间（$p_r$）的累积分布函数（CDF）。
10. `decode_token_execution_plus_preemption_times`：每个解码令牌执行时间和抢占时间的累积分布函数 - 即用户观察到的令牌间延迟。
11. `batch_num_tokens_cdf`：一批中要处理的总令牌数量（预填充令牌总数 + 每个解码请求一个）的累积分布函数。这个分布有助于理解计算负载如何在批次之间分布。注意，每次迭代都会形成一个新的批次。
12. `batch_sizes_cdf`：批次大小的累积分布函数 - 通常较大的批次大小意味着更高的吞吐量。
13. `prefill_time_e2e_cdf`：第一个输出令牌（时间到第一个字节）的端到端延迟的累积分布函数，即从请求到达至第一个输出生成的时间（$f_r - a_r$）。
14. `prefill_time_execution_plus_preemption_cdf`：不包括初始调度延迟的总预填充处理时间（$f_r - s_r$）的累积分布函数。这个指标有助于跟踪预填充效率。
15. `prefill_time_execution_plus_preemption_normalized_cdf`：类似于 `prefill_time_execution_plus_preemption_cdf`，但按预填充令牌数量归一化。这提供了与请求预填充长度无关的分布，因此更容易分析。
16. `decode_time_execution_plus_preemption_normalized_cdf`：按解码令牌数量归一化的总解码处理时间（$c_r - f_r$）的累积分布函数。这提供了与 `decode_token_execution_plus_preemption_times` 类似的指标，但是这个指标是所有解码令牌的平均值。
17. `request_completions_time_series`：请求完成时间的时间序列 - 这提供了对完成时间的指标，并通过对曲线斜率的分析帮助识别请求处理率（每秒请求数）。
18. `prefill_completions_time_series`：预填充令牌完成时间的时间序列 - 通过对曲线斜率的分析帮助识别预填充处理率（每秒预填充令牌数）。
19. `decode_completions_time_series`：解码完成时间的时间序列 - 通过对曲线斜率的分析帮助识别解码处理率（每秒解码令牌数）。
20. `replica_{replica_id}_memory_usage_weighted_mean`：每个副本级别的内存使用统计 - 跟踪整个执行时间内的平均利用率值。
21. `replica_{replica_id}_stage_{stage_id}_busy_time_percent_weighted_mean`：给定副本阶段在设备上执行的时间百分比 - 即不因调度问题或管道泡沫而等待的时间。
22. `replica_{replica_id}_stage_{stage_id}_mfu_weighted_mean`：每个副本阶段的模型 FLOPS 利用率（MFU） - 它告诉我们能够从硬件中提取多少价值。MFU 随着批次大小的增加、泡沫时间的减少、预填充令牌的增加而增加等。
23. `request_arrivals_time_series`：请求到达时间戳的时间序列。
