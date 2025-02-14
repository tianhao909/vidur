# How to add a new model to the simulator?如何向模拟器添加新模型？

## Structure of Profiling data 性能分析数据的结构

The profiling data is stored in the `data/profiling` directory. The profiling data is stored in CSV format. The profiling data is stored in the following format:
性能分析数据存储在 data/profiling 目录中。性能分析数据以 CSV 格式存储。性能分析数据的存储格式如下：
```yaml
    profiling/
        - compute
            - a100
                - codellama/CodeLlama-34b-Instruct-hf/
                    - mlp.csv
                    - attention.csv
                - internlm/internlm-20b/
                    - mlp.csv
                    - attention.csv
            - h100
                - meta-llama/Llama-2-70b-hf/
                    - mlp.csv
                    - attention.csv
        - network
            - a100_pair_nvlink
                - allreduce.csv
                - send_recv.csv
            - h100_dgx
                - allreduce.csv
                - send_recv.csv
```

For compute profiling, only the SKU matters not the network configuration of the node. So, we don't discriminate between `a100_pair_nvlink` (Azure Standard_NC96ads_A100_v4 with 4 A100s) and `a100_dgx` (A100 DGX with 8 A100s), the same compute data is used in both in the folder called `a100`.
对于计算性能分析，重要的是 SKU 而不是节点的网络配置。因此，我们不会区分 a100_pair_nvlink（Azure Standard_NC96ads_A100_v4，带 4 个 A100）和 a100_dgx（A100 DGX，带 8 个 A100），在名为 a100 的文件夹中使用相同的计算数据。

For network profiling, the network configuration of the node matters. So, we have separate folders for `a100_pair_nvlink` and `a100_dgx`. One example is that TP4 is different in these. In `a100_pair_nvlink`, there are two pairs connected by NVLink but between these pairs is a relatively slower link, but in `a100_dgx` where all 8 GPUs are connected by NVLink.
对于网络性能分析，节点的网络配置很重要。因此，我们为 a100_pair_nvlink 和 a100_dgx 设置了不同的文件夹。一个例子是，TP4 在这些配置中是不同的。在 a100_pair_nvlink 中，有两个通过 NVLink 连接的对，但这些对之间的连接相对较慢，而在 a100_dgx 中，所有 8 个 GPU 都通过 NVLink 连接。


## Adding a new model 添加新模型

We need actual GPUs to get profiling data for a new model. Once the profiling is done, simulations can be run on CPUs only.
我们需要实际的 GPU 来获取新模型的性能分析数据。一旦性能分析完成，模拟可以在仅使用 CPU 的情况下运行。

1. Clone the [`sarathi-serve`](https://github.com/microsoft/sarathi-serve) GitHub repo. 克隆 [`sarathi-serve`]
        git clone git@github.com:tianhao909/sarathi-serve.git
        fth git clone
        git branch 
        git fetch origin
        git branch -r
        git checkout -b vidur origin/vidur
        conda activate /app/software1/sarathi-serve_viudr/sarathi-serve/env
    a. Checkout branch [`vidur`](https://github.com/microsoft/sarathi-serve/tree/vidur) 检出分支 [`vidur`]
    b. Follow its README to install it. 按照其 README 安装。
    c. Let us assume that the Python virtual environment was created in `sarathi-serve/env`. 假设 Python 虚拟环境已在 `sarathi-serve/env` 中创建。

2. Now clone this repo [`vidur`](https://github.com/microsoft/vidur) but keep the `sarathi-serve/env` virtual environment activated. 现在克隆此仓库 [`vidur`](https://github.com/microsoft/vidur)，但保持 `sarathi-serve/env` 虚拟环境激活。
        conda activate /app/software1/sarathi-serve_viudr/sarathi-serve/env
        
3. Add a YAML model config for the new model in `data/model_configs`.  
    在 `data/model_configs` 中为新模型添加一个 YAML 模型配置。
    - Use the model's HuggingFace model id for the file name eg. `data/model_configs/meta-llama/Llama-2-70b-hf.yml`. 
        使用模型的 HuggingFace 模型 ID 作为文件名，例如 `data/model_configs/meta-llama/Llama-2-70b-hf.yml`。
        fth 
        https://huggingface.co/meta-llama/Llama-2-70b-hf
        https://hf-mirror.com/meta-llama/Llama-2-70b-hf


    - Refer HuggingFace `config.json` for the model eg. <https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json>. 
        参考 HuggingFace `config.json` 来配置模型，例如 <https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json>。

    - Ensure that correct parameters are set in the YAML file so that the reference transformer model [GPTModel](vidur/profiling/mlp/mlp_impl.py) closely resembles the new model.
        确保在 YAML 文件中设置了正确的参数，以便参考的 transformer 模型 [GPTModel](vidur/profiling/mlp/mlp_impl.py) 与新模型非常相似。
    - We use this reference model to profile only the MLP operations of all the models so the attention operations are no-op'ed here.
        我们使用这个参考模型仅分析所有模型的 MLP 操作，因此在这里注意力操作是无操作的。

4. Run the following command to install the simulator in the virtual environment: `python -m pip install -e .` from the `vidur/` directory.
    运行以下命令在虚拟环境中安装模拟器：`python -m pip install -e .` 从 `vidur/` 目录。

5. For compute profiling (mlp and attention), 1 GPU is enough even for tensor parallel degrees greater than 1. So `num_gpus` set to 1 is sufficient albeit slower for profiling.
    对于计算性能分析（mlp 和 attention），即使是对于大于 1 的张量并行度，1 个 GPU 就足够了。因此，设置 `num_gpus` 为 1 是足够的，尽管这样在性能分析时会慢一些。
    
6. Now we need to do the MLP profiling: 现在我们需要进行 MLP 性能分析：

    ```bash
        python vidur/profiling/mlp/main.py \
        --models codellama/CodeLlama-34b-Instruct-hf \
        --num_gpus 4

        fth

         python -m vidur.profiling.mlp.main \
        --models codellama/CodeLlama-34b-Instruct-hf \
        --num_gpus 4

        python -m vidur.profiling.mlp.main \
        --models codellama/CodeLlama-34b-Instruct-hf \
        --num_gpus 1

        python -m vidur.profiling.mlp.main \
        --models meta-llama/Llama-2-70b-hf \
        --num_gpus 1
    ```

    - Run `python vidur/profiling/mlp/main.py --help` for more options.
        运行 `python vidur/profiling/mlp/main.py --help` 以获取更多选项。
    - Copy the CSV file from `profiling_outputs/mlp/<timestamp>/codellama/CodeLlama-34b-Instruct-hf/mlp.csv` to `data/profiling/compute/a100/codellama/CodeLlama-34b-Instruct-hf/mlp.csv`.
        将 CSV 文件从 `profiling_outputs/mlp/<timestamp>/codellama/CodeLlama-34b-Instruct-hf/mlp.csv` 复制到 `data/profiling/compute/a100/codellama/CodeLlama-34b-Instruct-hf/mlp.csv`。
1. Now we need to do the attention profiling:
     现在我们需要进行注意力性能分析：
        pip install pandas -i https://mirrors.aliyun.com/pypi/simple/
        pip install ray -i https://mirrors.aliyun.com/pypi/simple/
        pip install -r require.txt -i https://mirrors.aliyun.com/pypi/simple/
        pip install tqdm -i https://mirrors.aliyun.com/pypi/simple/
        pip install sarathi -i https://mirrors.aliyun.com/pypi/simple/


        安装pytorch
        https://pytorch.org/

        https://pytorch.org/get-started/previous-versions/
        conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

    ```bash
        python vidur/profiling/attention/main.py \
        --models codellama/CodeLlama-34b-Instruct-hf \
        --num_gpus 4
    ```

    - Run `python vidur/profiling/attention/main.py --help` for more options.
        运行 `python vidur/profiling/attention/main.py --help` 以获取更多选项。
    - Copy the CSV file from `profiling_outputs/attention/<timestamp>/codellama/CodeLlama-34b-Instruct-hf/attention.csv` to `data/profiling/compute/a100/codellama/CodeLlama-34b-Instruct-hf/attention.csv`.
        将 CSV 文件从 `profiling_outputs/attention/<timestamp>/codellama/CodeLlama-34b-Instruct-hf/attention.csv` 复制到 `data/profiling/compute/a100/codellama/CodeLlama-34b-Instruct-hf/attention.csv`。
    - Note that we are using `a100` as the device name. If you are using `h100` or some other device, then you need to create a new folder for that device in `data/profiling/compute` and copy the CSV files there.
        注意，我们使用 `a100` 作为设备名称。如果您使用的是 `h100` 或其他设备，则需要在 `data/profiling/compute` 中为该设备创建一个新文件夹，并将 CSV 文件复制到那里。



## Network (Collectives) profiling 网络（集体操作）性能分析

Network profiling is not dependent on the model 🎉. So, we can use the same network profiling data for all models. However, we need to ensure that the network profiling data is available for the node configuration we are using. If not, then we need to profile the network for the device. 1.
网络性能分析不依赖于模型🎉。因此，我们可以使用相同的网络性能分析数据来分析所有模型。然而，我们需要确保我们使用的节点配置有可用的网络性能分析数据。如果没有，那么我们需要为设备进行网络性能分析。

For network profiling, the node setup i.e. type of connectivity between the gpus matter. This is why we have the concept of `network_device`. The network_device is an informal name for the network configuration of the node. Eg: `a100_pair_nvlink`, `a100_dgx`, `h100_dgx` etc.
对于网络性能分析，节点设置即 GPU 之间的连接类型很重要。这就是我们有 `network_device` 概念的原因。`network_device` 是节点网络配置的非正式名称。例如：`a100_pair_nvlink`、`a100_dgx`、`h100_dgx` 等。

    1. For tensor parallelism, 4 GPUs are needed for TP4 and 8 GPUs are needed for TP8 etc.
        对于张量并行，TP4 需要 4 个 GPU，TP8 需要 8 个 GPU 等。
    2. For pipeline parallelism across nodes, 2 nodes are needed to profile the link between the nodes.
        对于跨节点的流水线并行，需要 2 个节点来分析节点之间的连接。

Currently available data include: 目前可用的数据包括：

- `a100_pair_nvlink`: Azure Standard_NC96ads_A100_v4 with 4 80GB A100 PCIe GPUs with pair-wise NVLINK connectivity.
- `h100_pair_nvlink`: Azure internal VM with 4 80GB H100 NVL GPUs with pair-wise NVLINK connectivity.
- `a100_dgx`: A100 DGX with 8 80GB A100s.
- `h100_dgx`: H100 DGX with 8 H100s.


- `a100_pair_nvlink`：Azure Standard_NC96ads_A100_v4，带有 4 个 80GB A100 PCIe GPU，具有成对 NVLINK 连接。
- `h100_pair_nvlink`：Azure 内部 VM，带有 4 个 80GB H100 NVL GPU，具有成对 NVLINK 连接。
- `a100_dgx`：A100 DGX，带有 8 个 80GB A100。
- `h100_dgx`：H100 DGX，带有 8 个 H100。


### Steps to profile: 性能分析步骤：

1. Clone this (`vidur`) repo and create a Python virtual environment as in [Setup](README.md).
    克隆此（`vidur`）仓库，并按照[设置](README.md)中的方法创建 Python 虚拟环境。
2. Setup a ray cluster: 设置一个 ray 集群：
    a. Tensor parallelism is typically done on a single node so we don't need a multi-node cluster.
        张量并行通常在单个节点上完成，所以我们不需要多节点集群。
    b. However, pipeline parallelism is typically done across multiple nodes so we need at least 2 nodes there.
        然而，流水线并行通常跨越多个节点完成，所以我们至少需要 2 个节点。
    c. Run `ray start --head` from the root node. 从根节点运行 `ray start --head`。
    d. Run `ray start --address <head-node-ip>:<head-node-port>` from the other nodes. The other nodes also need to have the same git commit checked out.
        从其他节点运行 `ray start --address <head-node-ip>:<head-node-port>`。其他节点也需要检出相同的 git 提交。
3. Run the following command to profile for the `all_reduce` operation, (sufficient for TP):
    运行以下命令来分析 `all_reduce` 操作（对于 TP 足够）：

    ```bash
        python vidur/profiling/collectives/main.py \
        --num_workers_per_node_combinations 1,2,4,8 \
        --collective all_reduce
    ```

    ```bash
        python vidur/profiling/collectives/main.py \
        --num_workers_per_node_combinations 2 \
        --collective all_reduce
    ```
    - One may need to adjust `--num_workers_per_node_combinations` depending on the number of GPUs in the node eg. `--num_workers_per_node_combinations 1,2,4` for Azure Standard_NC96ads_A100_v4 node.
        根据节点中的 GPU 数量，可能需要调整 `--num_workers_per_node_combinations`，例如对于 Azure Standard_NC96ads_A100_v4 节点使用 `--num_workers_per_node_combinations 1,2,4`。
    - Copy the CSV file from `profiling_outputs/collectives/<timestamp>/all_reduce.csv` to `data/profiling/network/{network_device}/allreduce.csv`.
        将 CSV 文件从 `profiling_outputs/collectives/<timestamp>/all_reduce.csv` 复制到 `data/profiling/network/{network_device}/allreduce.csv`。
    - `network_device` is an informal name for the network configuration of the node. Eg: `a100_pair_nvlink`, `a100_dgx`, `h100_dgx` etc.
        `network_device` 是节点网络配置的非正式名称。例如：`a100_pair_nvlink`、`a100_dgx`、`h100_dgx` 等。
    - Run `python vidur/profiling/collectives/main.py --help` for more options.
        运行 `python vidur/profiling/collectives/main.py --help` 以获取更多选项。



1. Run the following command to profile for the `send_recv` operation, (required only for PP):
    运行以下命令来分析 `send_recv` 操作（仅 PP 需要）：

    ```bash
        python vidur/profiling/collectives/main.py \
        --num_workers_per_node_combinations 1,2,4,8 \
        --collective send_recv
    ```

    ```bash
    fth
        python /app/software1/vidur/vidur/profiling/collectives/main.py \
        --num_workers_per_node_combinations 1,2,4,8 \
        --collective send_recv

        python /app/software1/vidur/vidur/profiling/collectives/main.py \
        --num_workers_per_node_combinations 2 \
        --collective send_recv

        python /app/software1/vidur/vidur/profiling/collectives/main.py \
        --num_workers_per_node_combinations 1 \
        --collective send_recv
    ```

    - Typically, PP is done across nodes so `num_workers_per_node_combinations` should be the same as the number of GPUs available in one node. Profiling `num_workers_per_node_combinations` less than the number of GPUs in the node to have PP inside a node. This can be useful when each gpu is not connected to every other gpu using the same high speed link.
        通常，PP 是跨节点完成的，所以 `num_workers_per_node_combinations` 应该与单个节点中可用的 GPU 数量相同。分析 `num_workers_per_node_combinations` 小于节点中 GPU 数量的 PP 在节点内部。当每个 GPU 不是通过相同的高速链接连接到每个其他 GPU 时，这可能会很有用。
    - Copy the CSV file from `profiling_outputs/collectives/<timestamp>/send_recv.csv` to `data/profiling/network/{network_device}/send_recv.csv`.
        将 CSV 文件从 `profiling_outputs/collectives/<timestamp>/send_recv.csv` 复制到 `data/profiling/network/{network_device}/send_recv.csv`。
    - `network_device` is an informal name for the network configuration of the node. Eg: `a100_pair_nvlink`, `a100_dgx`, `h100_dgx` etc.
        `network_device` 是节点网络配置的非正式名称。例如：`a100_pair_nvlink`、`a100_dgx`、`h100_dgx` 等。



## CPU Overhead Profiling  CPU 开销性能分析

These include implementation overheads like scheduling time, sampling time, detokenization etc. For better fidelity, these should also be profiled. However, they tie the simulator closely to the implementation eg. `vLLM`. Scripts are available [here](vidur/profiling/cpu_overhead/) but not documented yet. These scripts follow a similar pattern to the compute and network profiling scripts.
这些包括实现开销，如调度时间、采样时间、去标记化等。为了更好的保真度，这些也应该进行性能分析。然而，它们将模拟器与实现紧密绑定，例如 `vLLM`。脚本可在[此处](vidur/profiling/cpu_overhead/)找到，但尚未文档化。这些脚本遵循与计算和网络性能分析脚本类似的模式。





