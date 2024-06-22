# vLLM
学习vLLM，使用vLLM部署Qwen2-0.5B的模型，并使用docker部署。


https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html
https://www.modelscope.cn/models/qwen/Qwen2-0.5B-Instruct-GPTQ-Int4
https://github.com/QwenLM/Qwen2?tab=readme-ov-file
https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4


# 环境

```commandline
python 3.8
cuda 12.5
pytorch #与cuda匹配的版本,这里选择与12.1的cuda匹配
transformers>=4.40.0

```

## 安装cuda
https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local


```commandline
wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42.02_linux.run
sudo sh cuda_12.5.0_555.42.02_linux.run
```

## 新建一个conda环境

```commandline
conda create -n qwen2 python=3.8
conda activate qwen2

```

## 安装pytorch
https://pytorch.org/get-started/locally/

```commandline
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple

```

## 安装其他依赖

```commandline
pip install auto-gptq optimum -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install modelscope  -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

```

## 运行jupyter
```commandline
# 前提：安装好jupyter
#进入新建环境安装ipykernel
conda activate {env_name}
conda update -n base -c conda-forge conda
pip3 install --default-timeout=60 ipykernel -i https://pypi.tuna.tsinghua.edu.cn/simple 
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ipykernel
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple ipywidgets

#写入环境
python -m ipykernel install --user --name 'qwen2' --display-name 'qwen2'



```

# jupyter中运行

```commandline
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2-0.5B-Instruct-GPTQ-Int4')

```

```commandline
from modelscope import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen2-0.5B-Instruct-GPTQ-Int4")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

```

# vLLM

## 安装

```commandline
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple

```


```commandline

python -m vllm.entrypoints.openai.api_server  \
    --port 8080 \
    --model /home/ubuntu/.cache/modelscope/hub/qwen/Qwen2-0___5B-Instruct-GPTQ-Int4 
        
curl http://localhost:8080/v1/models


curl http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/ubuntu/.cache/modelscope/hub/qwen/Qwen2-0___5B-Instruct-GPTQ-Int4",
        "prompt": "学习起来",
        "max_tokens": 7,
        "temperature": 0
    }'
    
    

curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/home/ubuntu/.cache/modelscope/hub/qwen/Qwen2-0___5B-Instruct-GPTQ-Int4",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }'





```

# vLLM + docker

https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html

因为网络原因，建议先把镜像下载下来，再传到服务器上。

```commandline
sudo docker pull vllm.tar vllm/vllm-openai:v0.4.0 

sudo docker save -o vllm.tar vllm/vllm-openai:v0.4.0 

sudo scp vllm.tar xxx@xxx.xxx.xxx:/home/ubuntu/...


sudo docker load < vllm.tar 


```

## 启动启动启动

```commandline
sudo docker run --runtime nvidia --gpus all \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    -p 8080:8080 \
    --ipc=host \
    vllm/vllm-openai:v0.4.0 \
    --model /root/.cache/modelscope/hub/qwen/Qwen2-0___5B-Instruct-GPTQ-Int4  --port=8080

```
```commandline

curl http://localhost:8080/v1/models

```

```commandline

curl http://localhost:8080/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/root/.cache/modelscope/hub/qwen/Qwen2-0___5B-Instruct-GPTQ-Int4",
        "prompt": "学习起来",
        "max_tokens": 7,
        "temperature": 0
    }'
 


```