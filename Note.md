# 创建conda虚拟环境（后面跟随同时下载包的名字）
conda create -n hateSpeech python=3.10 numpy pandas matplotlib pytorch tensorflow transformers

# 激活
conda activate my_env

# 退出
conda deactivate

# 删除
conda remove -n my_env --all

# 查看目前包
conda list

## 包下载失败原因可能有
# 1. 渠道里没有这个包，可尝试：增加conda-forge渠道（或者别的，请搜）
conda config --add channels conda-forge
conda config --set channel_priority strict

# 2. 包版本冲突