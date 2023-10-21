# ML-EnsembleHub 🚀

ML-EnsembleHub 是一款创新且用户友好的平台，旨在简化构建和评估机器学习模型的过程。通过基于拖放的方法，它使用户能够轻松地利用各种分类器、特征选择技术和模型选择方法。

## 先决条件 
- Python 3.x

## 安装 🛠
通过克隆存储库并安装所需依赖项来开始使用 ML-EnsembleHub：

```bash
git clone https://github.com/6ixGODD/ML-EnsembleHub.git
cd ML-EnsembleHub
pip install -r requirements.txt
```

## 使用 
使用以下命令行选项执行主要脚本：

```bash
python main.py  --data <path_to_data> 
                --cfg <path_to_config> 
                --save-dir <path_to_save_dir> 
                --name <name_of_experiment> 
                --save 
                --plot
```

- `--data` - 指定数据的路径
- `--cfg` - 指定配置的路径
- `--save-dir` - 设置保存结果的目录
- `--name` - 为实验定义一个名称
- `--save` - 启用结果保存
- `--plot` - 生成结果图表

## 配置 
利用 YAML 配置实验：

```yaml
shuffle: <bool>
random_state: <int>

preprocessing:
    method: <method_name>/null
    <method_name>:
        <param_name>: <param_value>

classifiers:
    method: [<method_name>, <method_name>, ...]
    <method_name>:
        <param_name>: <param_value>

feature_selection:
    method: <method_name>/null
    <method_name>:
        <param_name>: <param_value>

model_selection:
    method: <method_name>
    <method_name>:
        <param_name>: <param_value>
```

- `shuffle` - 启用数据混洗
- `random_state` - 定义随机种子
- `preprocessing` - 选择预处理方法 / 禁用预处理(null)
- `classifiers` - 配置分类器列表
- `feature_selection` - 选择特征选择方法 / 禁用特征选择(null)
- `model_selection` - 选择模型选择方法

## 数据集 
确保数据以 CSV 格式存储，其中一列指定为 `label`，其他列代表特征：

| label | feature1 | feature2 | ... |
|-------|----------|----------|-----|
| 0/1   | value1   | value2   | ... |
| ...   | ...      | ...      | ... |

## 示例 
使用以下命令运行示例实验：

```bash
python main.py  --data data/credit.csv --cfg configs/credit.yml --save-dir output --name credit --save --plot
```

## 结果 
- 指标：`output/<name_of_experiment>/metrics/metrics.csv`
- 图表：`output/<name_of_experiment>/plots/`    
- 模型：`output/<name_of_experiment>/models/`
- 日志：`output/<name_of_experiment>/log.txt`

:)