# ML-EnsembleHub 🚀
[中文](README.md) | [English](README.en.md)

基于scikit-learn，拖拽式配置机器学习实验，水论文工具 :)

## 先决条件 
- Python 3.x

## 安装 🛠
克隆存储库并安装依赖项,

```bash
git clone https://github.com/6ixGODD/ML-EnsembleHub.git
cd ML-EnsembleHub
pip install -r requirements.txt
```

## 使用 
使用命令行选项执行脚本,

```bash
python main.py  --data <path_to_data> 
                --cfg <path_to_config> 
                --save-dir <path_to_save_dir> 
                --name <name_of_experiment> 
                --save 
                --plot
```

- `--data` - 指定数据路径
- `--cfg` - 指定配置路径
- `--save-dir` - 设置保存结果的目录
- `--name` - 定义实验名称
- `--save` - 启用结果保存 (metrics, models)
- `--plot` - 启用plot

## 配置 
YAML 配置实验：

```yaml
shuffle: <bool>
random_state: <int>

preprocessing:
    method: <method_name>/null
    <method_name>:
        <param_name>: <param_value>

classifiers:
    method: 
      - <method1_name>
      - <method2_name>
      - ...
    <method1_name>:
        <param_name>: <param_value>
    <method2_name>:
        <param_name>: <param_value>
    ...

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
- `random_state` - 随机种子
- `preprocessing` - 预处理方法 / 禁用(null)
- `classifiers` - 分类器列表
- `feature_selection` - 特征选择方法 / 禁用(null)
- `model_selection` - model selection方法

## 数据集 
确保数据以 CSV 格式存储，其中一列指定为 `label`，其他列代表特征：

| label | feature1 | feature2 | ... |
|-------|----------|----------|-----|
| 0/1   | value1   | value2   | ... |
| ...   | ...      | ...      | ... |

## 示例 
使用命令运行示例,

```bash
python main.py  --data data/credit.csv --cfg configs/credit.yml --save-dir output --name credit --save --plot
```

## 结果 
- 指标：`output/<name_of_experiment>/metrics/metrics.csv`
- 图表：`output/<name_of_experiment>/plots/`    
- 模型：`output/<name_of_experiment>/models/`
- 日志：`output/<name_of_experiment>/log.txt`

:)