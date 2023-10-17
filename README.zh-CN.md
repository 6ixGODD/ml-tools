# ML-EnsembleHub
## 安装
- Python 3.x
```bash
git clone https://github.com/6ixGODD/ML-EnsembleHub.git
cd ML-EnsembleHub
pip install -r requirements.txt
```
## 使用
```bash
python main.py  --data <path_to_data> 
                --cfg <path_to_config> 
                --save-dir <path_to_save_dir> 
                --name <name_of_experiment> 
                --save 
                --plot
```
- `--data` - 数据路径
- `--cfg` - 配置路径
- `--save-dir` - 保存路径
- `--name` - 实验名称
- `--save` - 保存结果
- `--plot` - 可视化结果

## 配置
```yaml
shuffle: <bool>
random_state: <int>

preprocessing:
    method: <method_name>
    <method_name>:
        <param_name>: <param_value>

classifiers:
    method: [<method_name>, <method_name>, ...]
    <method_name>:
        <param_name>: <param_value>

feature_selection:
    method: <method_name>
    <method_name>:
        <param_name>: <param_value>

model_selection:
    method: <method_name>
    <method_name>:
        <param_name>: <param_value>
```
- `shuffle` - 是否打乱数据
- `random_state` - 随机种子
- `preprocessing` - 预处理方法
- `classifiers` - 分类方法
- `feature_selection` - 特征选择方法
- `model_selection` - 采样方法

## Dataset
- 数据应该是csv格式，其中第一列是`label`，其他列是特征

    | label | feature1 | feature2 | ... |
    |-------|----------|----------|-----|
    | 0/1   | value1   | value2   | ... |
    | ...   | ...      | ...      | ... |



## 示例
```bash
python main.py  --data data/iris.csv 
                --cfg config/iris.yaml 
                --save-dir exp
                --name iris 
                --save 
                --plot
```

## 结果
- 评估指标结果: `output/<name_of_experiment>/metrics/metrics.csv`
- 评估指标可视化: `output/<name_of_experiment>/plots/`    
- 模型文件: `output/<name_of_experiment>/models/`
- 日志: `output/<name_of_experiment>/log.txt`

