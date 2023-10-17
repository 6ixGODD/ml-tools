# ML-EnsembleHub
## Installation
- Python 3.x
```bash
cd ML-EnsembleHub
pip install -r requirements.txt
```
## Usage
```bash
python main.py  --data <path_to_data> 
                --cfg <path_to_config> 
                --save-dir <path_to_save_dir> 
                --name <name_of_experiment> 
                --save 
                --plot
```
- `--data` - path to data
- `--cfg` - path to config
- `--save-dir` - path to save dir
- `--name` - name of experiment
- `--save` - save results
- `--plot` - plot results

## Config
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
- `shuffle` - shuffle data
- `random_state` - random state
- `preprocessing` - preprocessing method
- `classifiers` - list of classifiers
- `feature_selection` - feature selection method
- `model_selection` - model selection method

## Dataset
- data should be in csv format, where first column is `label` and other columns are features

    | label | feature1 | feature2 | ... |
    |-------|----------|----------|-----|
    | 0/1   | value1   | value2   | ... |
    | ...   | ...      | ...      | ... |



## Example
```bash
python main.py  --data data/iris.csv 
                --cfg config/iris.yaml 
                --save-dir exp
                --name iris 
                --save 
                --plot
```

## Results
- metrics: `output/<name_of_experiment>/metrics/metrics.csv`
- plots: `output/<name_of_experiment>/plots/`    
- models: `output/<name_of_experiment>/models/`
- logs: `output/<name_of_experiment>/log.txt`

