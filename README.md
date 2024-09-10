# ML-Tools 

## Prerequisites
- Python 3.x

## Installation 
Clone the repository and install the dependencies,

```bash
git clone https://github.com/6ixGODD/ml-tools.git
cd ml-tools
pip install -r requirements.txt
```

## Usage
Execute the script with command line options,

```bash
python main.py  --data <path_to_data> 
                --cfg <path_to_config> 
                --save-dir <path_to_save_dir> 
                --name <name_of_experiment> 
                --save 
                --plot
```
in english:
- `--data` - specify the path to the data
- `--cfg` - specify the path to the config
- `--save-dir` - set the directory to save the results
- `--name` - define the name of the experiment
- `--save` - enable result saving (metrics, models)
- `--plot` - enable plot

## Configuration
YAML configure the experiment:

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

- `shuffle` - shuffle the data
- `random_state` - random state
- `preprocessing` - preprocessing method / disable(null)
- `classifiers` - list of classifiers
- `feature_selection` - feature selection method / disable(null)
- `model_selection` - model selection method

## DataSet
The data must be in the following format:

| label | feature1 | feature2 | ... |
|-------|----------|----------|-----|
| 0/1   | value1   | value2   | ... |
| ...   | ...      | ...      | ... |

## Example
Execute the script with command line options,

```bash
python main.py  --data data/credit.csv --cfg configs/credit.yml --save-dir output --name credit --save --plot
```

## Result
- metrics：`output/<name_of_experiment>/metrics/metrics.csv`
- plots：`output/<name_of_experiment>/plots/`    
- models：`output/<name_of_experiment>/models/`
- log：`output/<name_of_experiment>/log.txt`

:)
