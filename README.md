# ML-EnsembleHub 🚀

ML-EnsembleHub is an innovative and user-friendly platform designed to streamline the process of constructing and assessing machine learning models. Through a drag-and-drop approach, it enables users to effortlessly harness various classifiers, feature selection techniques, and model selection methods. 

## Prerequisites
- Python 3.x

## Installation 
Get started with ML-EnsembleHub by cloning the repository and installing the required dependencies:

```bash
git clone https://github.com/6ixGODD/ML-EnsembleHub.git
cd ML-EnsembleHub
pip install -r requirements.txt
```

## Usage
Execute the main script using the following command-line options:

```bash
python main.py  --data <path_to_data> 
                --cfg <path_to_config> 
                --save-dir <path_to_save_dir> 
                --name <name_of_experiment> 
                --save 
                --plot
```

- `--data` - Specify the path to your data
- `--cfg` - Indicate the path to your configuration
- `--save-dir` - Set the directory for saving results
- `--name` - Define a name for the experiment
- `--save` - Enable result saving
- `--plot` - Generate result plots

## Configuration
Utilize the YAML configuration to fine-tune experiments:

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

- `shuffle` - Enable data shuffling
- `random_state` - Define the random seed
- `preprocessing` - Choose a preprocessing method / disable preprocessing(null)
- `classifiers` - Configure the list of classifiers
- `feature_selection` - Select a feature selection method / disable feature selection(null)
- `model_selection` - Pick a model selection method

## Dataset 
Ensure data is in CSV format, where one column is specified as `label` and the other columns represent features:

| label | feature1 | feature2 | ... |
|-------|----------|----------|-----|
| 0/1   | value1   | value2   | ... |
| ...   | ...      | ...      | ... |

## Example 
Run an example experiment with the following command:

```bash
python main.py  --data data/credit.csv --cfg configs/credit.yml --save-dir output --name credit --save --plot
```

## Results 
- Metrics: `output/<name_of_experiment>/metrics/metrics.csv`
- Plots: `output/<name_of_experiment>/plots/`    
- Models: `output/<name_of_experiment>/models/`
- Logs: `output/<name_of_experiment>/log.txt`

:)