# price-prediction
House price prediction

# Introduction
In this repository, different types of regression models are implemented to predict house pirces.

The dataset used is the `House Sales in King County, USA` dataset which can be found [here](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).

# Repo structure
```
price-prediction
├─ price-prediction
│  ├─ experimentation
│  │  ├─ features.py
│  │  ├─ model.py
│  │  ├─ train_mlr.py
│  │  ├─ train_sr.py
│  │  └─ utils.py
│  └─ notebooks
│     ├─ EDA.ipynb
│     └─ utils_eda.py
├─ .gitignore
├─ poetry.lock
├─ pyproject.toml
└─ README.md
```

# Files description
- `price-prediction/experimentation/features.py`: Contains code for feature pre-processing.
- `price-prediction/experimentation/model.py`: Contains code for creating and fitting the different models.
- `price-prediction/experimentation/train_mlr.py`: Contains code for training the multiple linear regression model.
- `price-prediction/experimentation/train_sr.py`: Contains code for training the single regression models, i.e. linear regression and polynomial regression.
- `price-prediction/experimentation/utils.py`: Contains utility functions.
- `price-prediction/notebooks/EDA.ipynb`: Jupyter notebook for exploratory data analysis and dataset pre-processing, i.e. duplicates removal, outlier detection etc.
- `price-prediction/notebooks/utils_eda.py`: Contains utility functions for the EDA notebook.

# Installation
In order to install the dependencies via `poetry`:
- `poetry` needs to be installed.
- The following command should be run, inside the directory where the `pyproject.toml` file is located: `poetry install --no-root`

# Execution

### Execution of the single regression models experimentation:
This can be done by activating the virtual environment and running the following command:
``` bash
poetry env info --path  #In order to get the path of the virtual environment.
source path-to-virtual-environment/bin/activate  #In order to activate the virtual environment.
python train_sr.py --dataset_path /path/to/dataset.csv  #In order to run the script.
```

An alternative way to run the script is to use `poetry` directly, without activating the virtual environment. This can be done by running the following command:
```bash
poetry run train_sr.py --dataset_path /path/to/dataset.csv
```

After running the script, the results and the output files will be saved in the `price-prediction/experimentation/output_sr` directory.

### Execution of the multiple linear regression model experimentation:
This can be done by activating the virtual environment and running the experimentation script. The commands are the same as for the single regression models, but the script name is different. The command to run the script is:
``` bash
python train_mlr.py --dataset_path /path/to/dataset.csv  #In order to run the script.
```

Again, an alternative way to run the script is to use `poetry` directly. This can be done by running the following command:
```bash
poetry run train_mlr.py --dataset_path /path/to/dataset.csv
```

After running the script, the results and the output files will be saved in the `price-prediction/experimentation/output_mlr` directory.

