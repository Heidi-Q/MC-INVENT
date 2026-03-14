# MC-INVENT
Molecular generative model to design Macrocycle ligands. 

## Installation

1. Install [Conda](https://conda.io/projects/conda/en/latest/index.html)
2. Clone this Git repository
3. Open a shell, and go to the repository and create the Conda environment:
   
        $ conda env create -f mcinvent.yml

4. Activate the environment:

        $ conda activate mcinvent_env

5. Install in-house reinvent_scoring

        $ cd reinvent_scoring

        $ pip install reinvent_scoring-0.0.73_hq-py3-none-any.whl
		
		$ pip install reinvent_model-0.0.45_hq-py3-none-any.whl



## Usage
1. Edit template Json file (for example in CDK9/4bci_mcinvent_config.json).

   Templates can be manually edited before using. The only thing that needs modification for a standard run are the file and folder paths. Most running modes produce logs that can be monitored by tensorboard
2. Using the Json file path as the parameter of input.py.
   
        $ python input.py <config.json>
## Analyse the results
        
        $ tensorboard --logdir "progress.log"

        $ progress.log is the "logging_path" in template.json

## Data availability

The training data and pretrained model used in this study are provided in this repository.

Specifically, the following files can be found in the `CDK9` directory:

- `trainset.txt`: training dataset used to build the prior model
- `prior.model`: pretrained prior model

These files allow users to reproduce the training and molecular generation workflow described in this work.

