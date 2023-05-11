# ACT-R + CF

This repository hosts the code and the additional materials for the paper "Integrating the ACT-R Framework with
Collaborative Filtering for Explainable Sequential Music Recommendation" by xxx.

## Repository Structure

```bash
.
├── README.md
├── notebooks
└── actr_rs.yml
```

## Installation and configuration

Edit the variables in the `paths.py` file as follows:

- `BASE_FOLDER`: the main folder of the repository

### Environment

- Install the environment with
  `conda env create -f actr_rs.yml`
- Activate the environment with `conda activate actr_rs`

## Data preparation

### Listening events

Download the dataset from [Zenodo](https://zenodo.org/record/7923581#.ZFyRu5FBxkg). Then run the DatasetCreation
notebook. This will

- filter the [20-02-2020 -- 19-03-2020] month of the dataset
- remove users that listened to more tracks than 99% of the users
- apply 10 core 
- perform a 60-20-20 temporal split for each user
- create the dataset needed for training BPR, MultVAE, and GRU4Rec

## Run
