# ACT-R + CF

This repository hosts the code and the additional materials for the paper "Integrating the ACT-R Framework with
Collaborative Filtering for Explainable Sequential Music Recommendation" by Marta Moscati, Christian Wallmann, Markus
Reiter-Haas, Dominik Kowald, Elisabeth Lex, and Markus Schedl.

You can cite this work as follows:

```
@inproceedings{placeholder,
  author = {Marta Moscati and
  Christian Wallmann and
  Markus Reiter-Haas and
  Dominik Kowald and
  Elisabeth Lex and
  Markus Schedl},
  title = {Integrating the ACT-R Framework with Collaborative Filtering for Explainable Sequential Music Recommendation},
  booktitle = {Proceedings of the 17th {ACM} Conference on Recommender Systems, Singapore, September 18-22, 2023},
  publisher = {{ACM}},
  year = {2023},}
}
```

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
- apply 10 core filtering
- perform a 60-20-20 temporal split for each user
- create the dataset needed for training BPR, MultVAE, and GRU4Rec

## Run

## Acknowledgment

This research was funded in whole, or in part, by the Austrian Science Funds (FWF): P33526 and DFH-23, and by the State
of Upper Austria and the Federal Ministry of Education, Science, and Research, through grant LIT-2020-9-SEE-113.