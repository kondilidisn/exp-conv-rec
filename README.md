This repository was made to support the experimental findings of the paper:
"Category Aware Explainable Conversational Recommendation" N. Kondylidis et al.
Accepted at the MICROS Workshop, ECIR 2021.
(https://arxiv.org/abs/2103.08733)

i) preprocess data

    git clone https://github.com/kondilidisn/exp-conv-rec.git
    cd exp-conv-rec
    mkdir Datasets
    mkdir Datasets/redial
    wget -O Datasets/redial/redial_dataset.zip https://github.com/ReDialData/website/raw/data/redial_dataset.zip
    mkdir Datasets/movielens
    wget -O Datasets/movielens/ml-latest.zip http://files.grouplens.org/datasets/movielens/ml-latest.zip
    python3 scripts/split-redial.py Datasets/redial
    unzip Datasets/movielens/ml-latest.zip
    mv ml-latest Datasets/movielens 

Match ReDial movies with Movielens movies
    
    python3 scripts/match_movies.py --redial_movies_path=Datasets/redial/movies_with_mentions.csv --ml_movies_path=Datasets/movielens/ml-latest/movies.csv --destination=Datasets/redial/movies_merged.csv

Create a file with all movies, their IDs, their categories etc.

    python3 scripts/create_movie_details_csv.py --movielens_path Datasets/movielens/ml-latest --redial_path Datasets/redial

[In the following commands, you can define the "batch size" which actually defines the number of conversations to be included in the same batch, by defining the parameter (--conversations_per_batch X), and in case of insufficient (GPU) memory, you can set the parameter --max_samples_per_gpu Y, to something small.]

ii) Train Category_Preference_model

    python3 train_cat_pref_model.py

iii) Train Item_Rec_model

Ours:

    python3 train_complete.py --reproduce ours

End-to-End:

    python3 train_complete.py --reproduce e2e

Ours-GT:
    
    python3 train_complete.py --reproduce ours_gt
