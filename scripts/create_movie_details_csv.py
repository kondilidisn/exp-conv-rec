import os
import csv
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--movielens_path", default = "Datasets/movielens/ml-latest", type=str)
parser.add_argument("--redial_path", default = "Datasets/redial", type=str)
args = parser.parse_args()



dataset_path = args.movielens_path

movies_filename = os.path.join(dataset_path, "movies.csv")

# maintain a global set of all categories, we will need its length later on
all_gernes = set()

movie_lens_ID_to_gerne_list = {}
movie_lens_ID_to_name = {}

with open(movies_filename, 'r') as f:
    reader = csv.reader(f)
    # movieId,title,genres
    for row in reader:
        # skip first line of csv
        if row[0] == 'movieId':
            continue
        movie_lens_id = int(row[0])
        movie_name = row[1]
        gernes = row[2].split("|")
        all_gernes = all_gernes.union(set(gernes))
        movie_lens_ID_to_gerne_list[movie_lens_id] = gernes
        movie_lens_ID_to_name[movie_lens_id] = movie_name

# replacing all movied that do not have a gerne with a list of all gernes
# like a uniform distribution over gernes
standard_template = all_gernes.copy()

standard_template.remove("(no genres listed)")

standard_template = list(standard_template)

no_category_for_movie_counter = 0

movie_lens_movie_ids_missing_category_info = set()


for movie_id in movie_lens_ID_to_gerne_list:
    if movie_lens_ID_to_gerne_list[movie_id] == ["(no genres listed)"]:
        movie_lens_ID_to_gerne_list[movie_id] = standard_template
        no_category_for_movie_counter += 1
        movie_lens_movie_ids_missing_category_info.add(movie_id)


print('MovieLens : Total number of movies :', len(movie_lens_ID_to_gerne_list))
print("MovieLens : Total movies with missing category information :", no_category_for_movie_counter)

index_to_redial_id = {}

index_to_movie_lens_id = {}
movie_lens_id_to_index = {}

index_to_database_id = {}
database_id_to_index = {}

index_to_name = {}

indexes = []

redial_movie_counter = 0

movies_merged_file = os.path.join(args.redial_path, "movies_merged.csv")

with open(movies_merged_file, 'r') as f:
    reader = csv.reader(f)
    # index,movieName,databaseId,movielensId
    for row in reader:
        # skip first line of csv
        if row[0] == "index":
            continue

        index = int(row[0])
        movie_name = row[1]
        database_id = int(row[2])
        movie_lens_id = int (row[3])

        if database_id == -1:
            redial_id = -1
        else:
            redial_id = index
            redial_movie_counter += 1

        indexes.append(index)

        index_to_redial_id[index] = redial_id

        index_to_movie_lens_id[index] = movie_lens_id
        if movie_lens_id != -1:
            movie_lens_id_to_index[movie_lens_id] = index

        index_to_database_id[index] = database_id
        if database_id != -1:
            database_id_to_index[database_id] = index

        index_to_name[index] = movie_name

index_to_gernes = {}

redial_movies_without_category_infromation_counter = 0

# now, for every movie in the database, we retrieve its gernes,
# if the movie is not in the dataset we use the standard template instead
for index in indexes:
    movie_lens_id = index_to_movie_lens_id[index]
    if movie_lens_id == -1:
        index_to_gernes[index] = standard_template

        redial_movies_without_category_infromation_counter += 1
    else:
        index_to_gernes[index] = movie_lens_ID_to_gerne_list[movie_lens_id]

# After we have retrieved the set of gernes for each movie (or have generated a set with all gernes)
# we need to create a binary vector for each movie, of size equal to the number of gernes

# save the dimension of each gerne, according to their index on standard_template
gerne_to_dimension = {}
for i, gerne in enumerate(standard_template):
    gerne_to_dimension[gerne] = i
# save the total number of gernes
num_of_gernes = len(standard_template)

# make a dictionary that maps movie ids to the category vector of the movie

index_to_category_vector = {}

# for each movie in the dataset
# for database_id in database_id_to_gernes:
for index in indexes:

    # initialize the category vector with zeros
    numpy_vector = np.zeros(num_of_gernes)
    # idicate the gernes of the movie to the appropriate dimension of the vector
    for gerne in index_to_gernes[index]:
        numpy_vector[gerne_to_dimension[gerne]] = 1
    # save the vector to the dictionary
    index_to_category_vector[index] = numpy_vector

# save generated info and corralation among datasets to a csv file in the form :
# Database_id, Redial_id MovieLens_Id, Movie_Title, List_of_Gernes [ the value of the last column will be the gerne vector ]

# redial_movies_without_category_infromation_counter = 0

# redial_movies_without_category_infromation_counter = 0
movie_lens_movies_counter = 0


with open( os.path.join(args.redial_path ,"movie_details.csv") , 'w') as writeFile:
    writer = csv.writer(writeFile)

    writer.writerow(["DataBase_id", "Redial_id", "Movie_Lens_id", "Movie_Title", str(standard_template)])   
    # for each movie
    for index in indexes:

        
        redial_id = index_to_redial_id[index]
        database_id = index_to_database_id[index]
        movie_lens_id = index_to_movie_lens_id[index]
        movie_title = index_to_name[index]
        gerne_vector = index_to_category_vector[index]

        if movie_lens_id != -1:
            movie_lens_movies_counter += 1

        writer.writerow([str(database_id), str(redial_id), str(movie_lens_id), movie_title, str(gerne_vector) ])

print("Total ReDial movies:", redial_movie_counter)
print("Total ReDial movies missing category info :",redial_movies_without_category_infromation_counter)

