import os
from tqdm import tqdm
import numpy as np
import csv
import torch
from torch.autograd import Variable
import random

from collections import defaultdict

import config


def load_movies(path):
    # path = os.getcwd() + path
    with open(path, 'r') as f:
        reader = csv.reader(f)
        movies = {row[0]: row[1] for row in reader if row[0] != "movieId"}
    return movies


def load_movies_merged(path):
    # path = os.getcwd() + path
    with open(path, 'r') as f:
        reader = csv.reader(f)
        id2index = {row[3]: int(row[0]) for row in reader if row[0] != "index"}
    return id2index


def process_rating(rating, ratings01=False):
    if ratings01:
        # return 1 for ratings >= 2.5, 0 for lower ratings (this gives 87% of liked on movielens-latest)
        # return 1 for ratings >= 2, 0 for lower ratings (this gives 94% of liked on movielens-latest)
        return float(rating) >= 2
    # return a rating between 0 and 1
    return (float(rating) - .5) / 4.5


def load_ratings(path, as_array=True):
    """
    One data example per user.
    :param path:
    :param as_array:
    :return: if as_array = False, return a dictionary {userId: {movieId: rating}}
    otherwise, return an array [{movieId: rating}] where each element corresponds to one user.
    """
    # path = os.getcwd() + path

    data = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for userId, movieId, rating, timestamp in tqdm(reader):
            if userId != "userId":  # avoid the first row
                if userId not in data:
                    data[userId] = {}
                data[userId][movieId] = rating
    if as_array:
        return data.values()
    return data


class MlBatchLoader_with_categories(object):
    """
    Loads Movielens data
    """

    def __init__(self,
                 batch_size,
                 movie_path=config.MOVIE_PATH,
                 data_path=config.ML_DATA_PATH,
                 train_path=config.ML_TRAIN_PATH,
                 valid_path=config.ML_VALID_PATH,
                 test_path=config.ML_TEST_PATH,
                 ratings01=False
                 ):
        self.batch_size = batch_size
        self.movie_path = movie_path
        self.batch_index = {"train": 0, "valid": 0, "test": 0}
        self.data_path = {
            "train": os.path.join(data_path, train_path),
            "valid": os.path.join(data_path, valid_path),
            "test": os.path.join(data_path, test_path)
        }
        self.ratings01 = ratings01

        self.load_category_vectors()

        self.load_data()

        self.build_category_data()

        self.n_batches = {key: len(val) // self.batch_size for key, val in self.ratings.items()}

    def build_category_data(self):

        def softmax(np_array):
            return np.exp(np_array)/sum(np.exp(np_array))



        self.cat_data = {"train": {} , "valid": {}, "test": {}}
        self.number_of_ratings_per_user = {"train": defaultdict(int) , "valid": defaultdict(int), "test": defaultdict(int)}

        for subset in ["train", "valid", "test"]:

            subset_ratings = self.ratings[subset]

            for user_id in subset_ratings:

                # initializing the user category vector
                self.cat_data[subset][user_id] = np.zeros(len(self.categories))


                for movie_lens_id in subset_ratings[user_id]:

                    rating = subset_ratings[user_id][movie_lens_id]

                    movie_category_vector = self.index_to_category_vector[ self.movie_lens_id_to_index[int(movie_lens_id)] ]

                    # translate str rating to binary int rating
                    sentiment = 1 if float(rating) >= 2 else -1

                    self.cat_data[subset][user_id] += sentiment * movie_category_vector

                    self.number_of_ratings_per_user[subset][user_id] += 1


        # add train subset information to other subsets
        for subset in ["valid", "test"]:
            # for every user in the 
            for user_id in self.cat_data[subset]:
                # we exploit any given infromation of that user from the training set
                if user_id in self.cat_data["train"]:

                    self.cat_data[subset][user_id] += self.cat_data["train"][user_id]

                    self.number_of_ratings_per_user[subset][user_id] += self.number_of_ratings_per_user["train"][user_id]


        # normalize category vectors per user, and apply softmax
        for subset in ["train", "valid", "test"]:

            for user_id in self.cat_data[subset]:
                # we first normalize the vector to avoid overflowing errors
                user_category_vector = self.cat_data[subset][user_id] / self.number_of_ratings_per_user[subset][user_id]
                # then we apply softax, so that it is a probability distribution
                self.cat_data[subset][user_id] = softmax(user_category_vector)






    def load_data(self):
        # self.id2movies = load_movies(self.movie_path)
        # self.id2index = {id: i for (i, id) in enumerate(self.id2movies)}
        self.id2index = load_movies_merged(self.movie_path)

        # print(max(self.id2index.values()))
        self.n_movies = max(self.id2index.values()) + 1
        print("Loading movie ratings from {}".format(self.data_path))

        # del self.data_path["valid"]
        # del self.data_path["test"]

        self.ratings = {subset: load_ratings(path, as_array=False)
                        for subset, path in self.data_path.items()}

        # returns a dictionary in the form [user_id] => [rated movies], for each subset

        # we need to make a dictionary of category vectors, for each dataset, w.r.t. user_id

                        
        # self.ratings["valid"] = self.ratings["train"]
        # self.ratings["test"] = self.ratings["train"]
        # print(self.ratings)
        # exit()
        # list of userIds for each subset
        self.keys = {subset: ratings.keys() for subset, ratings in self.ratings.items()}

    def load_batch(self, subset="train", batch_input="full", max_num_inputs=None, ratings01=None):
        if batch_input == 'random_noise' and max_num_inputs is None:
            raise ValueError("batch_input set to random_noise, max_num_inputs should not be None")
        if ratings01 is None:
            ratings01 = self.ratings01
        # list of users for the batch
        batch_data = self.keys[subset][self.batch_index[subset] * self.batch_size:
                                       (self.batch_index[subset] + 1) * self.batch_size]

        self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]

        # As inputs: ratings for the same user in the training set
        # As targets: ratings for that user in the subset
        input = np.zeros((self.batch_size, self.n_movies))
        # unobserved ratings are -1 (those don't influence the loss)
        target = np.zeros((self.batch_size, self.n_movies)) - 1
        for (i, userId) in enumerate(batch_data):
            # Create input from training ratings
            if userId in self.ratings["train"]:
                train_ratings = self.ratings["train"][userId]
                if batch_input == 'random_noise':
                    # randomly chose a number of inputs to keep
                    max_nb_inputs = min(max_num_inputs, len(train_ratings) - 1)
                    n_inputs = random.randint(1, max(1, max_nb_inputs))
                    # randomly chose the movies that will be in the input
                    input_keys = random.sample(train_ratings.keys(), n_inputs)
                # Create input from training ratings
                for (movieId, rating) in train_ratings.items():
                    if batch_input == 'full' or (batch_input == 'random_noise' and movieId in input_keys):
                        # movie ratings in a [0.1, 1] range
                        input[i, self.id2index[movieId]] = process_rating(rating, ratings01=ratings01)
            # else:
            #     print("Warning user {} not in training set".format(userId))
            # Create targets
            for (movieId, rating) in self.ratings[subset][userId].items():
                target[i, self.id2index[movieId]] = process_rating(rating, ratings01=ratings01)
        input = Variable(torch.from_numpy(input).float())
        target = Variable(torch.from_numpy(target).float())
        return {"input": input, "target": target}

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def load_categories_batch(self, subset="train", batch_input="full", max_num_inputs=None, ratings01=None):
        if batch_input == 'random_noise' and max_num_inputs is None:
            raise ValueError("batch_input set to random_noise, max_num_inputs should not be None")
        if ratings01 is None:
            ratings01 = self.ratings01
        # list of users for the batch

        batch_data = list(self.keys[subset])[self.batch_index[subset] * self.batch_size:
                                       (self.batch_index[subset] + 1) * self.batch_size]

        self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]


        # As inputs: ratings for the same user in the training set
        # As targets: ratings for that user in the subset
        input = np.zeros((self.batch_size, self.n_movies))
        # unobserved ratings are -1 (those don't influence the loss)
        target = np.zeros((self.batch_size, self.n_movies)) - 1

        category_target = np.zeros((self.batch_size, len(self.categories)))

        for (i, userId) in enumerate(batch_data):
            # Create input from training ratings
            if userId in self.ratings["train"]:
                train_ratings = self.ratings["train"][userId]
                if batch_input == 'random_noise':
                    # randomly chose a number of inputs to keep
                    max_nb_inputs = min(max_num_inputs, len(train_ratings) - 1)
                    n_inputs = random.randint(1, max(1, max_nb_inputs))
                    # randomly chose the movies that will be in the input
                    input_keys = random.sample(train_ratings.keys(), n_inputs)
                # Create input from training ratings
                for (movieId, rating) in train_ratings.items():
                    if batch_input == 'full' or (batch_input == 'random_noise' and movieId in input_keys):
                        # movie ratings in a [0.1, 1] range
                        input[i, self.id2index[movieId]] = process_rating(rating, ratings01=ratings01)
            # else:
            #     print("Warning user {} not in training set".format(userId))
            # Create targets
            for (movieId, rating) in self.ratings[subset][userId].items():
                target[i, self.id2index[movieId]] = process_rating(rating, ratings01=ratings01)

            # retrieve user category vector
            category_target[i] = self.cat_data[subset][userId]

        input = Variable(torch.from_numpy(input).float())
        target = Variable(torch.from_numpy(target).float())
        category_target = Variable(torch.from_numpy(category_target).float())

        return {"input": input, "target": target, "category_target": category_target}

    def load_category_vectors(self, filename = "movie_details_full.csv"):
        # load data from movie_details.csv and create the appropriate dictionaries
        # Database_id, MovieLens_Id, Movie_Title, List_of_Gernes [ the value of the last column will be the gerne vector ]

        # self.database_id_to_name = {}
        # self.database_id_to_movieLens_id = {}
        # self.movieLens_id_to_database_id = {}
        self.database_id_to_cateory_vector = {}
        # self.database_id_to_redial_id = {}
        # self.redial_id_to_database_id = {}
        # self.redial_id_to_categories = {}

        self.movie_lens_id_to_index = {}
        self.index_to_movie_lens_id = {}
        self.index_to_category_vector = {}

        self.index_to_redial_id = {}
        self.redial_id_to_index = {}

        self.database_id_to_index = {}

        self.index_to_name = {}




        with open(filename, 'r') as f:
            reader = csv.reader(f)
            index = 0
            # Database_id, Redial_id MovieLens_Id, Movie_Title, List_of_Gernes [ the value of the last column will be the gerne vector ]
            for row in reader:

                # from the first row we need to extract the list of categories
                if row[0] == "DataBase_id":
                    categories = row[4].split("', '")
                    # save the list that defines which dimension refers to which category
                    self.categories = [category.strip("[]]'") for category in categories]
                else:
                    # extract information from row
                    database_id = int(row[0])
                    redial_id = int(row[1])
                    movie_lens_id = int(row[2])
                    movie_name = row[3]
                    movie_category_vector = np.fromstring(row[4].strip('[]'), sep=" ")


                    if movie_lens_id != -1 :
                        self.movie_lens_id_to_index[movie_lens_id] = index


                    self.index_to_movie_lens_id[index] = movie_lens_id
                    self.index_to_category_vector[index] = movie_category_vector

                    if redial_id != -1:
                        self.index_to_redial_id[index] = redial_id
                        self.redial_id_to_index[redial_id] = index

                    if database_id != -1:
                        self.database_id_to_index[database_id] = index

                    self.index_to_name[index] = movie_name

                    # store information to dictionaries
                    # self.database_id_to_name[database_id] = movie_name
                    # self.database_id_to_movieLens_id[database_id] = movie_lens_id
                    # self.movieLens_id_to_database_id[movie_lens_id] = database_id
                    # self.database_id_to_cateory_vector[database_id] = movie_category_vector
                    # self.database_id_to_redial_id[database_id] = redial_id
                    # self.redial_id_to_database_id[redial_id] = database_id
                    # self.redial_id_to_categories[redial_id] = movie_category_vector


                    index += 1

                    # i+= 1

                    # # check :
                    # print(database_id)
                    # print(self.database_id_to_name[database_id])
                    # print(self.db2name[database_id])
                    # exit()
                    # print()

        # test:
        # print(len(self.db2id)) # db is the ReDial ID !!!
        # print(len(self.db2name)) # db is the ReDial ID !!!
        # exit()
        # i = 0
        # for index in self.index_to_name:

        #     print("index id :", index)
        #     print("OUR redial movie id :", self.index_to_redial_id[index])



        #     print("Name :", self.index_to_name[index])
        #     print()



        #     i += 1
        #     print(i)
        #     if i == 2:
        #         exit()

    def get_categories_movies_matrix(self):
        # finally, return the category-movie matrix [|C| X |M|]
        categories_movies_matrix = []
        for i in range(len(self.index_to_name)):
                categories_movies_matrix.append(self.index_to_category_vector[i])

        return np.transpose( np.array(categories_movies_matrix) )

    # def preprocess_ratings_into_tokenized_lists(self, smooth_targets):
    #     self.ratings_token_ids_lists_data = { "train" : [], "valid" : [], "test" : [] }

    #     frequencies_of_rating_lengths = { "train" : defaultdict(int), "valid" : defaultdict(int), "test" : defaultdict(int) } 

    #     # remove samples with than 2 ratings
    #     removed_small_samples_counter = 0
    #     # for every data set
    #     for subset in self.ratings_token_ids_lists_data:
    #         # for every sample
    #         for sample in self.ratings_data[subset]:
    #             # increase frequency depending the subset and the length of the ratings
    #             frequencies_of_rating_lengths[subset][len(sample)] += 1
    #             if len(sample) > 2:
    #                 self.ratings_token_ids_lists_data[subset].append(([(movie_id, self.liked_token_id if sample[movie_id] > 0.5 else self.disliked_token_id) for movie_id in sample]))
    #             else:
    #                 removed_small_samples_counter += 1

    #     self.n_batches["Bert_train"] = math.ceil(len(self.ratings_token_ids_lists_data["train"])/self.batch_size)
    #     self.n_batches["Bert_valid"] = math.ceil(len(self.ratings_token_ids_lists_data["valid"])/self.batch_size)
    #     self.n_batches["Bert_test"] = math.ceil(len(self.ratings_token_ids_lists_data["test"])/self.batch_size)


    #     # define the values of the ratings thas should be used as targets
    #     if smooth_targets:
    #     # these might work better, because sigmoid is assymptotic to lines y=0 and y=1
    #         self.rating_targets = {"liked": 0.9, "disliked" :0.1}
    #     else:
    #         self.rating_targets = {"liked": 1, "disliked" :0}




    # task 3: give possitive rating, and predict the correct movie
    def _load_ratings_batch4Bert(self, subset, task = "sentiment_as_extra_embedding"):

        number_of_batches = self.n_batches["Bert_" + subset]

        batch_data = self.ratings_token_ids_lists_data[subset][self.batch_index[subset] * self.batch_size:
                                               (self.batch_index[subset] + 1) * self.batch_size]
        # increase batch counter 
        self.batch_index[subset] = (self.batch_index[subset] + 1) % number_of_batches

        # z = 0

        # save lengths of all inputs, in order to construct an input mask at the end (helping the model ingore all zero-padded inputs)
        input_lengths = []
        # the final batch will have a list of inputs and a list of targets
        final_batch = [[],[], [], []]
        # for every sample in the batch
        for sample in batch_data:

            # for training we randomly sample some of the ratings, in order to reconstruck all ratings
            if subset == "train":


                # maintain a random subset of the ratings ()
                # size of random subset 
                # sam_size = randint(2, len(sample))

                # get random subset ( of random length and random indexing) [without replacement] [ Equivalent to ReDial Autor's setting "random_noise"]
                # if self.permute_order:
                #     sample = random.sample(sample, len(sample))

                # prepare the batch according to the (training) task

                # maintain all given ratings
                targets = sample[:]
                movie_list = []
                temp_sentiment_list = []


                # sample the number of ratings that will be given on input
                num_of_inputs = np.random.randint(1 , high = len(sample) - 1)
                # sample some of the indexes that will be used as inputs
                input_ratings = random.sample(sample, num_of_inputs)

                for (movie_id, rating) in input_ratings:
                    movie_list.append(movie_id)
                    sentiment = 1 if rating == self.liked_token_id else -1
                    # if sentiment == -1
                    #     z += 1
                    temp_sentiment_list.append(sentiment)

                # the final token will be the "mask" token, with liked sentiment
                movie_list.append(self.mask_input_token_id)
                temp_sentiment_list.append(1)


                # indexes = [i for i in range( len(sample))]


                # for i in indexes 
                # for i, (movie_id , rating) in enumerate(sample):
                #     if i in indexes
                #     movie_list.append(movie_id)
                #     sentiment = 1 if rating == self.liked_token_id else -1
                #     temp_sentiment_list.append(sentiment)

                # # define the number of targets to be maksed ( at least one needs to be masked, and at least one needs to remain unmasked)
                # number_of_targets = np.random.randint(1 , high = len(sample) - 1)
                # # get a list of indexes for all movie-rating pairs in the remaining sample
                # indexes = [i for i in range( len(sample))]
                # # sample some of the indexes
                # indexes_to_mask = random.sample(indexes, number_of_targets)
                # # print("indexes_to_mask :", indexes_to_mask)

                # # mask movies
                # # if i is supposed to be masked
                # for i in indexes_to_mask:
                #     movie_list[i] = self.mask_input_token_id



                if  task == "sentiment_as_extra_embedding":
                    # input of final sample has to be in the form [, m_1, , m_2, ... , , m_n] [ b_r_1, b_r_2, ... b_r_N,]

                    # append the length of this sample
                    input_lengths.append(len(sample))

                    # the input list is only the movie list
                    input_list = movie_list

                    sentiment_list = temp_sentiment_list


                elif task == "sentiment_as_extra_token":
                    # input of final sample has to be in the form [b_r_1, m_1, b_r_2, m_2, ... , b_r_N, m_n]

                    # append the length of this sample (*2 because at this point the sample is a list of tuples)
                    input_lengths.append(len(sample)*2)

                    # the input list is the a list of sentiments and movie ids
                    input_list = []

                    # sentiments = sentiment_list[:]
                    sentiment_list = []

                    for i in range(len(movie_list)):
                        sentiment_token = self.liked_token_id if temp_sentiment_list[i] == 1 else self.disliked_token_id
                        input_list.append(sentiment_token)
                        input_list.append(movie_list[i])

                        sentiment_list.append(1)
                        sentiment_list.append(temp_sentiment_list[i])


                    # print(input_list)

                else:
                    raise ValueError("Expected task == sentiment_as_extra_embedding OR sentiment_as_extra_token. Instead, received {}".format(task))


                # append preprocessed sample to the final batch
                final_batch[0].append(np.asarray(input_list))
                # saving also the target that is being masked
                final_batch[1].append(np.asarray(sentiment_list))
                # saving also the target that is being masked
                final_batch[2].append(targets)
                # copy full original sample
                final_batch[3].append(sample)



            # for validation and testing, we mask only one rating and try to reconstruct all of them
            # producing N validation samples from every original sample, where N is the number of ratings in the original sample
            else:
                # for every target in the original sample, produce a sample that targets the rating or movie, depending on the task

                # # depending on the task, either sentiment follows movie, or movie follows sentiment
                # if  task == "predict_sentiment_given_movie":
                #     one_dimentional_sample = [ [movie_id, rating]  for (movie_id, rating) in sample ]
                # elif task == "predict_movie_given_sentiment" or task == "predict_movie_given_liked":
                #     one_dimentional_sample = [ [rating, movie_id]  for (movie_id, rating) in sample ]
                # else:
                #     raise ValueError("Expected task == predict_movie_given_sentiment or predict_sentiment_given_movie or predict_movie_given_liked. Instaed, received {}".format(task))
            
                # # flatten the list
                # one_dimentional_sample = reduce(add ,one_dimentional_sample)

                movie_list = []
                sentiment_list = []

                for (movie_id , rating) in sample:
                    movie_list.append(movie_id)
                    sentiment = 1 if rating == self.liked_token_id else -1
                    sentiment_list.append(sentiment)

                # create a sample for each movie. There will be one sample where only one movie is maksed, for every movie
                for i in range(len(sample)):

                    # # append the length of this sample
                    # input_lengths.append(len(sample))

                    temp_movie_list = movie_list[:]
                    temp_sentiment_list = sentiment_list[:]

                    # maintain all given ratings
                    targets = [sample[i]]

                    # # the target is only the one missing rating from input
                    # targets[]


                    # maksed token should to be always at the end
                    temp_movie_list[i] = temp_movie_list[-1]
                    temp_movie_list[-1] = self.mask_input_token_id

                    # also swap the sentiment
                    temp_sentiment_list[i] = temp_sentiment_list[-1]
                    temp_sentiment_list[-1] = 1




                    if  task == "sentiment_as_extra_embedding":
                        # input of final sample has to be in the form [, m_1, , m_2, ... , , m_n] [ b_r_1, b_r_2, ... b_r_N,]

                        # append the length of this sample
                        input_lengths.append(len(sample))

                        # the input list is only the movie list
                        input_list = temp_movie_list

                        sentiment_input_list = temp_sentiment_list


                    elif task == "sentiment_as_extra_token":
                        # input of final sample has to be in the form [b_r_1, m_1, b_r_2, m_2, ... , b_r_N, m_n]

                        # append the length of this sample (*2 because at this point the sample is a list of tuples)
                        input_lengths.append(len(sample)*2)

                        # the input list is the a list of sentiments and movie ids
                        input_list = []
                        # sentiments = sentiment_list[:]
                        sentiment_input_list = []

                        for i in range(len(movie_list)):
                            sentiment_token = self.liked_token_id if temp_sentiment_list[i] == 1 else self.disliked_token_id
                            input_list.append(sentiment_token)
                            input_list.append(temp_movie_list[i])

                            # sentiment corresponding to the (liked/disliked token)
                            sentiment_input_list.append(1)
                            # the actual sentiment of the movie
                            sentiment_input_list.append(temp_sentiment_list[i])

                    else:
                        raise ValueError("Expected task == sentiment_as_extra_embedding OR sentiment_as_extra_token. Instead, received {}".format(task))



                    # append preprocessed sample to the final batch
                    final_batch[0].append(np.asarray(input_list))
                    # saving also the target that is being masked
                    final_batch[1].append(np.asarray(sentiment_input_list))
                    # saving also the target that is being masked
                    final_batch[2].append(targets)
                    # copy full original sample
                    final_batch[3].append(sample)



        # when conted the max_input length, the samples were in the form of tuples
        max_input_length = max(input_lengths)

        # allocate space
        tensor_inputs = torch.zeros([len(final_batch[0]), max_input_length], dtype=torch.torch.long)
        # this tensor will be used for ignoring the zero padded inputs, that are made in order to have a batch of samples of equal size
        tensor_input_masks = torch.zeros([len(final_batch[0]), max_input_length], dtype=torch.torch.long)
        # this tensor will denote the sentiment of the user for the movie
        sentiment_input = torch.zeros([len(final_batch[0]), max_input_length], dtype=torch.torch.float)

        # notice the '-' at the begining, so all padded indexes will be ignored by the loss function
        if self.loss_function == "RMSE":
            # the targets need to be in "one-hot" encoding !
            tensor_targets = - torch.ones([len(final_batch[0]), self.n_movies + len(self.get_special_tokens2id_dict())], dtype=torch.torch.float)
            # masks, denoting the last index of the input, who's output will be evaluated
            output_index_masks = torch.zeros([len(final_batch[0]), max_input_length, self.n_movies + len(self.get_special_tokens2id_dict())], dtype=torch.torch.uint8)
        elif self.loss_function == "CrossEntropy":
            # the targets need to be just the target label
            tensor_targets = - torch.ones([len(final_batch[0]), max_input_length], dtype=torch.torch.long)
        else:
            raise ValueError("In dataset, loss function not set properly!")

        # print("ok")
        # print(self.loss_function)
        # print(len(final_batch[0]))
        # print("last dimension size  :", self.n_movies + len(self.get_special_tokens2id_dict()))
        # for every sample
        for i in range(len(final_batch[0])):
            # copy input sequence
            tensor_inputs[i, :final_batch[0][i].size] = torch.from_numpy(final_batch[0][i])
            # copy sentiment sequence 
            sentiment_input[i, :final_batch[1][i].size] = torch.from_numpy(final_batch[1][i])
            # set input masks
            tensor_input_masks[i, :input_lengths[i]] = 1



            # retrieve targets (same with sample (after random sampling from original sample) )
            targets = final_batch[2][i]
            # print(targets)
            # exit()

            if self.loss_function == "RMSE":
                # create a vector with all rating targets to be reconstructed
                reconstruction_targets = - torch.ones([ self.n_movies + len(self.get_special_tokens2id_dict())], dtype=torch.torch.float)

                for target in targets:
                    (movie_id, rating) = target

                    target_value = self.rating_targets["liked"] if rating ==self.liked_token_id else self.rating_targets["disliked"]
                    # since we are multiplying the outputs of the models with it's ratings, then, even negative ratings, need to output 1
                    reconstruction_targets[movie_id] = target_value

                # print("Sum :", torch.sum(reconstruction_targets != -1 ))
                    # retrieve rating and movie id form original sample ( for all given ratings )
                    # liked = True if rating == self.liked_token_id else False
                    # reconstruction_targets[movie_id] = self.rating_targets["liked"] if liked else self.rating_targets["disliked"]

                # make one more vector like the previous, that doen's have any targets, and it will be used as target for the liked/disliked token output
                # no_targets_vector = - torch.ones(self.n_movies + len(self.get_special_tokens2id_dict()), dtype=torch.torch.float)
                # reconstruction_targets = reconstruction_targets.unsqueeze(0)
                # no_targets_vector = no_targets_vector.unsqueeze(0)
                # join the two target vectors
                # reconstruction_targets = torch.cat( (no_targets_vector, reconstruction_targets), dim = 0)
                # repeat this vector for every (rating, movie_id) pair of input
                # reconstruction_targets = reconstruction_targets.repeat(len(targets),1)
                # print(reconstruction_targets.size())
                # exit()

                tensor_targets[i, :] = reconstruction_targets

                # enabling the mask on the last actual index of the input
                output_index_masks[i, input_lengths[i] -1, :] = 1

                # if  task == "sentiment_as_extra_embedding":
                #     # input of final sample has to be in the form [, m_1, , m_2, ... , , m_n] [ b_r_1, b_r_2, ... b_r_N,]
                #     tensor_targets[i, :] = reconstruction_targets
                # elif task == "sentiment_as_extra_token":
                #     for j in range ( len( targets )):
                #     # input of final sample has to be in the form [b_r_1, m_1, b_r_2, m_2, ... , b_r_N, m_n]
                #         tensor_targets[i, j*2, :] = - torch.ones(self.n_movies + len(self.get_special_tokens2id_dict()), dtype=torch.torch.float)
                #         tensor_targets[i, j*2 + 1, :] = reconstruction_targets[j]

                # else:
                #     raise ValueError("Expected task == sentiment_as_extra_embedding OR sentiment_as_extra_token. Instead, received {}".format(task))

                # for j in range(len(targets)):


                # print(reconstruction_targets.size())
                # # print(reconstruction_targets.size())
                # print(tensor_targets.size())
                # exit()
                # set it as target for the given sample (for all indexes)
                # tensor_targets[i,: reconstruction_targets.size(0),:] = reconstruction_targets
            elif self.loss_function == "CrossEntropy":
                # print(targets)
                for j in range(len(targets)):

                    (movie_id, rating) = targets[j]
                    # print(movie_id)



                    if  task == "sentiment_as_extra_embedding":
                        # input of final sample has to be in the form [, m_1, , m_2, ... , , m_n] [ b_r_1, b_r_2, ... b_r_N,]
                        tensor_targets[i, j] = int(movie_id)
                        # tensor_targets[i, j, :] = reconstruction_targets
                    elif task == "sentiment_as_extra_token":
                        # input of final sample has to be in the form [b_r_1, m_1, b_r_2, m_2, ... , b_r_N, m_n]
                        tensor_targets[i, j*2 ] = -1
                        tensor_targets[i, j*2 +1] = int(movie_id)


                        # tensor_targets[i, j*2, :] = - torch.ones(self.n_movies + len(self.get_special_tokens2id_dict()), dtype=torch.torch.float)
                        # tensor_targets[i, j*2 + 1, :] = reconstruction_targets

                    else:
                        raise ValueError("Expected task == sentiment_as_extra_embedding OR sentiment_as_extra_token. Instead, received {}".format(task))

        return tensor_inputs, tensor_input_masks, sentiment_input, tensor_targets, output_index_masks, final_batch[3]
