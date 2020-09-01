
import csv
from collections import defaultdict
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
from torch.autograd import Variable





class Categories_Dataset(Dataset):
    """

    """
    def __init__(self, ml_ratings_batch_loader):

        # # the total numer of tokens (movies + special tokens)
        # self.vocab_size = ml_ratings_batch_loader.vocab_size
        # # a dictionary of the special tokens ({"liked": self.liked_token_id, "disliked": self.disliked_token_id, "mask": self.mask_input_token_id} )
        # self.special_tokens = ml_ratings_batch_loader.get_special_tokens2id_dict()
        # # assigne the special tokens values to self variables
        # self.liked_token_id = self.special_tokens["liked"]
        # self.disliked_token_id = self.special_tokens["disliked"]
        # self.mask_input_token_id = self.special_tokens["mask"]

        # len of categories gives us the output size of the models
        self.categories = ml_ratings_batch_loader.categories

        self.number_of_movies = ml_ratings_batch_loader.n_movies

        # print(self.number_of_movies)
        # exit()


        # the dataset has a training, validation and test sets, corresponding to the appropriate sets of the Redial dataset.
        # the dataset will have user ratings for input and category distribution as targets.
        # the form of the input will be a list of ratings in the form [ (redial_movie_id, binary_user_rating_token), ...]
        # the target will be a the category vector.
        # different inputs depending on training or validation/test sets.
        # training set input : all ratings (during training a sampled subset of all the ratings will be used for each epoch)
        # validation/test set input: all-but one ratings (therefore, for one user, we will have N validation samples, where N is the number of ratings)

        # subset

        # number_of_batches = ml_ratings_batch_loader.n_batches["Bert_" + subset]

        data = {"train": [[], []], "valid": [[], []], "test": [[], []]}


        # ignore deviding by zero for numpy
        np.seterr(all="raise", divide="raise", over="raise", under="raise", invalid="raise")
        # subset = "train"

        for subset in ["train", "valid", "test"]:

            for i, user_id in enumerate(ml_ratings_batch_loader.ratings[subset]):

                sample = ml_ratings_batch_loader.ratings[subset][user_id]

                # print(sample)
                # print(ml_ratings_batch_loader.ratings[subset][rating])
                # exit()

                # sample = ml_ratings_batch_loader.ratings_token_ids_lists_data[subset][i]
                # initialize the user-category vector with zeros
                user_category_vector = np.zeros(len(ml_ratings_batch_loader.categories))
                # print(ml_ratings_batch_loader.categories)
                # print(user_category_vector)
                # print(sample)

                new_sample = []

                # ml_ratings_batch_loader.movie_lens_id_to_index

                for number_of_ratings, (movie_lens_movie_id) in enumerate(sample):

                    sentiment = sample[movie_lens_movie_id]

                    # print(movie_lens_movie_id, sentiment)

                    sentiment = 1 if float(sentiment) >= 2 else -1

                    index = ml_ratings_batch_loader.movie_lens_id_to_index[int(movie_lens_movie_id)]

                    # print("index :", index)
                    # print(ml_ratings_batch_loader.index_to_name[index])
                    # decode sentiment
                    # print("Redial movie id:", redial_movie_id)
                    # print("name :",ml_ratings_batch_loader.id2name[redial_movie_id])
                    # exit()
                    # sentiment = 1 if sentiment_token == ml_ratings_batch_loader.liked_token_id else -1
                    movie_category_vector = ml_ratings_batch_loader.index_to_category_vector[index]

                    new_sample.append((index, sentiment))

                    # print(movie_category_vector)

                    # exit()

                    movie_category_vector = movie_category_vector * sentiment
                    # print(movie_category_vector)

                    user_category_vector += movie_category_vector

                user_category_vector /= number_of_ratings + 1


                user_category_vector = np.exp(user_category_vector)/sum(np.exp(user_category_vector))


                input = [new_sample]
                target = [user_category_vector]

                data[subset][0] += input
                data[subset][1] += target

        #     dataset.set_data_samples(data)



        # def set_data_samples(self, data):

        self.data_samples = data
        self.number_of_samples = {}
        for subset in data:
            self.number_of_samples[subset] = len(data[subset][0])


    def get_batched_samples(self, subset, batch_size):   
    # def get_dataset_for_FFNN_ReDial(self):
        # this function will preprocess the dataset to its final form  and split it into batches
        # the final form of the dataset will consist of input vectors with cardinality equal to the number of movies,
        # and output vectors with cardinality equal to the number of categories

        input_size = self.number_of_movies
        # target_size = len(self.categories)

        # target_vectors = {}

        # print("number_of_samples :", self.number_of_samples[subset])

        indexes = np.arange(self.number_of_samples[subset])
        # making a list for randomizing index for training set, works like suffling the samples set
        if subset == "train":
            np.random.shuffle(indexes)

        # for subset in self.number_of_samples:

        # first list corresponds to input vectors, the second list corresponds to rating targets and third list corresponds to category targets
        sample_vectors = [[], [], []]
        for i in indexes:


            rating_sample = self.data_samples[subset][0][i]
            category_target = self.data_samples[subset][1][i]

            input_vector = np.zeros(input_size)
            # category_target = torch.tensor(target_sample, requires_grad=True)

            if subset == "train":

                rating_target = - np.ones(input_size)

                # get a random subset of the ratings for input 
                random_index = np.random.randint(2, size=len(rating_sample)) > 0


                # copy given ratings on input vector
                for i, (movie_id, rating) in enumerate(rating_sample):
                    rating_target[movie_id] = rating

                    if random_index[i]:
                        input_vector[movie_id] = rating

            # input = Variable(torch.from_numpy(input).float())

                sample_vectors[0].append(torch.from_numpy(input_vector).float()) 
                sample_vectors[1].append(torch.from_numpy(rating_target).float()) 
                sample_vectors[2].append(torch.from_numpy(category_target).float()) 

                # sample_vectors[0].append(torch.tensor(input_vector).float())
                # sample_vectors[1].append(torch.tensor(rating_target).float())
                # sample_vectors[2].append(torch.tensor(category_target).float())

            else:
                # Maintain the one target, given rest ratings, as the ReDial Authors did

                # copy given ratings on input vector
                for (movie_id, rating) in rating_sample:
                    input_vector[movie_id] = rating
                    # rating_target[movie_id] = rating


                
                # for, every rating, hide that rating from input and create one sample
                for i, (movie_id, rating) in enumerate(rating_sample):
                    # copy input vector
                    temp_input = np.copy(input_vector)
                    # hide one rating
                    temp_input[movie_id] = 0
                    # initialize target vector
                    rating_target = - np.ones(input_size)
                    # set the only this rating as target for this sample 
                    rating_target[movie_id] = rating

                    

                    # append this sample
                    sample_vectors[0].append(torch.from_numpy(temp_input).float())
                    sample_vectors[1].append(torch.from_numpy(rating_target).float())
                    sample_vectors[2].append(torch.from_numpy(category_target).float())
                    # sample_vectors[0].append(torch.tensor(temp_input).float())
                    # sample_vectors[1].append(torch.tensor(rating_target).float())
                    # sample_vectors[2].append(torch.tensor(category_target).float())


        # for training subset, we shuffle the samples
        if subset == "train":
            sample_vectors = list(zip(sample_vectors[0], sample_vectors[1], sample_vectors[2]))
            random.shuffle(sample_vectors)
            sample_vectors[0], sample_vectors[1], sample_vectors[2] = zip(*sample_vectors)

        # split the samples into batches

        # calculate number of batches
        number_of_batches = math.ceil(len(sample_vectors[0])/batch_size)

        # print("number of batches after preprocessing :", len(sample_vectors[0]))

        # print("number_of_batches :", number_of_batches)

        batch_data = []
        # print(sample_vectors[0])

        for i in range(number_of_batches):
            input_vector = Variable(torch.stack(sample_vectors[0][i * batch_size: (i + 1) * batch_size]), requires_grad = True)
            rating_target = Variable(torch.stack(sample_vectors[1][i * batch_size: (i + 1) * batch_size]), requires_grad = True)
            category_vector = Variable(torch.stack(sample_vectors[2][i * batch_size: (i + 1) * batch_size]), requires_grad = True)

            batch_data.append( (input_vector, rating_target, category_vector))

        return batch_data



