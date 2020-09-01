import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import copy
import math
import datetime 
from multiprocessing.pool import ThreadPool
import os
import threading

from tqdm import tqdm

class EvaluationClass(nn.Module):
    def __init__(self):
        super(EvaluationClass, self).__init__()
        #     self.decoder_layers.append( nn.Linear(in_features=self.layer_sizes[len(self.layer_sizes) -1 - i], out_features=self.layer_sizes[len(self.layer_sizes) -2 - i]))

        # calculate RMSE sum
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.mse_sum = 0
        self.total_targets = 0

        self.rankings = []
        self.correct_class_counter = 0

        self.lock = threading.Lock()

        # calculate RMSE sum
        self.cat_mse_loss = nn.MSELoss(reduction='sum')

        self.cat_mse_sum = 0

        self.cat_total_targets = 0


    def calculate_batch_loss(self, output, target):

        mask = (target != -1)
        observed_output = torch.masked_select(output, mask)
        observed_target = torch.masked_select(target, mask)

        loss = self.mse_loss(observed_output, observed_target)


        self.total_targets += len(observed_target)

        self.mse_sum += loss.item()

        # print(loss / float(len(observed_target)), loss, len(observed_target))

        return loss / float(len(observed_target))



    def get_normalized_loss(self):

        RMSE = math.sqrt(self.mse_sum/self.total_targets)

        self.total_targets = 0
        self.mse_sum = 0

        return RMSE

    def evaluate_output(self, input, target, output):
        # only consider the observed targets

        # get indexes of non zero input (given ratings), and exclude them from the final ranking (set their score to zero) [considering all positive ratings]
        given_ratings = (input >= 0.5).nonzero()

        positive_given_ratings_per_sample = [[]] * input.size(0)

        # split given ratings into samples
        for sample_idx, movie_id in given_ratings:
            positive_given_ratings_per_sample[sample_idx].append( (movie_id, input[sample_idx, movie_id]) )

        # get indexes of targets
        target_indexes = (target != -1).nonzero().tolist()

        for t in target_indexes:
            s_idx, t_idx = t
            t.append(target[s_idx, t_idx ].item())

        rankings = []

        correct_classifications_counter = 0

        # for each target 
        for s_idx, t_idx, rating in target_indexes:

            outputted_value = output[s_idx, t_idx]

            # check if output agrees with rating
            # print("Rating:",rating, "output:", outputted_value.item(), (rating > 0.5 and outputted_value.item() > 0.5) or (rating < 0.5 and outputted_value.item() < 0.5))
            if (rating > 0.5 and outputted_value.item() > 0.5) or (rating < 0.5 and outputted_value.item() < 0.5):
                correct_classifications_counter += 1

            # retrieve ratings given with original sample
            given_ratings = positive_given_ratings_per_sample[s_idx]

            # if the target is positive
            if rating > 0.5:

                # excluding all positive targets from ranking, except for the target
                list_of_ranks_to_exclude = [  movie_id  for (movie_id, rating_token) in given_ratings if rating_token == 1 and movie_id != t_idx]
                # make a copy of the output for this sample
                temp_output = output[s_idx].clone()
                # zero the outputs of the given ratings so that they go to the bottom of the list
                for movie_id in list_of_ranks_to_exclude:
                    temp_output[movie_id] = 0

                # get indexes of output, if they were sorted (in descending order)
                sorted_indexes = temp_output.argsort(dim=-1, descending=True)
                # append ranking of target to list of rankings
                rankings.append ( ((sorted_indexes == t_idx).nonzero()).item() )

            else:
                # # excluding all negative targets from ranking, except for the target
                # list_of_ranks_to_exclude = [  movie_id  for (movie_id, rating_token) in given_ratings if rating_token == special_tokens2id_dict["disliked"] and movie_id != t_idx]
                # # make a copy of the output for this sample
                temp_output = output[s_idx].clone()
                # # zero the outputs of the given ratings so that they go to the bottom of the list
                # for movie_id in list_of_ranks_to_exclude:
                #     temp_output[movie_id] = 1

                # get indexes of output, if they were sorted (in ascending order)
                sorted_indexes = temp_output.argsort(dim=-1, descending=False)
                # append ranking of target to list of rankings
                rankings.append ( ((sorted_indexes == t_idx).nonzero()).item() )

        # only consider the observed targets
        mask = (target != -1)
        observed_input = torch.masked_select(output, mask)
        observed_target = torch.masked_select(target, mask)
        # update metrics

        self.lock.acquire()
        try:
            # logging.debug('Acquired a lock')

            self.mse_loss = nn.MSELoss(reduction='sum')
            loss = self.mse_loss(observed_input, observed_target)

            # print(loss.item())

            self.mse_sum += loss.item()

            self.total_targets += len(observed_target)

            self.rankings += rankings
            self.correct_class_counter += correct_classifications_counter

            self.batch_loss += loss
            self.batch_targets += len(observed_target)

        finally:
            # logging.debug('Released a lock')
            self.lock.release()

        # exit()

        return loss

    # def get_index_with_distractors_multithreaded(self, original_batch, targets, model_outputs, num_of_workers):
    #     return

    def evaluate_output_multithreaded(self, input, target, output, num_of_workers):

        
        # number of workers should not exceed the batch_size
        num_of_workers = min(num_of_workers, input.size(0))


        self.batch_targets = 0
        self.batch_loss = 0

        if num_of_workers == 0:
            self.evaluate_output(input, target, output)

            loss = self.batch_loss / float(self.batch_targets)
            self.batch_loss = 0
            self.batch_targets = 0

            return loss

        # target_indexes = (target != -1).nonzero().tolist()
        # split the inputs into buckets
        buckets = []

        # for every thread
        for i in range(num_of_workers):
            # append empty lists for the parameters
            buckets.append([[],[],[]])

        # for every sample
        for i in range(input.size(0)):
            # get thread index
            thread_index = i%num_of_workers
            buckets[thread_index][0].append(input[i])
            buckets[thread_index][1].append(target[i])
            buckets[thread_index][2].append(output[i])

        for i in range(len(buckets)):
            buckets[i][0] = torch.stack(buckets[i][0])
            buckets[i][1] = torch.stack(buckets[i][1])
            buckets[i][2] = torch.stack(buckets[i][2])

        pool = ThreadPool(processes=num_of_workers)
        for i in range(num_of_workers):
            pool.apply_async(self.evaluate_output, args =  buckets[i])

        pool.close()
        pool.join()

        loss = self.batch_loss / float(self.batch_targets)
        self.batch_loss = 0
        self.batch_targets = 0


        return loss

    def get_result_metrics(self):

        RMSE = math.sqrt(self.mse_sum/ float(self.total_targets) )
        corr_class_rate = self.correct_class_counter / float(self.total_targets)

        # calculate ranking rates
        hits_at_1 = 0
        hits_at_10 = 0
        hits_at_100 = 0

        for ranking in self.rankings:
            if ranking < 1:
                hits_at_1 += 1
            if ranking < 10:
                hits_at_10 += 1
            if ranking < 100:
                hits_at_100 += 1

        hits_at_1 /= float(self.total_targets)
        hits_at_10 /= float(self.total_targets)
        hits_at_100 /= float(self.total_targets)

        av_ranking = sum(self.rankings) / float(len(self.rankings) )

        # reset metrics 
        self.mse_sum = 0
        self.total_targets = 0

        self.rankings = []
        self.correct_class_counter = 0


        return RMSE, corr_class_rate, hits_at_1, hits_at_10, hits_at_100, av_ranking


    def calculate_batch_cat_loss(self, output, target):

        loss = self.cat_mse_loss(output, target)

        self.cat_total_targets += output.size(0) * output.size(1)

        self.cat_mse_sum += loss.item()


        return loss / float(output.size(0) * output.size(1))



    def get_normalized_cat_loss(self):

        
        RMSE = math.sqrt(self.cat_mse_sum/self.cat_total_targets)

        self.cat_total_targets = 0
        self.cat_mse_sum = 0

        return RMSE






# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class AutoEncoder(EvaluationClass):
    """
    User-based Autoencoder for Collaborative Filtering
    """
    def __init__(self, n_movies, layer_sizes = [1000], num_of_categories = 19, load_model_path = ""):
        super(AutoEncoder, self).__init__()

        self.act_func = nn.Sigmoid()

        self.n_movies = n_movies

        self.num_of_categories = num_of_categories

        self.layer_sizes = layer_sizes

        # self.rec_layer_sizes = args.rec_layer_sizes
        # set layer sizes
        encoder_layer_sizes = [self.n_movies] + layer_sizes + [self.num_of_categories]

        decoder_layer_sizes = [self.num_of_categories] + list(reversed(layer_sizes)) + [self.n_movies]

        # initialie linear layers
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        for i in range( len(encoder_layer_sizes) -1):
            self.encoder_layers.append( nn.Linear(in_features=encoder_layer_sizes[i], out_features=encoder_layer_sizes[i+1]))
        for i in range( len(decoder_layer_sizes) -1):
            self.decoder_layers.append( nn.Linear(in_features=decoder_layer_sizes[i], out_features=decoder_layer_sizes[i+1]))





        # self.hidden_size = self.layer_sizes[-1]
        # self.encoder = nn.Linear(in_features=n_movies, out_features=self.hidden_size)

        # self.decoder = nn.Linear(in_features=self.hidden_size, out_features=n_movies)


        if load_model_path != "":
            load_model_path = os.path.join(load_model_path, "pruned.pickle")
            # load model pretrained on category loss minimization
            self.load_state_dict(torch.load(load_model_path))




    def forward(self, input):


        # input = sa_movie_vector

        for i in range( len(self.encoder_layers)):
            input = torch.sigmoid( self.encoder_layers[i](input))

        for i in range( len(self.decoder_layers)):
            input = torch.sigmoid( self.decoder_layers[i](input))

        # # the last layer uses softmax for activation function
        # item_rec_scores = torch.softmax(self.decoder_layers[-1](input), dim = -1)

        # out = self.encoder(input)
        # out = torch.sigmoid(out)
        # out = self.decoder(out)
        # out = torch.sigmoid(out)
        return input

    def evaluate(self, batch_loader, subset, args):

        self.eval()

        if args.dataset == "redial":
            batch_data, number_of_batches = batch_loader.get_batched_samples(subset = subset, batch_size = args.batch_size)
        else:
            # load_categories_batch
            number_of_batches = batch_loader.n_batches[subset]



        for i in tqdm(range(number_of_batches)):

            if args.dataset == "redial":
                batch = batch_data[i]

                input_vectors = batch[0]
                rating_targets = batch[1]
                category_targets = batch[2]
            else:
                batch = batch_loader.load_categories_batch(subset = subset)

                input_vectors = batch["input"]        
                rating_targets = batch["target"]
                category_targets = batch["category_target"]



            if next(self.parameters()).is_cuda:
                input_vectors = input_vectors.cuda()
                rating_targets = rating_targets.cuda()
                category_targets = category_targets.cuda()


            output = self(input_vectors)


            if args.dataset == "redial":
                # this is for redial trained on ratings
                self.evaluate_output_multithreaded(input = input_vectors, target = rating_targets, output = output, num_of_workers = args.num_of_workers)
            else:
                # this is for MovieLens trained on ratings
                self.calculate_batch_loss(output = output, target = rating_targets)


        if args.dataset == "redial":
            # this is for redial trained on ratings
            output = self.get_result_metrics()
        else:
            # this is for MovieLens trained on ratings
            output = self.get_normalized_loss()

        self.train()

        return output


    def save_pruned_model_for_redial(self, output_dir, redial_n_of_movies = 6924):

        temp_encoder = torch.nn.Linear(redial_n_of_movies, self.layer_sizes[0] )

        temp_encoder.weight = torch.nn.Parameter(self.encoder_layers[0].weight[:,:redial_n_of_movies].clone())
        temp_encoder.bias = torch.nn.Parameter(self.encoder_layers[0].bias.clone())


        # delete first layer
        self.encoder_layers.__delitem__(0)
        # put the temporary layer in its place
        self.encoder_layers.insert(0, temp_encoder)


        # self.encoder = temp_encoder
        # self.encoder_layers[0] = temp_encoder

        temp_decoder = nn.Linear(in_features=self.layer_sizes[-1], out_features=redial_n_of_movies)

        temp_decoder.weight = torch.nn.Parameter(self.decoder_layers[-1].weight[:redial_n_of_movies,:].clone())
        temp_decoder.bias = torch.nn.Parameter(self.decoder_layers[-1].bias[:redial_n_of_movies].clone())

        self.decoder_layers.__delitem__(-1)
        self.decoder_layers.append(temp_decoder)


        # self.decoder = temp_decoder
        # self.decoder_layers[-1] = temp_decoder

        torch.save(self.state_dict(), os.path.join(output_dir, "pruned.pickle"))

        # print(self)
        # exit()




class AutoEncoderWithCategories(EvaluationClass):
    """
    Class that uses the hidden representation of the auto encoder and uses a FFN to predict category distribution
    """
    def __init__(self, number_of_movies, num_of_categories, layer_sizes=[1000], decoder_mode = "symmetric", load_model_path = "", task = "categories", transfer_learning = False):
        super(AutoEncoderWithCategories, self).__init__()


        self.n_movies = number_of_movies

        self.task = task

        self.num_of_categories = num_of_categories

        self.decoder_mode = decoder_mode

        self.layer_sizes = layer_sizes

        # set layer sizes
        encoder_layer_sizes = [self.n_movies] + layer_sizes + [self.num_of_categories]


        # initialie linear layers
        self.encoder_layers = nn.ModuleList()

        for i in range( len(encoder_layer_sizes) -1):
            self.encoder_layers.append( nn.Linear(in_features=encoder_layer_sizes[i], out_features=encoder_layer_sizes[i+1]))



        # # set layer sizes
        # self.layer_sizes = [self.n_movies] + layer_sizes + [num_of_categories]
        # # initialie linear layers
        # self.layers = nn.ModuleList()
        # for i in range( len(self.layer_sizes) -1):
        #     self.layers.append( nn.Linear(in_features=self.layer_sizes[i], out_features=self.layer_sizes[i+1]))

        if load_model_path != "":

            lastly_modified_training_filename = load_model_path.split("From")[0]

            # checking whether we should also load the cat_to_movies_layer (if the source model was trained for ratings)
            if "ratings" in lastly_modified_training_filename:

                if self.decoder_mode == "symmetric":

                    decoder_layer_sizes = [self.num_of_categories] + list(reversed(layer_sizes)) + [self.n_movies]

                elif  self.decoder_mode == "1_layer":

                    decoder_layer_sizes = [self.num_of_categories] + [self.n_movies]
                    
                self.decoder_layers = nn.ModuleList()

                for i in range( len(decoder_layer_sizes) -1):
                    self.decoder_layers.append( nn.Linear(in_features=decoder_layer_sizes[i], out_features=decoder_layer_sizes[i+1]))


                # self.cat_to_movies = nn.Linear(in_features=self.num_of_categories, out_features=number_of_movies)


            if "movielens" in lastly_modified_training_filename:
                load_model_path = os.path.join(load_model_path, "pruned.pickle")
            else:
                load_model_path = os.path.join(load_model_path, "best_loss.pickle")

            # load model pretrained on category loss minimization
            self.load_state_dict(torch.load(load_model_path))


            if transfer_learning:
                # freeze the first layers up to category so that they don't update their weights
                for i in range( len(self.encoder_layers)):
                    for param in self.encoder_layers[i].parameters():
                        param.requires_grad = False


        if self.decoder_mode != "fixed" and task == "ratings":
            # if we haven't already loaded this layer from a file
            if hasattr(self, 'decoder_layers') == False:

                if self.decoder_mode == "symmetric":

                    decoder_layer_sizes = [self.num_of_categories] + list(reversed(layer_sizes)) + [self.n_movies]

                elif  self.decoder_mode == "1_layer":

                    decoder_layer_sizes = [self.num_of_categories] + [self.n_movies]
                    
                self.decoder_layers = nn.ModuleList()

                for i in range( len(decoder_layer_sizes) -1):
                    self.decoder_layers.append( nn.Linear(in_features=decoder_layer_sizes[i], out_features=decoder_layer_sizes[i+1]))

                # self.cat_to_movies = nn.Linear(in_features=self.num_of_categories, out_features=number_of_movies)
            else:
                print("Cat to movies Linear Layer was loaded from file, so we do not override it")


    def forward(self, input):
        # get hidden representation from the frozen_encoder of the autoencoder
        # category_scores = self.frozen_encoder(input)


        for i in range( len(self.encoder_layers) -1):
            input = torch.sigmoid( self.encoder_layers[i](input))

        # for i in range( len(self.decoder_layers)):
        #     input = torch.sigmoid( self.decoder_layers[i](input))


        # for i in range( len(self.layer_sizes) - 2):
        #     input = torch.sigmoid( self.layers[i](input))

        # the last layer uses softmax for activation function
        category_scores = torch.softmax(self.encoder_layers[-1](input), dim = -1)

        if self.task == "ratings":

            if self.decoder_mode != "fixed":

                cat_movie_scores = category_scores

                for i in range( len(self.decoder_layers)):
                    cat_movie_scores = torch.sigmoid( self.decoder_layers[i](cat_movie_scores))

                # cat_movie_scores = self.cat_to_movies(category_scores)
            else:
                cat_movie_scores = torch.sigmoid(torch.matmul(category_scores, self.categories_movies_matrix))

            # movie_scores = torch.sigmoid(cat_movie_scores)

            return cat_movie_scores

        return category_scores



    def set_categories_movies_matrix(self, categories_movies_matrix ):
        self.categories_movies_matrix  = torch.from_numpy(categories_movies_matrix).float()
        if next(self.parameters()).is_cuda:
            self.categories_movies_matrix = self.categories_movies_matrix.cuda()
        self.categories_movies_matrix.requires_grad = False


    def save_pruned_model_for_redial(self, output_dir, redial_n_of_movies = 6924):

        temp = torch.nn.Linear(redial_n_of_movies, self.encoder_layers[0].weight.size(0) )

        temp.weight = torch.nn.Parameter(self.encoder_layers[0].weight[:,:redial_n_of_movies].clone())
        temp.bias = torch.nn.Parameter(self.encoder_layers[0].bias.clone())

        # delete first layer
        self.encoder_layers.__delitem__(0)
        # put the temporary layer in its place
        self.encoder_layers.insert(0, temp)

        # if the model was trained on ML for ratings, then we need to prune the output layer as well
        if self.task == "ratings":
            temp = nn.Linear(in_features=self.decoder_layers[-1].weight.size(1), out_features=redial_n_of_movies)

            temp.weight = torch.nn.Parameter(self.decoder_layers[-1].weight[:redial_n_of_movies,:].clone())
            temp.bias = torch.nn.Parameter(self.decoder_layers[-1].bias[:redial_n_of_movies].clone())

            self.decoder_layers.__delitem__(-1)
            self.decoder_layers.append(temp)


        torch.save(self.state_dict(), os.path.join(output_dir, "pruned.pickle"))



    def evaluate(self, batch_loader, subset, args):

        self.eval()


        if args.dataset == "redial":
            batch_data, number_of_batches = batch_loader.get_batched_samples(subset = subset, batch_size = args.batch_size)
        else:
            # load_categories_batch
            number_of_batches = batch_loader.n_batches[subset]



        for i in tqdm(range(number_of_batches)):

            if args.dataset == "redial":
                batch = batch_data[i]

                input_vectors = batch[0]
                rating_targets = batch[1]
                category_targets = batch[2]
            else:
                batch = batch_loader.load_categories_batch(subset = subset)

                input_vectors = batch["input"]        
                rating_targets = batch["target"]
                category_targets = batch["category_target"]



            if next(self.parameters()).is_cuda:
                input_vectors = input_vectors.cuda()
                rating_targets = rating_targets.cuda()
                category_targets = category_targets.cuda()


            output = self(input_vectors)


            if args.task == "categories":
                # this is for anything trained on categories
                self.calculate_batch_cat_loss(output = output, target = category_targets)
            else:
                if args.dataset == "redial":
                    # this is for redial trained on ratings
                    self.evaluate_output_multithreaded(input = input_vectors, target = rating_targets, output = output, num_of_workers = args.num_of_workers)
                else:
                    # this is for MovieLens trained on ratings
                    self.calculate_batch_loss(output = output, target = rating_targets)


        if args.task == "categories":
            # this is for anything trained on categories
            output = self.get_normalized_cat_loss()
        else:
            if args.dataset == "redial":
                # this is for redial trained on ratings
                output = self.get_result_metrics()

            else:
                # this is for MovieLens trained on ratings
                output = self.get_normalized_loss()

        self.train()

        return output
