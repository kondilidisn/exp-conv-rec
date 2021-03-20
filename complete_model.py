import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import transformers

from category_pref_model import Cat_Pref_BERT


class Complete_Model(nn.Module):
	def __init__(self, vocab_size, cat_size, n_movies, args = None):
		super(Complete_Model, self).__init__()


		self.vocab_size = vocab_size
		self.cat_size = cat_size
		self.n_movies = n_movies
		self.args = args
		self.use_ground_truth = args.use_ground_truth

		# loss and evaluation parameters
		self.CrossE_Loss = torch.nn.CrossEntropyLoss()
		self.loss_sum = 0
		self.hit_at_1_counter = 0
		self.hit_at_10_counter = 0
		self.hit_at_100_counter = 0
		self.total_number_of_targets = 0


		# instantiate base model
		self.base_model = Cat_Pref_BERT(vocab_size = vocab_size, cat_size = cat_size, n_movies=n_movies,\
			ignore_model = args.use_ground_truth, args = args)

		# load base model
		if args.use_ground_truth == False and args.base_model_dir != '':
			self.base_model.load_state_dict(torch.load( os.path.join(args.base_model_dir, "best.pickle") ))

		# if we are not finetuning, but we do transfer learning, then we freeze the parameters of the base model
		if args.finetune == False or args.use_ground_truth == True:
			for param in self.base_model.parameters():
				param.requires_grad = False

		# initialize linear decoder layer (item Recommender module)
		self.decoder_layer =  nn.Linear(in_features = self.cat_size, out_features = self.n_movies)

		# optimizer
		self.optimizer = torch.optim.Adam(self.parameters())



	def evaluate_model(self, batch_loader, subset):

		n_batches = batch_loader.n_batches[subset]

		with torch.no_grad():

			for step in tqdm(range(n_batches)):

				# we retrieve a batch
				batch = batch_loader.load_batch(subset = subset)

				if batch == None:
					continue

				minibatches = self.base_model.split_batch_to_minibatches(batch)

				for minibatch in minibatches:
					
					output, loss = self(minibatch)

		return self.get_metrics_and_reset()


	def train_epoch(self, batch_loader):


		n_batches = batch_loader.n_batches["train"]

		for step in tqdm(range(n_batches)):

			self.zero_grad()

			# we retrieve a batch
			batch = batch_loader.load_batch(subset = "train")

			if batch == None:
				continue

			batch_size = batch["contexts"].size(0)

			minibatches = self.base_model.split_batch_to_minibatches(batch)

			for minibatch in minibatches:

				minibatch_size = minibatch["contexts"].size(0)

				minibatch_to_batch_ratio = minibatch_size / batch_size

				# do the forward pass
				output, loss = self(minibatch)

				loss *= minibatch_to_batch_ratio

				if loss != 0:
					loss.backward()

			self.optimizer.step()

		return self.get_metrics_and_reset()


	def update_hit_at_rank_metrics(self, item_rec_scores, complete_sample_movie_targets):

		_, ranked_movie_ids = torch.topk(item_rec_scores, k=100, dim=-1)

		ranked_movie_ids = ranked_movie_ids.cpu().tolist()

		for i in range(complete_sample_movie_targets.size(0)):
			# get target for that sample
			target_movie_id = complete_sample_movie_targets[i].item()

			if target_movie_id == ranked_movie_ids[i][0]:
				self.hit_at_1_counter += 1
				self.hit_at_10_counter += 1
				self.hit_at_100_counter += 1

			elif target_movie_id in ranked_movie_ids[i][:10]:
				self.hit_at_10_counter += 1
				self.hit_at_100_counter += 1

			elif target_movie_id in ranked_movie_ids[i]:
				self.hit_at_100_counter += 1

	def get_metrics_and_reset(self):

		metrics = {}
		metrics["Hit@1"] = self.hit_at_1_counter / float(self.total_number_of_targets)
		metrics["Hit@10"] = self.hit_at_10_counter / float(self.total_number_of_targets)
		metrics["Hit@100"] = self.hit_at_100_counter / float(self.total_number_of_targets)
		metrics["Loss"] = self.loss_sum / float(self.total_number_of_targets)

		self.hit_at_1_counter = 0
		self.hit_at_10_counter = 0
		self.hit_at_100_counter = 0
		self.loss_sum = 0
		self.total_number_of_targets = 0

		return metrics


	def forward(self, batch):

		batch_movie_mentions, complete_sample_movie_targets = batch["batch_movie_mentions"], batch["complete_sample_movie_targets"]

		if self.use_ground_truth:
			cat_pred = batch["category_targets"]
		else:
			cat_pred, cat_loss = self.base_model(batch)


		if next(self.parameters()).is_cuda:
			cat_pred = cat_pred.cuda()
			complete_sample_movie_targets = complete_sample_movie_targets.cuda()

		# Apply the Item Recommendation Scores Feed Forward network (1 layer)
		output = self.decoder_layer(cat_pred)

		# Then the softmax activation function
		item_rec_scores = torch.softmax(output , dim = -1)

		# calcualte loss, on the task of predicting exactly the recommended movie
		loss = self.CrossE_Loss(output, complete_sample_movie_targets)
		# add loss to sum, in order to properly average the epoch loss
		total_targets = complete_sample_movie_targets.size(0)


		self.update_hit_at_rank_metrics(item_rec_scores, complete_sample_movie_targets)
		self.loss_sum += loss.item() * total_targets
		self.total_number_of_targets += total_targets

		return item_rec_scores, loss

