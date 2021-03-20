# the code in this file is based on the code of ReDial authors
# https://github.com/RaymondLi0/conversational-recommendations/blob/master/batch_loaders/batch_loader.py

import os
import numpy as np
import re
from random import shuffle, randint, shuffle
import csv
import json
from tqdm import tqdm
import pickle
import random
from collections import Counter, defaultdict
from operator import add
from functools import reduce
import math

import torch
from torch.autograd import Variable

import transformers

import sys
import nltk


def load_data(path):
	# path = os.getcwd() + path
	"""

	:param path:
	:return:
	"""
	data = []
	for line in open(path):
		data.append(json.loads(line))
	return data

			# batch_loader = DialogueBatchLoader(sources="ratings", conversations_per_batch=16)
class DialogueBatchLoader4Transformers(object):
	def __init__(self, conversations_per_batch,
				 max_input_length = 512,
				 conversation_length_limit=40,
				 utterance_length_limit=80,
				 data_path='Datasets/redial',
				 train_path="train_data",
				 valid_path="valid_data",
				 test_path="test_data.jsonl",
				 movie_details_path="movie_details.csv",
				 process_at_instanciation=False,
				 special_tokens_list = ['SOS', 'EOS', 'MASK', 'PAD', 'UNK', 'SEP', 'Movie_Mentioned']):
		self.conversations_per_batch = conversations_per_batch
		self.batch_index = {"train": 0, "valid": 0, "test": 0}
		self.conversation_length_limit = conversation_length_limit
		self.utterance_length_limit = utterance_length_limit
		self.data_path = {"train": os.path.join(data_path, train_path),
						  "valid": os.path.join(data_path, valid_path),
						  "test": os.path.join(data_path, test_path)}
		self.movie_details_path = os.path.join(data_path, movie_details_path)

		self.word2id = None
		self.process_at_instanciation = process_at_instanciation

		self.max_input_length = max_input_length

		self.load_movie_information_with_category_vectors(filepath = self.movie_details_path)

		self.cls_tokens = [ "CLS_Cat_" + str(i) for i in range(len(self.categories)) ]

		self.special_tokens_list = special_tokens_list + [ "CLS_Cat_" + str(i) for i in range(len(self.categories)) ] + ["CLS"]

		# load data
		self.conversation_data = {key: load_data(val) for key, val in self.data_path.items()}
		# set up number of batches per subset
		self.n_batches = {key: len(val) // self.conversations_per_batch for key, val in self.conversation_data.items()}

		tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")#, additional_special_tokens = special_tokens)

		print("Initial Vocab Size:", tokenizer.vocab_size)

		tokenizer.add_tokens(self.special_tokens_list)

		self.bert_tokenizer = tokenizer

		self.vocabulary_size = tokenizer.vocab_size + len(self.special_tokens_list)

		print("Final Vocab Size:", self.vocabulary_size)

		self.cls_token_ids = [self.encode(cls_token)[0] for cls_token in self.cls_tokens]


	def encode(self, text):
		# if self.use_pretrained:
		token_ids = self.bert_tokenizer.encode(text)[1:-1] # Exclude EOS and SOS tokens.
		if isinstance(token_ids, int):
			return [token_ids]
		else:
			return token_ids

	def decode(self, list_of_token_ids):
		return self.bert_tokenizer.decode(list_of_token_ids, clean_up_tokenization_spaces=False)

	def load_movie_information_with_category_vectors(self, filepath):
		# load data from movie_details.csv and create the appropriate dictionaries
		# Database_id, MovieLens_Id, Movie_Title, List_of_Gernes [ the value of the last column will be the gerne vector ]

		# self.database_id_to_name = {}
		# self.database_id_to_movieLens_id = {}
		# self.movieLens_id_to_database_id = {}
		# self.database_id_to_cateory_vector = {}
		self.database_id_to_redial_id = {}
		# self.redial_id_to_database_id = {}
		self.redial_id_to_categories = {}

		# filepath = os.path.join(config.REDIAL_DATA_PATH, filename)

		with open(filepath, 'r') as f:
			reader = csv.reader(f)
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

					# store information to dictionaries
					if database_id != -1:
						# self.database_id_to_name[database_id] = movie_name
						# self.database_id_to_movieLens_id[database_id] = movie_lens_id
						# self.database_id_to_cateory_vector[database_id] = movie_category_vector
						self.database_id_to_redial_id[database_id] = redial_id
					# if movie_lens_id != -1:
					# 	self.movieLens_id_to_database_id[movie_lens_id] = database_id
					if redial_id != -1:
						# self.redial_id_to_database_id[redial_id] = database_id
						self.redial_id_to_categories[redial_id] = movie_category_vector

		# self.n_movies = len(self.redial_id_to_database_id)
		self.n_movies = len(self.redial_id_to_categories)

		print("Number of Movies: ", self.n_movies)

	def extract_dialogue4Bert(self, conversation):
		"""
		:param conversation: conversation dictionary. keys : 'conversationId', 'respondentQuestions', 'messages',
		 'movieMentions', 'respondentWorkerId', 'initiatorWorkerId', 'initiatorQuestions'
		 :param flatten_messages
		 :return:
			dialogue
			senders
			movie_mentions
			category_target
			answers_dict
		"""

		answers_dict = conversation["initiatorQuestions"]
		# if there are not initiatorQuestions answers, then we use the respondentQuestions answers
		if len(answers_dict) == 0:
			answers_dict = conversation["respondentQuestions"]

		# translate answers dict so that it uses integers of redial ids for movies
		temp_answers_dict = {}
		for db_id in answers_dict:
			redial_id = self.database_id_to_redial_id[int(db_id)]
			temp_answers_dict[redial_id] = answers_dict[db_id]

		answers_dict = temp_answers_dict

		#  -----------------------------------------------------------------------------------------------------------------

		conversationId = int(conversation["conversationId"])

		dialogue = []

		senders = []

		# a list that contains one list for every message. The inner list, containes tuples in the form (token_index, ReDial movie_id), for every mentioned movie in that message
		movie_mentions = []

		conversation_messages_in_token_ids = []

		for message_index, message in enumerate(conversation["messages"]):
			# role of the sender of message: 1 for seeker, -1 for recommender
			role = 1 if message["senderWorkerId"] == conversation["initiatorWorkerId"] else -1
			senders.append(role)
			# remove "@" and add spaces around movie mentions to be sure to count them as single tokens
			# tokens that match /^\d{5,6}$/ are movie mentions

			movie_mentions_in_message = re.findall(r"@[0-9]+", message["text"])

			# translate all db movie_ids with redial movie ids
			movie_mentions_in_message = [self.database_id_to_redial_id[ int(id[1:]) ] for id in movie_mentions_in_message]

			message_text = re.sub(r"@[0-9]+", " Movie_Mentioned ", message["text"]) 

			# surounding the message with Start of Sentence and End of Sentence tokens 
			message_text = "SOS " + message_text + " EOS"

			# get indexes of movies mentioned in the sentence

			encoded_text = self.encode(message_text)

			# find positions of mentioned movies, within the tokenized text
			movie_mentions_with_token_indexes = []
			token_index = 0
			while len(movie_mentions_with_token_indexes) != len(movie_mentions_in_message):

				if encoded_text[token_index] == self.encode("Movie_Mentioned")[0]:
					# storing a tuple of the form (token_index, redial_movie_id)
					movie_mentions_with_token_indexes.append( (token_index, movie_mentions_in_message[ len(movie_mentions_with_token_indexes) ]) )

				token_index += 1

			movie_mentions.append(movie_mentions_with_token_indexes)

			dialogue += [encoded_text]

		# initialize category vector with zeros
		category_target = np.zeros(len(self.categories))

		for movies_of_message in movie_mentions:
			for (token_index, redial_movie_id) in movies_of_message:
				# retrieve movie category vector
				movie_category_vector = self.redial_id_to_categories[redial_movie_id]
				# retrieve sentiment if there is one
				if redial_movie_id in answers_dict:
					# liked the movie
					if answers_dict[redial_movie_id]["liked"] == 1:
						sentiment = 1
					# didn't like
					elif answers_dict[redial_movie_id]["liked"] == 0:
						sentiment = -1
					# didn't say (answers_dict[redial_movie_id]["liked"] == 2)
					else:
						sentiment = 0
				else:
					# if the answer forms of this conversation are missing, then we are simply taking the average category vector of the mentioned movies
					sentiment = 1
				# add movie category vector, multiplied by the sentiment (sentiment is in set {-1,1}), to the conversation category vector
				category_target += movie_category_vector * sentiment

		# apply softmax
		category_target = np.exp(category_target)/sum(np.exp(category_target))

		dialogue, senders, movie_mentions, category_target = self.truncate(dialogue, senders, movie_mentions, category_target)

		return dialogue, senders, movie_mentions, category_target, answers_dict


	def truncate(self, dialogue, senders, movie_mentions, category_target):
		#  dialogue, target, senders, movie_occurrences
		# truncate conversations that have too many utterances
		if len(dialogue) > self.conversation_length_limit:
			dialogue = dialogue[:self.conversation_length_limit]
			senders = senders[:self.conversation_length_limit]
			movie_mentions = movie_mentions[:self.conversation_length_limit]
			category_target = category_target[:self.conversation_length_limit]

		# truncate utterances that are too long
		for (i, utterance) in enumerate(dialogue):
			if len(utterance) > self.utterance_length_limit:
				# we make sure that the last token remains the EOS token
				dialogue[i] = dialogue[i][:self.utterance_length_limit - 1] + [ dialogue[i][-1] ] 
		return dialogue, senders, movie_mentions, category_target


	def load_batch(self, subset="train"):
		# get batch
		batch_data = self.conversation_data[subset][self.batch_index[subset] * self.conversations_per_batch:
													(self.batch_index[subset] + 1) * self.conversations_per_batch]
		# update batch index
		self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]

		return self.load_batch_FLAT(batch_data)




	def load_batch_FLAT(self, batch_data):


		# if self.HIBERT then these two lists will contain lists of items, otherwise they will contain items
		batch_contexts = []
		batch_token_type_ids = []
		# One way or another, this list will have a |C| dimentional vector per sample
		batch_category_targets = []
		# these indexes will be used for pooling hidden representations of MovieMentioned tokens in order to output Sentiment Analysis predictions for each
		batch_movie_mentions = []

		# this is for predicted movie mentions from recommender
		complete_sample_movie_targets = []

		batch_indices_of_MM = []
		batch_movie_ids = []

		max_length = 0


		for conversation in batch_data:
			# retrieve conversation data
			if self.process_at_instanciation:
				dialogue, senders, movie_mentions, category_target, answers_dict = conversation
			else:
				dialogue, senders, movie_mentions, category_target, answers_dict = self.extract_dialogue4Bert(conversation)

			# for each message
			for i in range(len(senders)):
				# if this message was sent by the recommender
				if senders[i] == -1:
					context = []
					token_type_ids = []
					movie_mentions_temp = []

					# for every message preceding this message
					for j in reversed(range(i)):

						token_type_id = 0 if senders[j] == -1 else 1

						# we do not want the context + the current message to exceed the input lenght limit
						if len(context) + 1 + len(dialogue[j]) + 1 + len(dialogue[i]) >= self.max_input_length:
							break
						# we reversively add previous messages as context, separated by the "SEP" special token
						context = [self.encode("SEP")[0]] + dialogue[j] + context

						token_type_ids = [token_type_id] * ( len(dialogue[j]) + 1 ) + token_type_ids

						# update movie_mentions tokens given that we added at the begining of a context a new message, so the length of the new message needs to be added on their indexes
						movie_mentions_temp = [ (index + len(dialogue[j]) + 1, redial_movie_id) for (index, redial_movie_id) in movie_mentions_temp]
						# then we add the movie mentions token indexes of the new message
						movie_mentions_temp =  [ (token_index + len(self.cls_token_ids), redial_movie_id) for (token_index, redial_movie_id) in movie_mentions[j]  ]  +  movie_mentions_temp

					# and then we add the cls token ids at the begining
					context = self.cls_token_ids + context

					# we always se the CLS tokens to have token id equal to 0, so that it is homeomorphic
					token_type_ids = [0] * (len(self.cls_token_ids)) + token_type_ids

					# If the context is non empty (contains at least one sentence), then we properly create category and sentiment analysis targets
					if len(context) != len(self.cls_token_ids):

						# we remove the "SEP" token which has been added at the begining by the last added message on the context
						del context[len(self.cls_token_ids)]
						del token_type_ids[len(self.cls_token_ids)]

						# finally, we add the the current sentence masked to the context.
						context = context + [self.encode("SEP")[0]] + [self.encode("SOS")[0]] + len(dialogue[i][ 2 :])*[self.encode("MASK")[0]]

						token_type_id = 0 if senders[i] == -1 else 1
						token_type_ids = token_type_ids + [token_type_id] * (len(dialogue[i])) 

					# if there is no context, then we set the semmantic targets to  -1 
					else:
						# If the recommender starts the conversation, then we do not create a sample
						continue


					# We create exactly one sample per recommended movie (from the recommender)
					for idx, movie_id in movie_mentions[i]:

					# instead of givint the conversation CPD, we give the item's binary category vector
						target_binary_cat_vector = self.redial_id_to_categories[movie_id]

						# if category vector of item is unknown, we set the target to 0.5, which is he "neutral" of sigmoid
						if sum(target_binary_cat_vector) == len(self.categories):
							target_binary_cat_vector = np.ones(len(self.categories))/2

						temp_category_target = target_binary_cat_vector
						batch_contexts.append(context)
						batch_token_type_ids.append(token_type_ids)
						batch_category_targets.append(temp_category_target)
						batch_movie_mentions.append(movie_mentions_temp)

						complete_sample_movie_targets.append(torch.tensor(movie_id))

		# There is at least one case where a dialogue contains only one sentence, we skip that, as it is not a proper conversation
		if len(batch_contexts) == 0:
			return None

		num_of_samples = len(batch_contexts)

		max_length = np.max( [len(context) for context in batch_contexts ])


		# allocating tensors for saving the samples into containers of equal size
		contexts = np.full((num_of_samples, max_length), fill_value = self.encode("PAD")[0], dtype=np.int64)

		token_types = np.full((num_of_samples, max_length), fill_value=0, dtype=np.int64)

		attention_masks = np.full((num_of_samples, max_length), fill_value = 0, dtype=np.bool_)

		category_targets = np.zeros((num_of_samples, len(self.categories)), dtype=np.float32)

		for i in range(num_of_samples):
			# fill in the values in the containers
			contexts[i, : len(batch_contexts[i])] = batch_contexts[i]
			token_types[i, : len(batch_token_type_ids[i])] = batch_token_type_ids[i]
			attention_masks[i, : len(batch_contexts[i])] = True
			category_targets[i] = batch_category_targets[i]

		contexts = torch.tensor(contexts)
		token_types = torch.tensor(token_types)
		attention_masks = torch.tensor(attention_masks)
		category_targets = torch.tensor(category_targets)

		complete_sample_movie_targets = torch.stack(complete_sample_movie_targets)

		batch = {}

		batch["contexts"] = contexts
		batch["token_types"] = token_types
		batch["attention_masks"] = attention_masks
		batch["category_targets"] = category_targets
		batch["batch_movie_mentions"] = batch_movie_mentions
		batch["complete_sample_movie_targets"] = complete_sample_movie_targets
		
		return batch

