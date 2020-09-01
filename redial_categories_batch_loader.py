import config
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

from categories_dataset import Categories_Dataset

import transformers

import sys
import nltk





def tokenize(message):
    """
    Text processing: Sentence tokenize, then concatenate the word_tokenize of each sentence. Then lower.
    :param message:
    :return:
    """
    sentences = nltk.sent_tokenize(message)
    tokenized = []
    for sentence in sentences:
        tokenized += nltk.word_tokenize(sentence)
    return [word.lower() for word in tokenized]



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


def process_ratings(rating, ratings01=True):
    if ratings01:
        return rating
    if rating == 0:
        return 0.1
    elif rating == 1:
        return 0.8
    else:
        raise ValueError("Expected rating = 0 or 1. Instaed, received {}".format(rating))

            # batch_loader = DialogueBatchLoader(sources="ratings", conversations_per_batch=16)
class DialogueBatchLoader4Transformers(object):
    def __init__(self, conversations_per_batch, HIBERT = False, use_pretrained = False, CLS_mode = "1_CLS",
                 max_input_length = 512,
                 conversation_length_limit=config.CONVERSATION_LENGTH_LIMIT,
                 utterance_length_limit=config.UTTERANCE_LENGTH_LIMIT,
                 training_size=-1,
                 data_path=config.REDIAL_DATA_PATH,
                 train_path=config.TRAIN_PATH,
                 valid_path=config.VALID_PATH,
                 test_path=config.TEST_PATH,
                 movie_path=config.MOVIE_PATH,
                 vocab_path=config.VOCAB_PATH,
                 shuffle_data=False,
                 process_at_instanciation=False,
                 vocab_file = "vocabulary/nli_large_vocab.pkl",
                 special_tokens_list = ['SOS', 'EOS', 'MASK', 'PAD', 'UNK', 'SEP', 'Movie_Mentioned']):
        # sources paramater: string "dialogue/sentiment_analysis [movie_occurrences] [movieIds_in_target]"
        # self.sources = sources
        self.conversations_per_batch = conversations_per_batch
        self.batch_index = {"train": 0, "valid": 0, "test": 0}
        self.conversation_length_limit = conversation_length_limit
        self.utterance_length_limit = utterance_length_limit
        self.training_size = training_size
        self.data_path = {"train": os.path.join(data_path, train_path),
                          "valid": os.path.join(data_path, valid_path),
                          "test": os.path.join(data_path, test_path)}
        self.movie_path = movie_path
        self.vocab_path = vocab_path
        self.word2id = None
        self.shuffle_data = shuffle_data
        # if true, call extract_dialogue when loading the data. (for training on several epochs)
        # Otherwise, call extract_dialogue at batch loading. (faster for testing the code)
        self.process_at_instanciation = process_at_instanciation

        self.max_input_length = max_input_length
        self.HIBERT = HIBERT
        self.use_pretrained = use_pretrained
        self.CLS_mode = CLS_mode


        self.load_movie_information_with_category_vectors()


        if self.CLS_mode == "1_CLS":
            self.cls_tokens = ["CLS"]
        else:
            self.cls_tokens = [ "CLS_Cat_" + str(i) for i in range(len(self.categories)) ]

        self.special_tokens_list = special_tokens_list + [ "CLS_Cat_" + str(i) for i in range(len(self.categories)) ] + ["CLS"]

        # load data
        self.conversation_data = {key: load_data(val) for key, val in self.data_path.items()}
        # set up number of batches per subset
        self.n_batches = {key: len(val) // self.conversations_per_batch for key, val in self.conversation_data.items()}


        # set up the tokenizers

        if self.use_pretrained:
            # tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")#, additional_special_tokens = special_tokens)
            tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")#, additional_special_tokens = special_tokens)

            print("Initial Vocab Size:", tokenizer.vocab_size)

            tokenizer.add_tokens(self.special_tokens_list)

            self.gpt2_tokenizer = tokenizer

            self.vocabulary_size = tokenizer.vocab_size + len(self.special_tokens_list)


            print("Final Vocab Size:", self.vocabulary_size)


            # exit()

        else:

            # word2id and id2word dictionaries are being set
            self.train_vocabulary = self._get_vocabulary()
            # load vocabulary
            vocab = pickle.load( open(vocab_file, 'rb'),  encoding='latin1')
            # Word to index mappings
            word2id = vocab['word2id']              # id2word = vocab['id2word']

            self.set_word2id(word2id)

            self.vocabulary_size = len(self.word2id)

        self.cls_token_ids = [self.encode(cls_token)[0] for cls_token in self.cls_tokens]

        self.max_conv_len = 0
        self.max_utt_len = 0

        if self.process_at_instanciation:
            # pre-process dialogues
            self.conversation_data = {key: [self.extract_dialogue4Bert(conversation)
                                            for conversation in val]
                                      for key, val in self.conversation_data.items()}



    def encode(self, text):
        if self.use_pretrained:
            token_ids = self.gpt2_tokenizer.encode(text)[1:-1]
            if isinstance(token_ids, int):
                return [token_ids]
            else:
                return token_ids
        else:

            # tokenize string using nltk tokenizer
            sentences = nltk.sent_tokenize(text)
            tokenized = []
            for sentence in sentences:
                tokenized += nltk.word_tokenize(sentence)
            tokenized =  [word if word in self.special_tokens_list else word.lower() for word in tokenized]

            # translate tokens to token_ids
            tokenized_ids = []
            for token in tokenized:
                if token in self.word2id:
                    tokenized_ids.append(self.word2id[token])
                    
                elif isinstance(token, int):
                    tokenized_ids.append(token)
                else:
                    tokenized_ids.append(self.word2id['UNK'])

            return tokenized_ids


    def decode(self, list_of_token_ids):
        if self.use_pretrained:
            return self.gpt2_tokenizer.decode(list_of_token_ids, clean_up_tokenization_spaces=False)
        else:
            if isinstance(list_of_token_ids, list):
                decoded_text = ""
                for token_id in list_of_token_ids:
                    decoded_text += " " + self.id2word[token_id]
                return decoded_text
            else:
                if list_of_token_ids in self.id2word:
                    return self.id2word[list_of_token_ids]
                else:
                    return list_of_token_ids


    def load_movie_information_with_category_vectors(self, filename = "movie_details_full.csv"):
        # load data from movie_details.csv and create the appropriate dictionaries
        # Database_id, MovieLens_Id, Movie_Title, List_of_Gernes [ the value of the last column will be the gerne vector ]

        self.database_id_to_name = {}
        self.database_id_to_movieLens_id = {}
        self.movieLens_id_to_database_id = {}
        self.database_id_to_cateory_vector = {}
        self.database_id_to_redial_id = {}
        self.redial_id_to_database_id = {}
        self.redial_id_to_categories = {}

        filepath = os.path.join(config.REDIAL_DATA_PATH, filename)

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
                        self.database_id_to_name[database_id] = movie_name
                        self.database_id_to_movieLens_id[database_id] = movie_lens_id
                        self.database_id_to_cateory_vector[database_id] = movie_category_vector
                        self.database_id_to_redial_id[database_id] = redial_id
                    if movie_lens_id != -1:
                        self.movieLens_id_to_database_id[movie_lens_id] = database_id
                    if redial_id != -1:
                        self.redial_id_to_database_id[redial_id] = database_id
                        self.redial_id_to_categories[redial_id] = movie_category_vector

        self.n_movies = len(self.redial_id_to_database_id)



    def get_categories_movies_matrix(self):
        # finally, return the category-movie matrix [|C| X |M|]
        categories_movies_matrix = []
        for i in range(self.n_movies):
            categories_movies_matrix.append(self.redial_id_to_categories[i])

        return np.transpose( np.array(categories_movies_matrix) )



    def _get_vocabulary(self):
        """
        get the vocabulary from the train data
        :return: vocabulary
        """
        if os.path.isfile(self.vocab_path):
            print("Loading vocabulary from {}".format(self.vocab_path))
            with open(self.vocab_path, 'rb') as f:
                return  pickle.load(f, encoding='utf-8')
        print("Loading vocabulary from data")
        counter = Counter()
        # get vocabulary from dialogues
        for subset in ["train", "valid", "test"]:
            for conversation in tqdm(self.conversation_data[subset]):
                for message in conversation["messages"]:
                    # remove movie Ids
                    pattern = re.compile(r'@(\d+)')
                    text = tokenize(pattern.sub(" ", message["text"]))
                    counter.update([word.lower() for word in text])
        # get vocabulary from movie names
        for movieId in self.database_id_to_name:
            if self.database_id_to_redial_id[movieId] == -1:
                continue
            tokenized_movie = tokenize(self.database_id_to_name[movieId])
            counter.update([word.lower() for word in tokenized_movie])
        # Keep the most common words
        kept_vocab = counter.most_common(15000)
        vocab = [x[0] for x in kept_vocab]
        print("Vocab covers {} word instances over {}".format(
            sum([x[1] for x in kept_vocab]),
            sum([counter[x] for x in counter])
        ))
        vocab += self.special_tokens_list

        with open(self.vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        print("Saved vocabulary in {}".format(self.vocab_path))
        return vocab


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


        # retrieve_conversation_ratings
        # conv_ratings = {}

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

            # token_ids_of_message = [self.token2id(token) for token in encoded_text]

        # # if we are missing answer forms from both users, (0.29% of total conversations (33 out of 11348)),
        # # then we assume that all movies were watched and liked, and treat it as noise
        # # we assume them liked, because 94.3% of the movies in this dataset is liked, known phenomenon in recommending datasets as ReDial authors mention
        # if len(answers_dict) == 0:

        #     # if len(movie_mentions) == 0:
        #     #     for c in conversation:
        #     #         print(c)
        #     #         if c == "messages":
        #     #             for m in conversation[c]:
        #     #                 print(m)
        #     #         else:
        #     #             print(conversation[c])
        #     #     print()

        #     for i, movies_of_message in enumerate(movie_mentions):
        #         for (token_index, redial_movie_id) in movies_of_message:
        #             answers_dict[redial_movie_id] = {'seen': 1, 'liked': 1}
        #             # we retrieve who suggested them by looking a the sender id of that message
        #             # role of the sender of message: 0 for seeker, 1 for recommender
        #             answers_dict[redial_movie_id]["suggested"] = 0 if senders[i] == 1 else 1


        # form user category vector:

        # if we are missing sentiment for the movies, then we asume that all mentioned movies were liked by the user,
        # we do so because the statistics of the dateset sugests so (94.3% of the mentioned movies are noted as "liked" regarding the seekers preferences)

        # initialize category vector with zeros
        category_target = np.zeros(len(self.categories))

        for movies_of_message in movie_mentions:
            for (token_index, redial_movie_id) in movies_of_message:
                # retrieve movie category vector
                movie_category_vector = self.redial_id_to_categories[redial_movie_id]
                # retrieve sentiment if there is one
                if redial_movie_id in answers_dict:
                    sentiment = 1 if answers_dict[redial_movie_id]["liked"] == 1 else -1
                else:
                    # if the answer forms of this conversation are missing, then we are simply taking the average category vector of the mentioned movies
                    sentiment = 1
                # add movie category vector, multiplied by the sentiment (sentiment is in set {-1,1}), to the conversation category vector
                category_target += movie_category_vector * sentiment

        # apply softmax
        category_target = np.exp(category_target)/sum(np.exp(category_target))


        dialogue, senders, movie_mentions, category_target = self.truncate(dialogue, senders, movie_mentions, category_target)

        # if len(dialogue) > self.max_conv_len:
        #     self.max_conv_len = len(dialogue)
        # if len(dialogue) != 0 and max([len(utt) for utt in dialogue ]) > self.max_utt_len:
        #     self.max_utt_len = max([len(utt) for utt in dialogue ])

        return dialogue, senders, movie_mentions, category_target, answers_dict


    def truncate(self, dialogue, senders, movie_mentions, category_target):
        #  dialogue, target, senders, movie_occurrences
        # truncate conversations that have too many utterances
        if len(dialogue) > self.conversation_length_limit:
            dialogue = dialogue[:self.conversation_length_limit]
            senders = senders[:self.conversation_length_limit]
            movie_mentions = movie_mentions[:self.conversation_length_limit]
            category_target = category_target[:self.conversation_length_limit]



            # dialogue = dialogue[:self.conversation_length_limit]
            # target = target[:self.conversation_length_limit]
            # senders = senders[:self.conversation_length_limit]
            # if "movie_occurrences" in self.sources:
            #     movie_occurrences = {
            #         key: val[:self.conversation_length_limit] for key, val in movie_occurrences.items()
            #     }

        # truncate utterances that are too long
        for (i, utterance) in enumerate(dialogue):
            if len(utterance) > self.utterance_length_limit:
                # we make sure that the last token remains the EOS token
                dialogue[i] = dialogue[i][:self.utterance_length_limit - 1] + [ dialogue[i][-1] ] 
                # target[i] = target[i][:self.utterance_length_limit]
                # if "movie_occurrences" in self.sources:
                #     for movieId, value in movie_occurrences.items():
                #         value[i] = value[i][:self.utterance_length_limit]

        return dialogue, senders, movie_mentions, category_target





    def set_word2id(self, temp_word2id):
        self.word2id = {}
        i = 0

        for key in enumerate(temp_word2id):
            if key in self.train_vocabulary:
                self.word2id[key] = i
                i += 1

        for key in self.train_vocabulary:
            if key not in self.word2id:
                self.word2id[key] = i
                i += 1

        for s_token in self.special_tokens_list:
            if s_token not in self.word2id:
                self.word2id[s_token] = i
                i += 1

        self.id2word = {id: word for (word, id) in self.word2id.items()}


    def load_batch(self, subset="train", complete = False):

        # get batch
        batch_data = self.conversation_data[subset][self.batch_index[subset] * self.conversations_per_batch:
                                                    (self.batch_index[subset] + 1) * self.conversations_per_batch]

        # update batch index
        self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]


        if self.HIBERT:
            return self.load_batch_HIBERT(batch_data, complete)
        else:
            return self.load_batch_FLAT(batch_data, complete)



    def load_batch_HIBERT(self, batch_data, complete = False):

        # if self.HIBERT then these two lists will contain lists of items, otherwise they will contain items
        batch_contexts = []
        # batch_token_type_ids = []

        # One way or another, this list will have a |C| dimentional vector per sample
        batch_category_targets = []

        # these indexes will be used for pooling hidden representations of MovieMentioned tokens in order to output Sentiment Analysis predictions for each
        # batch_movie_mentioned_indexes = []

        batch_nlg_targets = []
        batch_nlg_gt_input = []


        batch_hidden_representations_to_pool_mask = []
        batch_dialoge_trans_positional_embeddings = []
        batch_sentiment_analysis_targets = []
        batch_dialogue_trans_token_type_ids = []
        batch_NLG_pooled_tokens_mask = []
        batch_movie_mentions = []
        batch_CLS_pooled_tokens_mask = []


        # this is for predicted movie mentions from recommender
        complete_sample_movie_targets = []



        max_length = 0


        for conversation in batch_data:
            # retrieve conversation data
            if self.process_at_instanciation:
                dialogue, senders, movie_mentions, category_target, answers_dict = conversation
            else:
                dialogue, senders, movie_mentions, category_target, answers_dict = self.extract_dialogue4Bert(conversation)

            # for each message
            for i in range(len(senders)):
                # print(self.decode(dialogue[i]))
                # print(senders[i])
                # # print(category_target)
                # print(movie_mentions[i])

                # if this message was sent by the recommender
                if senders[i] == -1:
                    context = []
                    token_type_ids = []

                    movie_mentions_temp = []

                    # for every message preceding this message
                    for j in range(i):

                        token_type_id = 0 if senders[j] == -1 else 1

                        # cls tokens precede every message
                        context.append( self.cls_token_ids + dialogue[j] )

                        # token type ids will be used by the conversation transformer
                        token_type_ids.append(token_type_id)

                        # movie mentions_indeces will be used by the message transformer, in order to pool the hidden representations of the Movie_Mentioned tokens
                        movie_mentions_temp.append( [ (token_index + len(self.cls_token_ids), redial_movie_id)  for (token_index, redial_movie_id) in movie_mentions[j]  ] )
                        # the hidden representations of the cls tokens will also be pooled, but we know their exact positions in every message *at the begining [0 : len(self.cls_token_ids)]

                    # append token id (sender) for current message
                    token_type_ids.append( 0 if senders[i] == -1 else 1)
 
                    # print(movie_mentions_temp)



                    # we need to add cls tokens at the begining of every message
                    # we need to add masked nlg input at the end of the messages
                    # we need to have an nlg target (current message, shifted by one to the left)
                    # we need to have semantic targets ( one category target for the sample, and one sentiment analysis targert for each MovieMentioned token on the appropriate position)
                    # we need to have a list of indexes for MovieMentined tokens for every message
                    # we need to have token type ids for for each message
                    # we need to have positional ids for the conversation transformer, (calcualte the number of tokens per message depending on cls tokens and MovieMentioned tokens)
                    # and make sure that all tokens of the same message have the same positional ids


                    # regarding attentions:
                        # message transformer: applies attention to all given tokens (to the extend of the message length + special tokens)
                        # dialoge transformer:
                            # for Sentiment task, dialogue transformer applies attention to hidden representations of all pooled tokens
                            # for NLG task, dialogue transformer applies attention only to hidden representations of EOS pooled tokens
                            # for combining tasks, obviously the union of the above two would take place, so attention to all pooled hidden representations

                    # so we only need to specify the attention mask for NLG, that only pays attention to the EOS tokens' hid repr

                    # if we do not have seekers utterance yet, we set the category target to 0
                    if 1 not in senders[ : -1]:
                        temp_category_target = - np.ones(len(self.categories))
                    else:
                        temp_category_target = category_target


                    # the input of the dialogue transformer will have the form :
                    # [CLS TOKENS] [MovieMentioned]* len(movie_mentions_temp[0]) [EOS] [CLS TOKENS] [MovieMentioned]* len(movie_mentions_temp[1]) [EOS] ... 
                    # where [TOKEN] represents the hidden representation of message tranformer for token [TOKEN], and [CLS TOKENS] is either [CLS] or [CLS_Cat_0, CLS_Cat_1, ..., CLS_Cat_|C|]

                    hidden_representations_to_pool_mask = []

                    # we set the positional embeddings for the dialogue transformer, in order to make sure that hidden representations that were pooled form the same message will have same positional embeddings
                    dialoge_trans_positional_embeddings = []

                    sentiment_analysis_targets = []

                    dialogue_trans_token_type_ids = []

                    NLG_pooled_tokens_mask = []

                    CLS_pooled_tokens_mask = []

                    movie_mentions_for_dialogue_pooled_indexes = []
                    # print("--------------------------------------------------------------------------")

                    # for each message in the context
                    for j in range(len(context)):
                        # print(movie_mentions_temp[j])
                        # print(self.decode(context[j]))
                        # increasing index, by number of pooled tokens so far
                        # temp_movie_mentions_for_dialogue_pooled_indexes = []

                        movie_mentions_for_dialogue_pooled_indexes.append( [(i + len(self.cls_tokens) + len(sentiment_analysis_targets), redial_movie_id) for i, (token_index, redial_movie_id) in enumerate(movie_mentions_temp[j])] )

                        # we set the sentiment analysis targets of CLS TOKENS to -1
                        sentiment_analysis_targets += [-1] * len(self.cls_tokens)
                        # we set the proper sentiment analysis targets for the MovieMentioned tokens. If the sentiment analysis targets are missing, we set them to -1
                        sentiment_analysis_targets += [answers_dict[redial_movie_id]["liked"] if redial_movie_id in answers_dict else -1 for (token_index, redial_movie_id) in movie_mentions_temp[j]]
                        # we set the sentiment analysis targets of EOS TOKENS to -1
                        sentiment_analysis_targets += [-1]


                        # we instantiate a list of 0s equal to the length of current message (including special tokens)
                        hidden_representations_to_pool_from_this_message_mask = [0] * len(context[j])

                        # we pool the hidden representations of the CLS TOKENS
                        for z in range(len(self.cls_tokens)):
                            hidden_representations_to_pool_from_this_message_mask[z] = 1

                        # we pool the hidden representations of the MovieMentioned tokens
                        for (token_index, redial_movie_id) in  movie_mentions_temp[j]:
                            hidden_representations_to_pool_from_this_message_mask[token_index] = 1

                        # finally we pool the hidden representation of the EOS token, which is always the last token
                        hidden_representations_to_pool_from_this_message_mask[-1] = 1

                        hidden_representations_to_pool_mask.append( hidden_representations_to_pool_from_this_message_mask)

                        total_pooled_tokens_from_this_message = len(self.cls_tokens) + len(movie_mentions_temp[j]) + 1 

                        dialoge_trans_positional_embeddings += [j] * total_pooled_tokens_from_this_message

                        dialogue_trans_token_type_ids += [token_type_ids[j]] * total_pooled_tokens_from_this_message

                        CLS_tokens_mask_for_this_message = [1 if i < len(self.cls_tokens) else 0 for i in range(total_pooled_tokens_from_this_message)]


                        CLS_pooled_tokens_mask += CLS_tokens_mask_for_this_message

                        NLG_pooled_tokens_mask += [0] * total_pooled_tokens_from_this_message
                        # we set the mask to 1 for the last token that corresponds to the EOS token
                        NLG_pooled_tokens_mask[-1] = 1





                    # we add tha current message to the token for nlg input (everything besides "SOS", will be masked)
                    # the current message also starts with the CLS tokens in order to have homiomorphy over all messages,
                    # which would affect for example the possitional embeddings, and the undersanding of the model
                    current_message_masked = self.cls_token_ids + [self.encode("SOS")[0]] + len(dialogue[i][ 2 :])*[self.encode("MASK")[0]]
                    context.append( current_message_masked )
                    # we do not care about the EOS token, as it will not be used during evaluation time, due to the fact that it has no target
                    nlg_gt_input = self.cls_token_ids + dialogue[i][:-1]

                    # print(current_message_masked)
                    # the last context is the current message.
                    # From this message we pool all tokens, so that the MASK tokens, will be predicted by the dialogue encoder
                    # last_message_representations_to_pool_mask = [1 if i < len(self.cls_tokens) else 0 for i in range(len(current_message_masked)) ]
                    last_message_representations_to_pool_mask = [1] * len(current_message_masked)



                    hidden_representations_to_pool_mask.append(last_message_representations_to_pool_mask)
                    # taret is ground truth message shifted by 1 to the left


                    total_pooled_tokens_from_this_message = len(current_message_masked)

                    # we also pool the CLS tokens of the current message (1 for the CLS tokens, 0 for the SOS and MASK tokens)
                    CLS_pooled_tokens_mask += len(self.cls_tokens) * [1] + ( total_pooled_tokens_from_this_message - len(self.cls_tokens)) * [0]

                    dialoge_trans_positional_embeddings += [len(context) -1 ] * total_pooled_tokens_from_this_message

                    # the last message does not have any sentiment analysis tokens
                    sentiment_analysis_targets += [-1] * total_pooled_tokens_from_this_message

                    dialogue_trans_token_type_ids += [token_type_ids[-1]] * total_pooled_tokens_from_this_message
                    # 0 for the CLS tokens, 1 for the SOS and MASK tokens
                    NLG_pooled_tokens_mask += [0] *len(self.cls_tokens) + [1] * ( total_pooled_tokens_from_this_message - len(self.cls_tokens))

                    # set the nlg target to have lenght equal to the dialogue transformer output
                    nlg_target = [-1] * len(dialoge_trans_positional_embeddings)
                    # set the nlg targets
                    ending_of_nlg_target = dialogue[i][ 1 :]
                    # copy the ending of the nlg target
                    for j in range(1, len(ending_of_nlg_target) +1 ):
                        nlg_target[-j] = ending_of_nlg_target[-j]




                    # if we are creating samples for the complete system that acts as a movie recommender (we create one sample for every MM in a recommender's response)
                    if complete:
                        # then we create exactly one sample per recommended movie (from the recommender)
                        for idx, movie_id in movie_mentions[i]:

                            batch_contexts.append(context)
                            batch_category_targets.append(temp_category_target)
                            batch_nlg_targets.append(nlg_target)
                            batch_nlg_gt_input.append(nlg_gt_input)
                            batch_hidden_representations_to_pool_mask.append(hidden_representations_to_pool_mask)
                            batch_dialoge_trans_positional_embeddings.append(dialoge_trans_positional_embeddings)
                            batch_sentiment_analysis_targets.append(sentiment_analysis_targets)
                            batch_dialogue_trans_token_type_ids.append(dialogue_trans_token_type_ids)
                            batch_NLG_pooled_tokens_mask.append(NLG_pooled_tokens_mask)
                            batch_movie_mentions.append(movie_mentions_for_dialogue_pooled_indexes)
                            batch_CLS_pooled_tokens_mask.append(CLS_pooled_tokens_mask)


                            complete_sample_movie_targets.append(torch.tensor(movie_id))

                    # if we are creating samples for SA, Cat prediction, or nlg (we create one sample for every recommender's response)
                    else:

                        batch_contexts.append(context)
                        batch_category_targets.append(temp_category_target)
                        batch_nlg_targets.append(nlg_target)
                        batch_nlg_gt_input.append(nlg_gt_input)
                        batch_hidden_representations_to_pool_mask.append(hidden_representations_to_pool_mask)
                        batch_dialoge_trans_positional_embeddings.append(dialoge_trans_positional_embeddings)
                        batch_sentiment_analysis_targets.append(sentiment_analysis_targets)
                        batch_dialogue_trans_token_type_ids.append(dialogue_trans_token_type_ids)
                        batch_NLG_pooled_tokens_mask.append(NLG_pooled_tokens_mask)
                        batch_movie_mentions.append(movie_mentions_for_dialogue_pooled_indexes)
                        batch_CLS_pooled_tokens_mask.append(CLS_pooled_tokens_mask)
                #     print()


                    # print("Sample: ")
                    # print("Contexts:")
                    # for j in range(len(context)):
                    #     print("Context: ", self.decode(context[j]))
                    # #     print("Context: ", len(context[j]))
                    # #     # print("Sender :", len(token_type_ids[j]))
                    #     print("hidden_representations_to_pool_mask:", hidden_representations_to_pool_mask[j])
                    # # print("hidden_representations_to_pool_mask:", np.sum(hidden_representations_to_pool_mask))

                    # print(":dialoge_trans_positional_embeddings", dialoge_trans_positional_embeddings)
                    # # print(":sentiment_analysis_targets", len(sentiment_analysis_targets))
                    # print(":dialogue_trans_token_type_ids", dialogue_trans_token_type_ids)
                    # # print(":NLG_pooled_tokens_mask", len(NLG_pooled_tokens_mask))
                    # # print(":CLS_pooled_tokens_mask", len(CLS_pooled_tokens_mask))
                    # #     # print(":", )
                    # #     # print(":", )
                    # # print("nlg_target: ", len(nlg_target))
                    # # print()
                    # # print(temp_category_target)
                    # # exit()
                    # print()
                    # break

        #     print()
        #     for i in range(len(batch_contexts)):

        #         print(self.decode(batch_contexts[i]), len(batch_contexts[i]))
        #         print(batch_token_type_ids[i], len(batch_token_type_ids[i]))
        #         print(batch_category_targets[i])
        #         for index in batch_movie_mentioned_indexes[i]:
        #             print(self.decode(batch_contexts[i][index]))
        #         print(batch_movie_mentioned_indexes[i])
        #         # print(self.decode(batch_nlg_inputs[i]))

        #         printable_nlg_target = [ -1 if token_id == -1 else self.decode(token_id) for token_id in batch_nlg_targets[i]]
        #         # print(batch_nlg_targets[i])
        #         print(printable_nlg_target)
        #         print()

        # print()


        # then bring the batch into its final form (Padded tensors of same size, attentions [maybe Semantic att and NLG att], form index tensors for extracting the MM token hid reprs )

        # +1 in order to ensure that the nlg_gt_input wull fit, because the masked inputs and the targets are shorter by 1 w.r.t. nlg_gt_input
        # (due to predicting the next token given the current one, so there is no target for the last token etc.)

        # There is at least one case where a dialogue contains only one sentence, we skip that, as it is not a proper conversation
        if len(batch_contexts) == 0:
            return None


        max_message_length = np.max( [ np.max( [ len(message) for message in context]) for context in batch_contexts ] )

        max_dialogue_length = np.max( [len(context) for context in batch_contexts ])

        max_dialogue_trans_input = np.max([ len(pos_embeddings) for pos_embeddings in batch_dialoge_trans_positional_embeddings])

        # print("max_message_length: ", max_message_length)
        # print("max_dialogue_length: ", max_dialogue_length)
        # print("max_dialogue_trans_input: ", max_dialogue_trans_input)

        # exit()

        num_of_samples = len(batch_contexts)

        # allocate memory for all variables needed, that has same shape for every sample
        contexts = np.full((num_of_samples, max_dialogue_length, max_message_length), fill_value = self.encode("PAD")[0], dtype=np.int64)
        context_attentions = np.full((num_of_samples, max_dialogue_length, max_message_length), fill_value = 0, dtype=np.bool_)
        category_targets = np.zeros((num_of_samples, len(self.categories)), dtype=np.float32)
        nlg_targets = np.full((num_of_samples, max_dialogue_trans_input), fill_value = -1, dtype=np.int64)
        nlg_gt_inputs = np.full((num_of_samples, max_message_length), fill_value = self.encode("PAD")[0], dtype=np.int64)
        pool_hidden_representations_mask = np.full((num_of_samples, max_dialogue_length, max_message_length), fill_value = 0, dtype=np.bool_)
        dialogue_trans_positional_embeddings = np.full((num_of_samples, max_dialogue_trans_input), fill_value = 0, dtype=np.int64)
        # sentiment_analysis_targets = np.full((num_of_samples, max_dialogue_trans_input), fill_value = -1, dtype=np.int64)
        dialogue_trans_token_type_ids = np.full((num_of_samples, max_dialogue_trans_input), fill_value = 0, dtype=np.int64)
        nlg_dialogue_mask_tokens = np.full((num_of_samples, max_dialogue_trans_input), fill_value = 0, dtype=np.bool_)
        dialogue_trans_attentions = np.full((num_of_samples, max_dialogue_trans_input), fill_value = 0, dtype=np.bool_)
        sentiment_analysis_targets = np.full((num_of_samples, max_dialogue_trans_input), fill_value = -1, dtype=np.float32)
        CLS_pooled_tokens_mask = np.full((num_of_samples, max_dialogue_trans_input), fill_value = 0, dtype=np.bool_)

                    # batch_contexts.append(context)
                    # batch_category_targets.append(temp_category_target)
                    # batch_nlg_targets.append(nlg_target)
                    # batch_hidden_representations_to_pool_mask.append(hidden_representations_to_pool_mask)
                    # batch_dialoge_trans_positional_embeddings.append(dialoge_trans_positional_embeddings)
                    # batch_sentiment_analysis_targets.append(sentiment_analysis_targets)
                    # batch_dialogue_trans_token_type_ids.append(dialogue_trans_token_type_ids)
                    # batch_NLG_pooled_tokens_mask.append(NLG_pooled_tokens_mask)


        for i in range(num_of_samples):

            for j in range(len(batch_contexts[i])):
                contexts[i][j][ : len(batch_contexts[i][j])] = batch_contexts[i][j]
                context_attentions[i][j][ : len(batch_contexts[i][j])] = 1
                pool_hidden_representations_mask[i][j][ : len(batch_hidden_representations_to_pool_mask[i][j]) ] = batch_hidden_representations_to_pool_mask[i][j]

            category_targets[i] = batch_category_targets[i]
            nlg_targets[i, : len(batch_nlg_targets[i])] = batch_nlg_targets[i]
            nlg_gt_inputs[i, : len(batch_nlg_gt_input[i])] = batch_nlg_gt_input[i]
            dialogue_trans_positional_embeddings[i, : len(batch_dialoge_trans_positional_embeddings[i])] = batch_dialoge_trans_positional_embeddings[i]
            sentiment_analysis_targets[i, : len(batch_sentiment_analysis_targets[i])] = batch_sentiment_analysis_targets[i]
            dialogue_trans_token_type_ids[i, : len(batch_dialogue_trans_token_type_ids[i]) ] = batch_dialogue_trans_token_type_ids[i]
            nlg_dialogue_mask_tokens[i, : len(batch_NLG_pooled_tokens_mask[i]) ] = batch_NLG_pooled_tokens_mask[i]
            dialogue_trans_attentions[i, : len(batch_dialogue_trans_token_type_ids[i]) ] = 1
            CLS_pooled_tokens_mask[i, : len(batch_CLS_pooled_tokens_mask[i])] = batch_CLS_pooled_tokens_mask[i]

        contexts = torch.tensor(contexts)
        context_attentions = torch.tensor(context_attentions)
        category_targets = torch.tensor(category_targets)
        nlg_targets = torch.tensor(nlg_targets)
        nlg_gt_inputs = torch.tensor(nlg_gt_inputs)
        pool_hidden_representations_mask = torch.tensor(pool_hidden_representations_mask)
        dialogue_trans_positional_embeddings = torch.tensor(dialogue_trans_positional_embeddings)
        sentiment_analysis_targets = torch.tensor(sentiment_analysis_targets)
        dialogue_trans_token_type_ids = torch.tensor(dialogue_trans_token_type_ids)
        nlg_dialogue_mask_tokens = torch.tensor(nlg_dialogue_mask_tokens)
        dialogue_trans_attentions = torch.tensor(dialogue_trans_attentions)
        CLS_pooled_tokens_mask = torch.tensor(CLS_pooled_tokens_mask)

        if complete:
            complete_sample_movie_targets = torch.stack(complete_sample_movie_targets)



        batch = {}

        batch["contexts"] = contexts
        batch["context_attentions"] = context_attentions
        batch["category_targets"] = category_targets
        batch["nlg_targets"] = nlg_targets
        batch["nlg_gt_inputs"] = nlg_gt_inputs
        batch["pool_hidden_representations_mask"] = pool_hidden_representations_mask
        batch["dialogue_trans_positional_embeddings"] = dialogue_trans_positional_embeddings
        batch["sentiment_analysis_targets"] = sentiment_analysis_targets
        batch["dialogue_trans_token_type_ids"] = dialogue_trans_token_type_ids
        batch["nlg_dialogue_mask_tokens"] = nlg_dialogue_mask_tokens
        batch["dialogue_trans_attentions"] = dialogue_trans_attentions
        batch["CLS_pooled_tokens_mask"] = CLS_pooled_tokens_mask
        batch["batch_movie_mentions"] = batch_movie_mentions
        batch["complete_sample_movie_targets"] = complete_sample_movie_targets

        return batch

        # return contexts, context_attentions, category_targets, nlg_targets, nlg_gt_inputs, pool_hidden_representations_mask, dialogue_trans_positional_embeddings, \
            # sentiment_analysis_targets, dialogue_trans_token_type_ids, nlg_dialogue_mask_tokens, dialogue_trans_attentions, CLS_pooled_tokens_mask, batch_movie_mentions, complete_sample_movie_targets








    def load_batch_FLAT(self, batch_data, complete = False):


        # if self.HIBERT then these two lists will contain lists of items, otherwise they will contain items
        batch_contexts = []
        batch_token_type_ids = []

        # One way or another, this list will have a |C| dimentional vector per sample
        batch_category_targets = []

        # these indexes will be used for pooling hidden representations of MovieMentioned tokens in order to output Sentiment Analysis predictions for each
        batch_movie_mentions = []

        batch_nlg_targets = []

        batch_nlg_gt_input = []

        batch_sentiment_analysis_targets = []

        # this is for predicted movie mentions from recommender
        complete_sample_movie_targets = []



        max_length = 0


        for conversation in batch_data:
            # retrieve conversation data
            if self.process_at_instanciation:
                dialogue, senders, movie_mentions, category_target, answers_dict = conversation
            else:
                dialogue, senders, movie_mentions, category_target, answers_dict = self.extract_dialogue4Bert(conversation)

            # for each message
            for i in range(len(senders)):
                # print(self.decode(dialogue[i]))
                # print(senders[i])
                # print(category_target)
                # print(movie_mentions[i])

                # if this message was sent by the recommender
                if senders[i] == -1:
                    context = []
                    token_type_ids = []

                    movie_mentions_temp = []

                    # for every message preceding this message
                    for j in reversed(range(i)):

                        token_type_id = 0 if senders[j] == -1 else 1


                        # In case of FLAT and NOT HIBERT,  we do not want the context + the current message to exceed the input lenght limit
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

                        temp_category_target = category_target

                        # we remove the "SEP" token which has been added at the begining by the last added message on the context
                        del context[len(self.cls_token_ids)]
                        del token_type_ids[len(self.cls_token_ids)]



                        # we also define the nlg input, tha will begiven, for the perplexity to be calculated
                        nlg_gt_input = context + [self.encode("SEP")[0]] + dialogue[i][:-1]
                        # the targets of the context tokens are being set to -1. (context +1 for the "SEP" token)
                        nlg_target = ( len(context) + 1)*[-1] + dialogue[i][ 1 :] # + [self.encode("EOS")[0]]

                        # finally, we add the the current sentence masked to the context.
                        context = context + [self.encode("SEP")[0]] + [self.encode("SOS")[0]] + len(dialogue[i][ 2 :])*[self.encode("MASK")[0]]

                        token_type_id = 0 if senders[i] == -1 else 1
                        token_type_ids = token_type_ids + [token_type_id] * (len(dialogue[i])) 



                                                                                    # uniform distribution over categories
                                                                                    # This way we handle new user, by assuming a unifor distribution over category preference
                    # if there is no context, then we set the semmantic targets to  -1 
                    else:

                        temp_category_target = - np.ones(len(self.categories))


                        # we also define the nlg input, tha will begiven, for the perplexity to be calculated
                        nlg_gt_input = context + dialogue[i][:-1]

                        # the targets of the context tokens are being set to -1.
                        nlg_target = ( len(context))*[-1] + dialogue[i][ 1 :] # + [self.encode("EOS")[0]]


                        # finally, we add the the current sentence masked to the context.
                        context = context + [self.encode("SOS")[0]] + len(dialogue[i][ 2 :])*[self.encode("MASK")[0]]

                        # nlg_gt_input = context + [self.encode("SOS")[0]] + len(dialogue[i][ 2 :])*[self.encode("MASK")[0]]

                        token_type_id = 0 if senders[i] == -1 else 1
                        token_type_ids = token_type_ids + [token_type_id] * (len(dialogue[i]) -1 ) 



                    # printable_nlg_target = [ -1 if token_id == -1 else self.decode(token_id) for token_id in nlg_target]

                    # exit()
                    # print("LENGTH Context :", len(context))
                    # print("token_type_ids :", token_type_ids)
                    # print("LENGTH token_type_ids :", len(token_type_ids))
                    # # print("NLG target :", nlg_target)
                    # print("NLG target :", printable_nlg_target)
                    # print("LENGTH NLG target :", len(printable_nlg_target))



                    # set up the sentiment analysis targets for the sample
                    sentiment_analysis_targets = [-1] * len(context)

                    for (index, redial_movie_id) in movie_mentions_temp:
                        # if the sentiment analysis targets are missing, then we set the targets to 0
                        sentiment_analysis_targets[index] = answers_dict[redial_movie_id]["liked"] if redial_movie_id in answers_dict else -1

                    # make sentiment analysis targets

                    # the context is added to the nlg input
                    # the NLG input which starts with SOS, and continues with MASK tokens euqal to the length of the current message
                    # nlg_input = context + [self.encode("SEP")[0]] + [self.encode("SOS")[0]] + len(dialogue[i][ 1 :])*[self.encode("MASK")[0]]
                    # -1 are added for behalf of the context for the targets
                    # we also define the NLG target, which is the current message, shifted by 1 to the left
                    # nlg_target = ( len(context) + 1)*[-1] + dialogue[i][ 1 :] + [self.encode("EOS")[0]]

                    # print("NLG Input", nlg_input)
                    # print("NLG Target", nlg_target)


                    # print(" -- i:",i,", context = ", context)

                    # if we are creating samples for the complete system that acts as a movie recommender (we create one sample for every MM in a recommender's response)
                    if complete:
                        # then we create exactly one sample per recommended movie (from the recommender)
                        for idx, movie_id in movie_mentions[i]:

                            batch_contexts.append(context)
                            batch_token_type_ids.append(token_type_ids)
                            batch_category_targets.append(temp_category_target)
                            batch_movie_mentions.append(movie_mentions_temp)
                            batch_nlg_targets.append(nlg_target)
                            batch_nlg_gt_input.append(nlg_gt_input)
                            batch_sentiment_analysis_targets.append(sentiment_analysis_targets)

                            complete_sample_movie_targets.append(torch.tensor(movie_id))

                    # if we are creating samples for SA, Cat prediction, or nlg (we create one sample for every recommender's response)
                    else:

                        batch_contexts.append(context)
                        batch_token_type_ids.append(token_type_ids)
                        batch_category_targets.append(temp_category_target)
                        batch_movie_mentions.append(movie_mentions_temp)
                        batch_nlg_targets.append(nlg_target)
                        batch_nlg_gt_input.append(nlg_gt_input)
                        batch_sentiment_analysis_targets.append(sentiment_analysis_targets)





        # There is at least one case where a dialogue contains only one sentence, we skip that, as it is not a proper conversation
        if len(batch_contexts) == 0:
            return None



        num_of_samples = len(batch_contexts)

        max_length = np.max( [len(context) for context in batch_contexts ])


        # if complete:
        #     # we need to keep only the samples, where the recommender, recommends at least on movie
        #     # for each movie that the recommender proposes, we create one sample, with current conversation as context, and the proposed movie as target

        #     # indexes_to_keep = [False]* num_of_samples
        #     for i in range(number_of_samples):
        #         pass
        #         # for each Movie Mentioned token in the current sample




        # print(batch_contexts)
        # exit()



        # else:

        # allocating tensors for saving the samples into containers of equal size
        contexts = np.full((num_of_samples, max_length), fill_value = self.encode("PAD")[0], dtype=np.int64)

        token_types = np.full((num_of_samples, max_length), fill_value=0, dtype=np.int64)

        attention_masks = np.full((num_of_samples, max_length), fill_value = 0, dtype=np.bool_)

        category_targets = np.zeros((num_of_samples, len(self.categories)), dtype=np.float32)

        sentiment_analysis_targets = np.full((num_of_samples, max_length), fill_value = -1, dtype=np.float32)

        # attentions = np.full((num_of_samples, max_length), fill_value = 0, dtype=np.bool_)

        nlg_targets = np.full((num_of_samples, max_length), fill_value = -1, dtype=np.int64)

        nlg_gt_inputs = np.full((num_of_samples, max_length), fill_value = self.encode("PAD")[0], dtype=np.int64)

        for i in range(num_of_samples):
            # fill in the values in the containers
            contexts[i, : len(batch_contexts[i])] = batch_contexts[i]
            token_types[i, : len(batch_token_type_ids[i])] = batch_token_type_ids[i]
            attention_masks[i, : len(batch_contexts[i])] = True
            category_targets[i] = batch_category_targets[i]
            sentiment_analysis_targets[i, : len(batch_sentiment_analysis_targets[i])] = batch_sentiment_analysis_targets[i]
            nlg_targets[i, : len(batch_nlg_targets[i])] = len(batch_nlg_targets[i])
            nlg_gt_inputs[i, : len(batch_nlg_gt_input[i])] = len(batch_nlg_gt_input[i])


        contexts = torch.tensor(contexts)
        token_types = torch.tensor(token_types)
        attention_masks = torch.tensor(attention_masks)
        category_targets = torch.tensor(category_targets)
        sentiment_analysis_targets = torch.tensor(sentiment_analysis_targets)
        nlg_targets = torch.tensor(nlg_targets)
        nlg_gt_inputs = torch.tensor(nlg_gt_inputs)


        if complete:
            complete_sample_movie_targets = torch.stack(complete_sample_movie_targets)



        batch = {}

        batch["contexts"] = contexts
        batch["token_types"] = token_types
        batch["attention_masks"] = attention_masks
        batch["category_targets"] = category_targets
        batch["sentiment_analysis_targets"] = sentiment_analysis_targets
        batch["nlg_targets"] = nlg_targets
        batch["nlg_gt_inputs"] = nlg_gt_inputs
        batch["batch_movie_mentions"] = batch_movie_mentions
        batch["complete_sample_movie_targets"] = complete_sample_movie_targets
        
        return batch



        # return contexts, token_types, attention_masks, category_targets, sentiment_analysis_targets, nlg_targets, nlg_gt_inputs, batch_movie_mentions, complete_sample_movie_targets






# ------------------------------------------------------------------------------------------------------------------------------------











    def token2id(self, token):
        """
        :param token: string or movieId
        :return: corresponding ID
        """
        if token in self.word2id:
            return self.word2id[token]
        if isinstance(token, int):
            return token
        return self.word2id['[UNK]']






def get_cut_indices(movie_mentions, cut_width):
    """
    Get the utterance indices to cut the dialogue around the movie mentions
    At cut_width=0, return the index of the first mention, and the index of the last mention
    Higher cut_width adds utterances before the first mention, and after the last mention.
    :param movie_mentions:
    :param cut_width:
    :return:
    """
    utterance_mentions = [sum(utterance) > 0 for utterance in movie_mentions]
    first = next((i for (i, x) in enumerate(utterance_mentions) if x), 0)  # index of first occurrence
    last = next((i for (i, x) in enumerate(reversed(utterance_mentions)) if x), 0)  # reversed index of last occurrence
    last = len(utterance_mentions) - last
    return max(0, first - cut_width), min(len(utterance_mentions), last + cut_width)





def get_movies(path):
    # path = os.getcwd() + path
    id2name = {}
    db2id = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        # remove date from movie name
        date_pattern = re.compile(r'\(\d{4}\)')
        for row in reader:
            if row[0] != "index":
                id2name[int(row[0])] = date_pattern.sub('', row[1])
                db2id[int(row[2])] = int(row[0])
    del db2id[-1]
    print("loaded {} movies from {}".format(len(id2name), path))
    return id2name, db2id




    # def get_special_tokens2id_dict(self):
    #     return {"liked": self.liked_token_id, "disliked": self.disliked_token_id, "mask": self.mask_input_token_id}

    def _get_dataset_characteristics(self):
        # load movies. "db" refers to the movieId in the ReDial dataset, whereas "id" refers to the global movieId
        # (after matching the movies with movielens in match_movies.py).
        # So db2id is a dictionary mapping ReDial movie Ids to global movieIds
        self.id2name, self.db2id = get_movies(self.movie_path)
        self.db2name = {db: self.id2name[id] for db, id in self.db2id.items()}
        self.n_movies = len(self.db2id.values())  # number of movies mentioned in ReDial
        print('{} movies'.format(self.n_movies))
        # load data
        print("Loading and processing data")
        self.conversation_data = {key: load_data(val) for key, val in self.data_path.items()}

        
        # if self.training_size > 0:
        #     self.conversation_data["train"] = self.conversation_data["train"][:self.training_size]
        # if "sentiment_analysis" in self.sources:
        #     self.form_data = {key: self.extract_form_data(val) for key, val in self.conversation_data.items()}
        # if "ratings" in self.sources:
        #     self.ratings_data = {key: self.extract_ratings_data(val) for key, val in self.conversation_data.items()}

            # train_mean = np.mean([np.mean(list(conv.values())) for conv in self.ratings_data["train"]])
            # print("Mean training rating ", train_mean)
            # print("validation MSE made by mean estimator: {}".format(
            #     np.mean([np.mean((np.array(list(conv.values())) - train_mean) ** 2)
            #              for conv in self.ratings_data["valid"]])))
        # load vocabulary
        if self.use_pretrained == False:
            self.train_vocabulary = self._get_vocabulary()
            print("Vocabulary size : {} words.".format(len(self.train_vocabulary)))

        # if "dialogue_to_categories" in self.sources:
        data = self.conversation_data
        # elif "dialogue" in self.sources:
        #     data = self.conversation_data
        # elif "sentiment_analysis" in self.sources:
        #     data = self.form_data
        # elif "ratings" in self.sources:
        #     data = self.ratings_data

        if self.shuffle_data:
            # shuffle each subset
            for _, val in data.items():
                shuffle(val)


        self.n_batches = {key: len(val) // self.conversations_per_batch for key, val in data.items()}


        # self.N_train_data = len(data["train"])





    def extract_dialogue(self, conversation, flatten_messages=True):
        """
        :param conversation: conversation dictionary. keys : 'conversationId', 'respondentQuestions', 'messages',
         'movieMentions', 'respondentWorkerId', 'initiatorWorkerId', 'initiatorQuestions'
         :param flatten_messages
         :return:
        """
        dialogue = []
        target = []
        senders = []
        occurrences = None
        if "movie_occurrences" in self.sources:
            # initialize occurrences. Ignore empty movie names
            occurrences = {self.db2id[int(dbId)]: [] for dbId in conversation["movieMentions"]
                           if int(dbId) in self.db2name and not self.db2name[int(dbId)].isspace()}
        for message in conversation["messages"]:
            # role of the sender of message: 1 for seeker, -1 for recommender
            role = 1 if message["senderWorkerId"] == conversation["initiatorWorkerId"] else -1
            # remove "@" and add spaces around movie mentions to be sure to count them as single tokens
            # tokens that match /^\d{5,6}$/ are movie mentions
            pattern = re.compile(r'@(\d+)')
            message_text = pattern.sub(lambda m: " " + m.group(1) + " ", message["text"])
            text = tokenize(message_text)

            if "movie_occurrences" in self.sources:
                text, message_target, message_occurrences = self.replace_movies_in_tokenized(text)
            else:
                text = self.replace_movies_in_tokenized(text)
                message_target = text

            # if flatten messages, append message when the sender is the same as in the last message
            if flatten_messages and len(senders) > 0 and senders[-1] == role:
                dialogue[-1] += ["\n"] + text
                target[-1] += ["\n"] + message_target
                if "movie_occurrences" in self.sources:
                    for movieId in occurrences:
                        if movieId in message_occurrences:
                            occurrences[movieId][-1] += [0] + message_occurrences[movieId]
                        else:
                            occurrences[movieId][-1] += [0] * (len(text) + 1)
            # otherwise finish the previous utterance and add the new utterance
            else:
                if len(senders) > 0:
                    dialogue[-1] += ['</s>']
                    target[-1] += ['</s>', '</s>']
                    if "movie_occurrences" in self.sources:
                        for movieId in occurrences:
                            occurrences[movieId][-1] += [0]
                senders.append(role)
                dialogue.append(['<s>'] + text)
                target.append(message_target)
                if "movie_occurrences" in self.sources:
                    for movieId in occurrences:
                        if movieId in message_occurrences:
                            occurrences[movieId].append([0] + message_occurrences[movieId])
                        else:
                            occurrences[movieId].append([0] * (len(text) + 1))
        # finish the last utterance
        dialogue[-1] += ['</s>']
        target[-1] += ['</s>', '</s>']
        if "movie_occurrences" in self.sources:
            for movieId in occurrences:
                occurrences[movieId][-1] += [0]
        dialogue, target, senders, occurrences = self.truncate(dialogue, target, senders, occurrences)
        if "movie_occurrences" in self.sources:
            return dialogue, target, senders, occurrences
        return dialogue, target, senders, None





    def extract_form_data(self, data):
        """
        get form data from data. For sentiment analysis.
        :param data:
        :return: form_data. Array where each element is of the form (MovieId, Movie Name, answers, conversation_index)
        """
        form_data = []
        for (i, conversation) in enumerate(data):
            init_q = conversation["initiatorQuestions"]
            resp_q = conversation["respondentQuestions"]
            # get movies that are in both forms. Do not take empty movie names
            gen = (key for key in init_q if key in resp_q and not self.db2name[int(key)].isspace())
            for key in gen:
                answers = [init_q[key]["suggested"],
                           init_q[key]["seen"],
                           init_q[key]["liked"],
                           resp_q[key]["suggested"],
                           resp_q[key]["seen"],
                           resp_q[key]["liked"]]
                form_data.append((self.db2id[int(key)], self.db2name[int(key)], answers, i))
        return form_data

    def extract_ratings_data(self, data):
        """
        Get ratings data from each conversation.
        :param data:
        :return: array of dictionaries {movieId: rating}. One dictionary corresponds to one conversation
        """
        ratings_data = []

        total_data = 0
        missing_init_answers = 0
        missing_both_init_respond_answers = 0

        for (i, conversation) in enumerate(data):
            conv_ratings = {}

            total_data += 1

            # if there are not initiatorQuestions answers, then we use the respondentQuestions answers
            answers_dict = conversation["initiatorQuestions"]
            if len(answers_dict) == 0:
                answers_dict = conversation["respondentQuestions"]
                missing_init_answers += 1

            if len(answers_dict) == 0:
                missing_both_init_respond_answers += 1

            gen = (key for key in answers_dict if not self.db2name[int(key)].isspace())

            conversationId = int(conversation["conversationId"])

            for dbId in gen:
                movieId = self.db2id[int(dbId)]

                liked = int(answers_dict[dbId]["liked"])
                # Only consider "disliked" or "liked", ignore "did not say"
                if liked in [0, 1]:
                    conv_ratings[movieId] = process_ratings(liked)

            # Do not append empty ratings
            if conv_ratings:
                # we are appending a tuple, so that the conversation id is also perserved. We will need that later for defining the conversation category targets, based on movie ratings
                ratings_data.append( (conversationId, conv_ratings) )

        print("Rating data stats:")
        print("Out of total {:d} conversations, {:d} of them are missing only Initiator's answers, and {:d} of them are missing answers from both users.".format(total_data, missing_init_answers, missing_both_init_respond_answers))
        print("when Initiator's answers are missing, we are using the Respondent's answers.")
        # exit()

        return ratings_data





    
    def replace_movies_in_tokenized(self, tokenized):
        """
        replace movieId tokens in a single tokenized message.
        Eventually compute the movie occurrences and the target with (global) movieIds
        :param tokenized:
        :return:
        """
        output_with_id = tokenized[:]
        occurrences = {}
        pattern = re.compile(r'^\d{5,6}$')
        index = 0
        while index < len(tokenized):
            word = tokenized[index]
            # Check if word corresponds to a movieId.
            if pattern.match(word) and int(word) in self.db2id and not self.db2name[int(word)].isspace():
                # get the global Id
                movieId = self.db2id[int(word)]
                # add movie to occurrence dict
                if movieId not in occurrences:
                    occurrences[movieId] = [0] * len(tokenized)
                # remove ID
                del tokenized[index]
                # put tokenized movie name instead. len(tokenized_movie) - 1 elements are added to tokenized.
                tokenized_movie = tokenize(self.id2name[movieId])
                tokenized[index:index] = tokenized_movie

                # update output_with_id: replace word with movieId repeated as many times as there are words in the
                # movie name. Add the size-of-vocabulary offset.
                output_with_id[index:index + 1] = [movieId + len(self.word2id)] * len(tokenized_movie)

                # update occurrences
                if "movie_occurrences" in self.sources:
                    # extend the lists
                    for otherIds in occurrences:
                        # the zeros in occurrence lists can be appended at the end since all elements after index are 0
                        # occurrences[otherIds][index:index] = [0] * (len(tokenized_movie) - 1)
                        occurrences[otherIds] += [0] * (len(tokenized_movie) - 1)
                    # update list corresponding to the occurring movie
                    occurrences[movieId][index:index + len(tokenized_movie)] = [1] * len(tokenized_movie)

                # increment index
                index += len(tokenized_movie)

            else:
                # do nothing, and go to next word
                index += 1
        if "movie_occurrences" in self.sources:
            if "movieIds_in_target" in self.sources:
                return tokenized, output_with_id, occurrences
            return tokenized, tokenized, occurrences
        return tokenized

    # def truncate(self, dialogue, target, senders, movie_occurrences):
    #     # truncate conversations that have too many utterances
    #     if len(dialogue) > self.conversation_length_limit:
    #         dialogue = dialogue[:self.conversation_length_limit]
    #         target = target[:self.conversation_length_limit]
    #         senders = senders[:self.conversation_length_limit]
    #         if "movie_occurrences" in self.sources:
    #             movie_occurrences = {
    #                 key: val[:self.conversation_length_limit] for key, val in movie_occurrences.items()
    #             }
    #     # truncate utterances that are too long
    #     for (i, utterance) in enumerate(dialogue):
    #         if len(utterance) > self.utterance_length_limit:
    #             dialogue[i] = dialogue[i][:self.utterance_length_limit]
    #             target[i] = target[i][:self.utterance_length_limit]
    #             if "movie_occurrences" in self.sources:
    #                 for movieId, value in movie_occurrences.items():
    #                     value[i] = value[i][:self.utterance_length_limit]
    #     return dialogue, target, senders, movie_occurrences

    def _load_sentiment_analysis_batch(self, subset, flatten_messages=True, cut_dialogues=-1):
        batch = {"senders": [], "dialogue": [], "lengths": [], "forms": [], "movieIds": []}
        # forms: answer to movie forms (conversations_per_batch, 6)
        if "movie_occurrences" in self.sources:
            # movie occurrences (conversations_per_batch, max_conv_length, max_utt_length)
            batch["movie_occurrences"] = []

        # get batch
        batch_data = self.form_data[subset][self.batch_index[subset] * self.conversations_per_batch:
                                            (self.batch_index[subset] + 1) * self.conversations_per_batch]

        for i, example in enumerate(batch_data):
            # (movieId, movieName, answers, conversationIndex) = example
            conversation = self.conversation_data[subset][example[3]]
            if self.process_at_instanciation:
                dialogue, target, senders, movie_occurrences = conversation
            else:
                dialogue, target, senders, movie_occurrences = self.extract_dialogue(conversation,
                                                                                     flatten_messages=flatten_messages)
            if "movie_occurrences" in self.sources:
                movie_occurrences = movie_occurrences[example[0]]
                if cut_dialogues == "random":
                    cut_width = random.randint(1, len(dialogue))
                else:
                    cut_width = cut_dialogues
                # cut dialogues around occurrence
                if cut_width >= 0:
                    start_index, end_index = get_cut_indices(movie_occurrences, cut_width)
                    dialogue = dialogue[start_index:end_index]
                    senders = senders[start_index:end_index]
                    movie_occurrences = movie_occurrences[start_index:end_index]
            batch['movieIds'].append(example[0])
            batch["lengths"].append([len(message) for message in dialogue])
            batch["dialogue"].append(dialogue)
            batch["senders"].append(senders)
            batch["forms"].append(example[2])
            if "movie_occurrences" in self.sources:
                batch["movie_occurrences"].append(movie_occurrences)

        max_utterance_len = max([max(x) for x in batch["lengths"]])
        max_conv_len = max([len(conv) for conv in batch["dialogue"]])
        batch["conversation_lengths"] = np.array([len(x) for x in batch["lengths"]])
        # replace text with ids and pad sentences
        batch["lengths"] = np.array(
            [lengths + [0] * (max_conv_len - len(lengths)) for lengths in batch["lengths"]]
        )
        batch["dialogue"] = Variable(torch.LongTensor(
            self.text_to_ids(batch["dialogue"], max_utterance_len, max_conv_len)))  # (batch, conv_len, utt_len)
        batch["senders"] = Variable(torch.FloatTensor(
            [senders + [0] * (max_conv_len - len(senders)) for senders in batch["senders"]]))
        batch["forms"] = Variable(torch.LongTensor(batch["forms"]))  # (batch, 6)
        if "movie_occurrences" in self.sources:
            batch["movie_occurrences"] = Variable(torch.FloatTensor(
                [[utterance + [0] * (max_utterance_len - len(utterance)) for utterance in conv] +
                 [[0] * max_utterance_len] * (max_conv_len - len(conv)) for conv in batch["movie_occurrences"]]
            ))
        return batch



    def _load_dialogue_batch(self, subset, flatten_messages):
        batch = {"senders": [], "dialogue": [], "lengths": [], "target": []}
        if "movie_occurrences" in self.sources:
            # movie occurrences: Array of dicts
            batch["movie_occurrences"] = []

        # get batch
        batch_data = self.conversation_data[subset][self.batch_index[subset] * self.conversations_per_batch:
                                                    (self.batch_index[subset] + 1) * self.conversations_per_batch]

        for i, conversation in enumerate(batch_data):
            if self.process_at_instanciation:
                dialogue, target, senders, movie_occurrences = conversation
            else:
                dialogue, target, senders, movie_occurrences = self.extract_dialogue(conversation,
                                                                                     flatten_messages=flatten_messages)
            batch["lengths"].append([len(message) for message in dialogue])
            batch["dialogue"].append(dialogue)
            batch["senders"].append(senders)
            batch["target"].append(target)
            if "movie_occurrences" in self.sources:
                batch["movie_occurrences"].append(movie_occurrences)

        max_utterance_len = max([max(x) for x in batch["lengths"]])
        max_conv_len = max([len(conv) for conv in batch["dialogue"]])
        batch["conversation_lengths"] = np.array([len(x) for x in batch["lengths"]])
        # replace text with ids and pad sentences
        batch["lengths"] = np.array(
            [lengths + [0] * (max_conv_len - len(lengths)) for lengths in batch["lengths"]]
        )
        batch["dialogue"] = Variable(torch.LongTensor(
            self.text_to_ids(batch["dialogue"], max_utterance_len, max_conv_len)))
        batch["target"] = Variable(torch.LongTensor(
            self.text_to_ids(batch["target"], max_utterance_len, max_conv_len)))
        batch["senders"] = Variable(torch.FloatTensor(
            [senders + [0] * (max_conv_len - len(senders)) for senders in batch["senders"]]))
        if "movie_occurrences" in self.sources:
            batch["movie_occurrences"] = [
                {key: [utterance + [0] * (max_utterance_len - len(utterance)) for utterance in value] +
                      [[0] * max_utterance_len] * (max_conv_len - len(value)) for key, value in conv.items()}
                for conv in batch["movie_occurrences"]
            ]
        return batch


    def _load_ratings_batch(self, subset, batch_input, max_num_inputs=None):
        if batch_input == 'random_noise' and max_num_inputs is None:
            raise ValueError("batch_input set to random_noise, max_num_inputs should not be None")
        # One element in batch_data corresponds to one conversation
        batch_data = self.ratings_data[subset][self.batch_index[subset] * self.conversations_per_batch:
                                               (self.batch_index[subset] + 1) * self.conversations_per_batch]
        # WARNING : loading differs from the one used in autorec_batch_loader
        if batch_input == "full":
            if subset == "train":
                target = np.zeros((self.conversations_per_batch, self.n_movies)) - 1
                input = np.zeros((self.conversations_per_batch, self.n_movies))
                for (i, ratings) in enumerate(batch_data):
                    for (movieId, rating) in ratings.items():
                        input[i, movieId] = rating
                        target[i, movieId] = rating
            # if not training, for all ratings of conversation,
            # use this rating as target, and use all other ratings as inputs
            # so batch has as many examples as there are movie mentions in the `conversations_per_batch` conversations
            else:
                input = []
                target = []
                for ratings in batch_data:
                    complete_input = [0] * self.n_movies
                    # populate input with ratings
                    for movieId, rating in ratings.items():
                        complete_input[movieId] = rating
                    for movieId, rating in ratings.items():
                        # for each movie, zero out in the input and put target rating
                        input_tmp = complete_input[:]
                        input_tmp[movieId] = 0
                        target_tmp = [-1] * self.n_movies
                        target_tmp[movieId] = rating
                        input.append(input_tmp)
                        target.append(target_tmp)
                input = np.array(input)
                target = np.array(target)
            input = Variable(torch.from_numpy(input).float())
            target = Variable(torch.from_numpy(target).float())
            return {"input": input, "target": target}
        # take random inputs
        elif batch_input == "random_noise":
            input = np.zeros((self.conversations_per_batch, self.n_movies))
            target = np.zeros((self.conversations_per_batch, self.n_movies)) - 1
            for (i, ratings) in enumerate(batch_data):
                # randomly chose a number of inputs to keep
                max_nb_inputs = min(max_num_inputs, len(ratings) - 1)
                n_inputs = random.randint(1, max(1, max_nb_inputs))
                # randomly chose the movies that will be in the input
                input_keys = random.sample(ratings.keys(), n_inputs)
                # Create input
                for (movieId, rating) in ratings.items():
                    if movieId in input_keys:
                        input[i, movieId] = rating
                # Create target
                for (movieId, rating) in ratings.items():
                    target[i, movieId] = rating
            return {"input": Variable(torch.from_numpy(input).float()),
                    "target": Variable(torch.from_numpy(target).float())}





    # def load_batch(self, subset="train",
    #                flatten_messages=True, batch_input="random_noise", cut_dialogues=-1, max_num_inputs=None):
    #     """
    #     Get next batch
    #     :param batch_input:
    #     :param cut_dialogues:
    #     :param max_num_inputs:
    #     :param task : one of : "predict_movie_given_sentiment", "predict_sentiment_given_movie", "predict_movie_given_liked"
    #     :param subset: "train", "valid" or "test"
    #     :param flatten_messages: if False, load the conversation messages as they are. If True, concatenate consecutive
    #     messages from the same sender and put a "\n" between consecutive messages.
    #     :return: batch
    #     """
    #     if "dialogue_to_categories" in self.sources:


    #     elif "dialogue" in self.sources:
    #         if self.word2id is None:
    #             raise ValueError("word2id is not set, cannot load batch")
    #         batch = self._load_dialogue_batch(subset, flatten_messages)
    #     elif "sentiment_analysis" in self.sources:
    #         if self.word2id is None:
    #             raise ValueError("word2id is not set, cannot load batch")
    #         batch = self._load_sentiment_analysis_batch(subset, flatten_messages, cut_dialogues=cut_dialogues)
    #     elif "ratings" in self.sources:
    #         batch = self._load_ratings_batch(subset, batch_input=batch_input, max_num_inputs=max_num_inputs)

    #     self.batch_index[subset] = (self.batch_index[subset] + 1) % self.n_batches[subset]

    #     return batch
    

    # def text_to_ids(self, dialogue, max_utterance_len, max_conv_len):
    #     """
    #     replace with corresponding ids.
    #     Pad each utterance to max_utterance_len. And pad each conversation to max_conv_length
    #     :param dialogue: [[[word1, word2, ...]]]
    #     :param max_utterance_len:
    #     :param max_conv_len:
    #     :return: padded dialogue
    #     """
    #     dialogue = [[[self.token2id(w) for w in utterance] +
    #                  [self.word2id['<pad>']] * (max_utterance_len - len(utterance)) for utterance in conv] +
    #                 [[self.word2id['<pad>']] * max_utterance_len] * (max_conv_len - len(conv)) for conv in dialogue]
    #     return dialogue








# ------------------------------------------------------------------------------------------------------------------------------------------------

    # def preprocess_for_categories(self):

    #     # self.categories = self.categories

    #     self.number_of_movies = self.n_movies

    #     self.conversation_id_to_categories_target = {}


    #     self.uniform_category_vector = np.ones(len(self.categories)) / len(self.categories)


    #     # the dataset has a training, validation and test sets, corresponding to the appropriate sets of the Redial dataset.
    #     # the dataset will have user ratings for input and category distribution as targets.
    #     # the form of the input will be a list of ratings in the form [ (redial_movie_id, binary_user_rating_token), ...]
    #     # the target will be a the category vector.
    #     # different inputs depending on training or validation/test sets.
    #     # training set input : all ratings (during training a sampled subset of all the ratings will be used for each epoch)
    #     # validation/test set input: all-but one ratings (therefore, for one user, we will have N validation samples, where N is the number of ratings)

    #     # subset

    #     data = {"train": [[], []], "valid": [[], []], "test": [[], []]}

    #     for subset in ["train", "valid", "test"]:

    #         for i in range(len(self.ratings_token_ids_lists_data[subset])):

    #             conversation_id, sample = self.ratings_token_ids_lists_data[subset][i]
    #             # initialize the user-category vector with zeros
    #             user_category_vector = np.zeros(len(self.categories))

    #             new_sample = []

    #             for (redial_movie_id, sentiment_token) in sample:

    #                 sentiment = 1 if sentiment_token == self.rating_targets["liked"] else -1
    #                 movie_category_vector = self.redial_id_to_categories[redial_movie_id]

    #                 rating = 1 if sentiment == 1 else 0

    #                 new_sample.append((redial_movie_id, rating))

    #                 movie_category_vector = movie_category_vector * sentiment
    #                 user_category_vector += movie_category_vector

    #             # apply softmax on user category vectors
    #             user_category_vector = np.exp(user_category_vector)/sum(np.exp(user_category_vector))

    #             self.conversation_id_to_categories_target[conversation_id] = user_category_vector


    #             input = [new_sample]
    #             target = [user_category_vector]

    #             data[subset][0] += input
    #             data[subset][1] += target

    #     self.data_samples = data
    #     self.number_of_samples = {}
    #     for subset in data:
    #         self.number_of_samples[subset] = len(data[subset][0])


    # def get_batched_samples(self, subset, conversations_per_batch):   
    # # def get_dataset_for_FFNN_ReDial(self):
    #     # this function will preprocess the dataset to its final form  and split it into batches
    #     # the final form of the dataset will consist of input vectors with cardinality equal to the number of movies,
    #     # and output vectors with cardinality equal to the number of categories

    #     input_size = self.number_of_movies

    #     indexes = np.arange(self.number_of_samples[subset])
    #     # making a list for randomizing index for training set, works like suffling the samples set
    #     if subset == "train":
    #         np.random.shuffle(indexes)

    #     # first list corresponds to input vectors, the second list corresponds to rating targets and third list corresponds to category targets
    #     sample_vectors = [[], [], []]
    #     for i in indexes:


    #         rating_sample = self.data_samples[subset][0][i]
    #         category_target = self.data_samples[subset][1][i]

    #         input_vector = np.zeros(input_size)

    #         if subset == "train":

    #             rating_target = - np.ones(input_size)

    #             # get a random subset of the ratings for input 
    #             random_index = np.random.randint(2, size=len(rating_sample)) > 0


    #             # copy given ratings on input vector
    #             for i, (movie_id, rating) in enumerate(rating_sample):
    #                 rating_target[movie_id] = rating

    #                 if random_index[i]:
    #                     input_vector[movie_id] = rating

    #             sample_vectors[0].append(torch.from_numpy(input_vector).float()) 
    #             sample_vectors[1].append(torch.from_numpy(rating_target).float()) 
    #             sample_vectors[2].append(torch.from_numpy(category_target).float()) 


    #         else:
    #             # Maintain the one target, given rest ratings, as the ReDial Authors did

    #             # copy given ratings on input vector
    #             for (movie_id, rating) in rating_sample:
    #                 input_vector[movie_id] = rating
                
    #             # for, every rating, hide that rating from input and create one sample
    #             for i, (movie_id, rating) in enumerate(rating_sample):
    #                 # copy input vector
    #                 temp_input = np.copy(input_vector)
    #                 # hide one rating
    #                 temp_input[movie_id] = 0
    #                 # initialize target vector
    #                 rating_target = - np.ones(input_size)
    #                 # set the only this rating as target for this sample 
    #                 rating_target[movie_id] = rating

    #                 # append this sample
    #                 sample_vectors[0].append(torch.from_numpy(temp_input).float())
    #                 sample_vectors[1].append(torch.from_numpy(rating_target).float())
    #                 sample_vectors[2].append(torch.from_numpy(category_target).float())


    #     # for training subset, we shuffle the samples
    #     if subset == "train":
    #         sample_vectors = list(zip(sample_vectors[0], sample_vectors[1], sample_vectors[2]))
    #         random.shuffle(sample_vectors)
    #         sample_vectors[0], sample_vectors[1], sample_vectors[2] = zip(*sample_vectors)

    #     # split the samples into batches

    #     # calculate number of batches
    #     number_of_batches = math.ceil(len(sample_vectors[0])/conversations_per_batch)

    #     batch_data = []


    #     for i in range(number_of_batches):
    #         input_vector = Variable(torch.stack(sample_vectors[0][i * conversations_per_batch: (i + 1) * conversations_per_batch]), requires_grad = True)
    #         rating_target = Variable(torch.stack(sample_vectors[1][i * conversations_per_batch: (i + 1) * conversations_per_batch]), requires_grad = True)
    #         category_vector = Variable(torch.stack(sample_vectors[2][i * conversations_per_batch: (i + 1) * conversations_per_batch]), requires_grad = True)

    #         batch_data.append( (input_vector, rating_target, category_vector))

    #     return batch_data, number_of_batches



    # def preprocess_ratings_into_tokenized_lists(self, smooth_targets = False):
    #     self.ratings_token_ids_lists_data = { "train" : [], "valid" : [], "test" : [] }

    #     frequencies_of_rating_lengths = { "train" : defaultdict(int), "valid" : defaultdict(int), "test" : defaultdict(int) } 



    #     # define the values of the ratings thas should be used as targets
    #     if smooth_targets:
    #     # these might work better, because sigmoid is assymptotic to lines y=0 and y=1
    #         self.rating_targets = {"liked": 0.9, "disliked" :0.1}
    #     else:
    #         self.rating_targets = {"liked": 1, "disliked" :0}



    #     # remove samples with less than 2 ratings
    #     removed_small_samples_counter = 0
    #     # for every data set
    #     for subset in self.ratings_token_ids_lists_data:
    #         # for every sample
    #         for conversation_id, sample in self.ratings_data[subset]:
    #             # increase frequency depending the subset and the length of the ratings
    #             frequencies_of_rating_lengths[subset][len(sample)] += 1
    #             if len(sample) > 2:
    #                 self.ratings_token_ids_lists_data[subset].append((conversation_id, [(movie_id, self.rating_targets["liked"] if sample[movie_id] > 0.5 else self.rating_targets["disliked"]) for movie_id in sample]))
    #             else:
    #                 removed_small_samples_counter += 1



    # def _load_dialogue_to_categories_batch(self, batch_data):

    #     # at the end, there should be one sample per recommender's response.
    #     # This intuitively makes sense, due to the reason that the category prediction needs to be calculated and used when the recommender is about to respond

    #     # print("[CLS] :", self.token2id("[CLS]"))
    #     # print("[SEP] :", self.token2id("[SEP]"))


    #     batch_contexts = []
    #     batch_token_type_ids = []
    #     batch_category_targets = []


    #     for conversation in batch_data:
    #         # retrieve conversation data
    #         if self.process_at_instanciation:
    #             dialogue, senders, movie_mentions, category_target, answers_dict = conversation
    #         else:
    #             dialogue, senders, movie_mentions, category_target, answers_dict = self.extract_dialogue4Bert(conversation)

    #         for i in range(len(senders)):
    #             # print(dialogue[i])
    #             # print(senders[i])
    #             # print(category_target[i])
    #             # if this message was sent by the recommender
    #             if senders[i] == -1:
    #                 context = []
    #                 token_type_ids = []

    #                 # if there is no context, then we do not create a sample
    #                 if i == 0:
    #                     continue

    #                 # for every message proceeding this message
    #                 for j in reversed(range(i)):
    #                     # we do not want the context to exceed the input lenght limit
    #                     if len(context) + 1 + len(dialogue[j]) >= self.max_input_length:
    #                         break
    #                     # we reversively add previous messages as context, separated by the "SEP" special token
    #                     context = ["[SEP]"] + dialogue[j] + context

    #                     token_type_id = 0 if senders[j] == -1 else 1
    #                     token_type_ids = [token_type_id] * ( len(dialogue[j]) + 1 ) + token_type_ids

    #                 # we replace the first token of the context (which is "SEP"), with the "CLS" token.
    #                 context[0] = "[CLS]"
    #                 # we always se the CLS token to have token id equal to 0, so that it is homeomorphic
    #                 token_type_ids[0] = 0


    #                 # if there is no context, then we do not create a sample
    #                 if len(context) == 0:
    #                     continue

    #                 # translate tokens to token_ids
    #                 context = [self.token2id(token) for token in context]

    #                 # print(" -- i:",i,", context = ", context)

    #                 batch_contexts.append(context)
    #                 batch_token_type_ids.append(token_type_ids)
    #                 batch_category_targets.append(category_target)


    #         # for i in range(len(batch_contexts)):

    #         #     print(batch_contexts[i], len(batch_contexts[i]))
    #         #     print(batch_token_type_ids[i], len(batch_token_type_ids[i]))
    #         #     print(batch_category_targets[i])
    #         #     print()
    #         # exit()


    #     max_length = np.max( [len(context) for context in batch_contexts ])

    #     num_of_samples = len(batch_contexts)

    #     # allocating tensors for saving the samples into containers of equal size
    #     contexts = np.full((num_of_samples, max_length), fill_value = 0, dtype=np.int64)

    #     token_types = np.full((num_of_samples, max_length), fill_value=0, dtype=np.int64)

    #     attention_masks = np.full((num_of_samples, max_length), fill_value = 0, dtype=np.bool_)

    #     targets = np.zeros((num_of_samples, len(self.categories)), dtype=np.float32)

    #     last_context_token_mask = np.full((num_of_samples, max_length), fill_value = 0, dtype=np.bool_)

    #     for i in range(num_of_samples):
    #         # fill in the values in the containers
    #         contexts[i, : len(batch_contexts[i])] = batch_contexts[i]
    #         token_types[i, : len(batch_token_type_ids[i])] = batch_token_type_ids[i]
    #         attention_masks[i, : len(batch_contexts[i])] = True
    #         targets[i] = batch_category_targets[i]
    #         last_context_token_mask[i, len(batch_contexts[i]) - 1 ] = True


    #     contexts = torch.tensor(contexts)
    #     token_types = torch.tensor(token_types)
    #     attention_masks = torch.tensor(attention_masks)
    #     targets = torch.tensor(targets)
    #     last_context_token_mask = torch.tensor(last_context_token_mask)

    #     return contexts, token_types, attention_masks, targets, last_context_token_mask

