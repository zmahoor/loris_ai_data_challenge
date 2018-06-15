
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import re
from textblob import TextBlob

input_directory = '/Users/zahra/git/loris_ai_data_challenge/data/'

topic_dict = {'1': 'Ordinary Life', '2': 'School Life', '3': 'Culture_Education',
              '4': 'Attitude_Emotion', '5': 'Relationship', '6': 'Tourism' , '7': 'Health', 
              '8': 'Work', '9': 'Politics', '10': 'Finance'}

action_dict = {'1': 'inform', '2': 'question', '3': 'directive', '4': 'commissive'}


def read_glove_vecs(glove_file):
    
    with open(glove_file, 'r',encoding='UTF-8') as f:
        words = set()
        word_to_vec_map = {}
        
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
            
    return words_to_index, index_to_words, word_to_vec_map


def convert_topic(topic):
    return topic_dict[topic]


def convert_action(act):
    return action_dict[act]


def load_conversations(category='train'):
    
    conversations_text = 'dialogues_' + category + '.txt'
    conversations_emotion = 'dialogues_emotion_' + category + '.txt'
    conversations_action = 'dialogues_act_' + category + '.txt'
        
    dial_dir = os.path.join(input_directory+category, conversations_text)
    emo_dir = os.path.join(input_directory+category, conversations_emotion)
    act_dir = os.path.join(input_directory+category, conversations_action)
    
    # Open files
    in_dial = open(dial_dir, 'r')
    in_emo = open(emo_dir, 'r')
    in_act = open(act_dir, 'r')
    
    # build a list of dictionaries: a dictionary per dialogue
    conversations_list = [
        {
            'utterances': utterances,
            'emotions': emotions,
            'actions': actions
        }
        for utterances, emotions, actions in (
            (dialogue.split('__eou__')[:-1], 
             emotion.split(), 
             action.split())
            for dialogue, emotion, action in zip(in_dial, in_emo, in_act)
        )
        if len(utterances) == len(emotions) == len(actions)
    ]
            
    return conversations_list


def find_sentiments(conversations_list):
    
    for conversation_dict in conversations_list:
        
        conversation_text = conversation_dict['utterances']
        conversation_emotion_blob = []

        for sentence in conversation_text:
            blob = TextBlob(sentence)
            conversation_emotion_blob.append(blob.sentiment.polarity)
        conversation_dict['blob_emotions'] = conversation_emotion_blob
    

def create_samples(conversations_list):
    
    samples_list = []
    
    for conversaton_dict in conversations_list:
                    
        for index, utterance in enumerate(conversaton_dict['utterances']):
            if index == 0: 
                continue
                
            change_in_emotion = conversaton_dict['blob_emotions'][index] - conversaton_dict['blob_emotions'][index-1]
            samples_list.append({'utterance': utterance, 
                                 'prev_emotion': conversaton_dict['blob_emotions'][index-1], 
                                 'current_emotion': conversaton_dict['blob_emotions'][index]
                                 })          
    return samples_list
    

