import sys
import openai
import tiktoken

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_completion_and_token_count(messages, #Here I can count the number of tokens
                                   model="gpt-3.5-turbo-16k-0613", 
                                   temperature=0, 
                                   max_tokens=4096):
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=max_tokens,
    )
    
    content = response.choices[0].message["content"]
    
    token_dict = {
'prompt_tokens':response['usage']['prompt_tokens'],
'completion_tokens':response['usage']['completion_tokens'],
'total_tokens':response['usage']['total_tokens'],
    }

    return content, token_dict

def prompt(entra: str):
    text = f"""Step 1 - Sentiment Analysis: Read the text and determine the overall sentiment expressed. Is the text predominantly positive, negative, or neutral? Look for keywords and emotional cues that indicate the general sentiment.

Step 2 - Identify Emotions: Within the text, identify specific emotions that are expressed or implied. Look for words or phrases that convey joy, sadness, excitement, relaxation, inspiration, etc.

Step 3 - Analyze Tone and Context: Consider the tone and context of the text. Is it uplifting and motivational, reflective and contemplative, dramatic and intense, or soothing and peaceful? Take note of the overall mood portrayed.

Step 4 - Connect with Music Genres: Based on the sentiment, emotions, and mood identified, think about music genres that align well with those elements. For example, upbeat and positive sentiments may match well with pop or indie music, while reflective and contemplative emotions may suit acoustic or instrumental pieces.

Step 5 - Describe the Song Genre and Mood: After analyzing the text and making connections with music genres, write a sentence that describes the type of genre and mood that would best complement the content. For instance, "A soothing acoustic ballad that captures feelings of nostalgia and introspection," or "An energetic and upbeat pop track conveying a message of empowerment and joy.

For the text presented next, execute the five steps described earlier to come up with sentence describing the type of music genre and mood that would best suit the content.

text: {entra}"""
    messages = [  
    {'role':'system', 
    'content':"""You are a musician who is an expert in incorporating audio into podcasts."""},    
    {'role':'user', 'content':text},  
    ] 
    response, token_dict = get_completion_and_token_count(messages, temperature=0)
    
    return response