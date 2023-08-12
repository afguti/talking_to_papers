#!/usr/bin/env python3
import os
import openai
import sys
import arxiv
#import chromadb

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

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

#Function to answer questions
def query(question: str) -> str:
    q_counter = 0
    key = "no"
    while key == "no":
        q_counter += 1 
        question = question
        result = qa_chain({"query": question})
        ans = result["result"]
        print(f"Output[{q_counter}]:\n{ans}\n")
        key = input("Is the result ok? [yes/(no)]: ")
        if key == "":
            key = "no"
    return ans

#Document loading
from langchain.document_loaders import PyPDFLoader
location = input("PDF location: ")
loader = PyPDFLoader(location)
pages = loader.load()
print(f"Document loaded. There is a total of {len(pages)} pages.")

#Documents splitting
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
docs = text_splitter.split_documents(pages)
print(f"Documents splitted into {len(docs)} parts.")

#Vectorizyng and embedding
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding
)
print(f"vectordb created with a collection count of {vectordb._collection.count()}")

#Question answering
llm_name = "gpt-3.5-turbo-0613"
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever()
)

###Gueting the main topics to drive the conversation
#ans = query("List 10 topics from the paper for an engaging and entertaining conversation between the autors of the paper and a regular person with no background in science who wants to understand the implications of the content of this paper for an average person's life. The questions have to be in an order that can help to understand the content of the paper. The questions have to be relevant to the content of the paper. The questions have to be put in a python list.")
search = arxiv.Search(id_list=["2308.02624"])
paper = next(search.results())
#print(paper.summary,"\n\n")
text = f'List 10 topics from the TEXT enclosed below for an engaging conversation between the autors of the paper and a regular person with no background in science who wants to understand the implications of the content of the TEXT for an average person. The questions have to be in an order that can help to understand the TEXT. The questions have to be relevant to the content of the TEXT. The questions have to be put in a python list.\nTEXT:\n{paper.summary}'
messages =  [  
{'role':'system', 
 'content':"""You are a popular science communicator"""},    
{'role':'user', 'content':text},  
]
ans, token_dict = get_completion_and_token_count(messages, temperature=0)
lines = ans.strip().split("\n")
qq = [line.split(". ", 1)[1] for line in lines] #These are the topics that will be used for the conversation
print(ans,"\n")

###The introduction of the episode
#intro = query('Based on the content, print out an engaging and entertaining introduction for an episode of a podcast named "Talking to papers". MENTION that all the content is created by Artificial Intelligence. The introduction has to mention that we will discuss about a particular paper. Provide the main topic and a summary of the paper in simple terms for popular science communication. The name of the host is "Ethan".')
text = f'Based on the TEXT enclosed below, print out an engaging introduction for an episode of a podcast named "Talking to papers". MENTION that all the content is created by Artificial Intelligence. The introduction has to mention that we will discuss about a particular paper titled {paper.title}. Provide the main topic and a summary of the paper in simple terms for popular science communication. Mention that the link to the paper is in the description of the episode. The name of the host is "Ethan".\nTEXT:\n{paper.summary}'
messages =  [  
{'role':'system', 
 'content':"""You are a popular science communicator"""},    
{'role':'user', 'content':text},  
]
ans, token_dict = get_completion_and_token_count(messages, temperature=0)
print(f"whe have the intro!!\n{ans}\n")

###The start of the conversation
print(f"Authors are: {paper.authors}")
author = input("Author: ")
firsq = query(f'Based on the content, print out the beginning of an engaign conversation between a podcast host and a guest named {author} about {qq[0]}. The conversation has to be easy to understand for popular science communication. This part is the ontinuation after an introduction. AVOID LISTING NAMES.')
print("we quicked off the conversation!!")

###The conversation
for i in range(len(qq)-1):
    print(f"topic {i+1}: {qq[i+1]}\n")
    conv = query(f'Based on the content, print out a part of an engaign conversation between a podcast host and a guest about {qq[i+1]}. The conversation has to be easy to understand for popular science communication. The conversation is just part of a more broader conversation. AVOID LISTING NAMES.')
    print(f"We got topic {i+1}")
print("We have the conversation!!")

###For testing
#q_counter = 0
#while True:
#    q_counter += 1
#    question = input(f"Question[{q_counter}]: ")
#    result = qa_chain({"query": question})
#    ans = result["result"]
#    print(f"Answer[{q_counter}]:\n{ans}\n")
