#!/usr/bin/env python3
import os
import openai
import sys
import arxiv
import subprocess

###External
from pydub import AudioSegment, silence #for mp3 manipulation. pip install pydub

###My scrips
from filehandle import remove_text, save_output
from audiogen import audio as gen_audio
from dialog import separate
from shortaudio import voice, combine

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

#Font colors
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prGreenIn(skk): input("\033[92m {}\033[00m" .format(skk))

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

#Function to answer questions based on the PDF
def query(question: str) -> str:
    q_counter = 0
    key = "no"
    while key == "no":
        q_counter += 1 
        question = question
        result = qa_chain({"query": question})
        ans = result["result"]
        prGreen(f"Output[{q_counter}]:")
        print(ans,"\n")
        prRed("Are we good to continue?")
        key = input("Is the result ok? [yes/(no)]: ")
        if key == "":
            key = "no"
    return ans

#Document loading
from langchain.document_loaders import PyPDFLoader
paper_id = sys.argv[1]
if len(sys.argv) < 2:
    location = input("PDF location: ")
else:
    location = f"~/Documents/pets/{paper_id}.pdf"
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

###Gueting the main topics to drive the conversation WHE NEED TO SAVE THIS IN A FILE AND IF THE FILE EXIST SKIP THIS STEP
search = arxiv.Search(id_list=[paper_id])
paper = next(search.results())
if not(os.path.exists('./output/topics.txt')):
    prGreen("\nTopics to drive the conversation:")
    text = f'List 10 topics from the TEXT enclosed below for an engaging conversation between the autors of the paper and a regular person with no background in science who wants to understand the implications of the content of the TEXT for an average person. The questions have to be in an order that can help to understand the TEXT. The questions have to be relevant to the content of the TEXT. The questions have to be put in a python list.\nTEXT:\n{paper.summary}'
    messages =  [  
    {'role':'system', 
     'content':"""You are a popular science communicator"""},    
    {'role':'user', 'content':text},  
    ]
    ans, token_dict = get_completion_and_token_count(messages, temperature=0)
    save_output("topics.txt",ans)
    lines = ans.strip().split("\n")
    qq = [line.split(". ", 1)[1] for line in lines] #These are the topics that will be used for the conversation
    print(ans)
else:
    prGreen("Topics were chose already!")
    with open("./output/topics.txt", "r") as file:
        ans = file.read()
    lines = ans.strip().split("\n")
    qq = [line.split(". ", 1)[1] for line in lines] #These are the topics that will be used for the conversation
    print(ans)   

###The introduction of the episode
if not(os.path.exists('./output/final_p1.mp3')):
    prGreen("\nThe introduction:")
    text = f'Based on the TEXT below, print out an engaging introduction for an episode of a podcast named "Talking to papers". The introduction MUST BE MENTION that the content is created by Artificial Intelligence. The introduction has to mention that we will discuss about a particular paper titled {paper.title}. Provide the main topic and a summary of the paper in simple terms for popular science communication. Mention that the link to the paper is in the description of the episode. The name of the host is "Ethan".\nTEXT:\n{paper.summary}'
    messages =  [  
    {'role':'system', 
     'content':"""You are a popular science communicator"""},    
    {'role':'user', 'content':text},  
    ]
    ans, token_dict = get_completion_and_token_count(messages, temperature=0)
    print(f"{ans}")
    save_output("main.txt", f'{ans}\n\n')
    save_output("intro.txt", f'{ans}\n')
    subprocess.run(["open", "./output/intro.txt"], check=True) #Open file for revision in default OS text editor
    prGreenIn("\nNOW REVIEW intro.txt AND PRESS ENTER TO CONTINUE")
    gen_audio("./output/intro.txt", 'Stephen', 1)
else:
	print("\nPart 1, Introduction is already completed.")	

###The start of the conversation
if not(os.path.exists('./output/conversation_0.mp3')):
    prGreen("\nQuicking off the conversation:")
    print(f"Authors are: {paper.authors}")
    author = input("Author: ")
    firsq = query(f'Based on the content, print out the beginning of an engaign conversation between a podcast host and a guest named {author} about {qq[0]}. The conversation has to be easy to understand for popular science communication. This part is the ontinuation after an introduction. AVOID LISTING NAMES.')
    save_output("main.txt", f'{firsq}\n\n')
    save_output("conversation_0.txt", f'{firsq}')
    subprocess.run(["open", "./output/conversation_0.txt"], check=True)
    prGreenIn("\nNOW REVIEW conversation_0.txt AND PRESS ENTER TO CONTINUE")
    with open('./output/conversation_0.txt', 'r') as file:
    	firsq = file.read()	
    hostd, guestd = separate(firsq,"Podcast Host",author)
    combined = combine(hostd, guestd)
    combined.export(f"./output/conversation_0.mp3", format="mp3")
else:
	print("\nPart2, Start of the conversation is already completed.")

###The conversation
prGreen("\nThe conversation:")
for i in range(len(qq)-1):
    if not(os.path.exists(f'./output/conversation_{i+1}.mp3')):
        prGreen(f"topic {i+1}: {qq[i+1]}\n")
        conv = query(f'Based on the content, print out a part of an engaign conversation between a podcast host and a guest about {qq[i+1]}. The conversation has to be easy to understand for popular science communication. The conversation is just part of a more broader conversation. AVOID LISTING NAMES.')
        prGreen(f"We got topic {i+1}")
        save_output("main.txt", f'{conv}\n\n')

        save_output(f"conversation_{i+1}.txt", f'{conv}')
        subprocess.run(["open", f"./output/conversation_{i+1}.txt"], check=True)
        prGreenIn(f"\nNOW REVIEW conversation_{i+1}.txt AND PRESS ENTER TO CONTINUE")
        with open(f'./output/conversation_{i+1}.txt', 'r') as file:
        	conv = file.read()
        hostd, guestd = separate(conv,"Podcast Host","Guest")
        combined = combine(hostd, guestd)
        combined.export(f"./output/conversation_{i+1}.mp3", format="mp3")
    else:
	    print(f"\nconversation_{i+1}.mp3 is already completed.")

prGreen("We have the conversation!!")

###Combining the conversations from 0 to 9

###The closure
prGreen("\nThe ending:")
text = f'Based on the TEXT enclosed below, print out an engaging outro for an episode of a podcast named "Talking to papers". Start by extending an appreciation to the authors of the paper. Invite listeners to suggest papers for the next episode. \nTEXT:\n{paper.summary}'
messages =  [  
{'role':'system', 
 'content':"""You are a popular science communicator"""},    
{'role':'user', 'content':text},  
]
ans, token_dict = get_completion_and_token_count(messages, temperature=0)
prGreen("The outro:")
print(f"{ans}\n")
save_output("main.txt", ans)

