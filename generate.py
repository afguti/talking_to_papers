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

#for GPT-4
def get_completion_gpt4(messages, #Here I can count the number of tokens
                                   model="gpt-4-0613", 
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
def query(question: str, auto = 0) -> str:
    q_counter = 0
    key = "no"
    while key == "no":
        q_counter += 1 
        question = question
        result = qa_chain({"query": question})
        ans = result["result"]
        prGreen(f"Output[{q_counter}]:")
        if auto == 0:
            print(ans,"\n")
            prRed("Are we good to continue?")
            key = input("Is the result ok? [yes/(no)]: ")
            if key == "":
                key = "no"
        else:
            key = "yes"
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

###Gueting the main topics to drive the conversation
search = arxiv.Search(id_list=[paper_id])
paper = next(search.results())
if not(os.path.exists('./output/topics.txt')):
    prGreen("\nTopics to drive the conversation:")
    text = f'Generate a list of 10 engaging topics based on the provided TEXT that could facilitate a conversation between \
    the author of the TEXT and an individual without a science background. This person is seeking to comprehend the implications \
    of the TEXT\'s content on an average person\'s perspective. Organize the topics in a coherent sequence that aids in the \
    reader\'s understanding of the TEXT. Ensure the topics do not overlap to prevent redundancy. \
    Formulate topics that are directly related to the content of the TEXT. Print out only the list of topics. \
    Here is the TEXT for reference:\n{paper.summary}'
    messages =  [  
    {'role':'system', 
     'content':"""You are a popular science communicatora and content creator"""},    
    {'role':'user', 'content':text},  
    ]
    #ans, token_dict = get_completion_and_token_count(messages, temperature=0)
    ans, token_dict = get_completion_gpt4(messages, temperature=0)
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

###Gueting the description
search = arxiv.Search(id_list=[paper_id])
paper = next(search.results())
if not(os.path.exists('./output/description.txt')):
    prGreen("\nGenerating the description:")
    text = f'Generate a description of about 100 words based on the provided TEXT below. \
    The description is for a podcast episode in which there will be a discussion about TEXT. \
    The description has to be engaging and trigger the curiosity of the reader.\
    The description has to include at the bottom: LINK TO THE PAPER:\nhttps://arxiv.org/pdf/{paper_id}.pdf \
    Write a short but provoking title for the podcast below the LINK, present it as TITLE: \
    Here is the TEXT for reference:\n{paper.summary}'
    messages =  [  
    {'role':'system', 
     'content':"""You a professional writer and a content creator."""},    
    {'role':'user', 'content':text},  
    ]
    ans, token_dict = get_completion_and_token_count(messages, temperature=0)
    save_output("description.txt",ans)
    subprocess.run(["open", "./output/description.txt"], check=True) #Open file for revision in default OS text editor
    prGreenIn("\nNOW REVIEW description.txt AND PRESS ENTER TO CONTINUE")
else:
    prGreen("Description was printed already!")

###The introduction of the episode
if not(os.path.exists('./output/final_p1.mp3')):
    prGreen("\nThe introduction:")
    text = f'Based on the TEXT below, print out an engaging introduction for an episode of a podcast named "The PaperCast". \
    The introduction MUST BE MENTION that the content is created by Artificial Intelligence. The introduction has to mention \
    that we will discuss about a particular paper titled {paper.title}. Provide the main topic and a summary of the paper \
    in simple terms for popular science communication. Mention that the link to the paper is in the description of the episode. \
    The name of the host is "Santiago".\nTEXT:\n{paper.summary}'
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

###The conversation
if not(os.path.exists('./output/Conversation.mp3')):
    prGreen("\nThe conversation:")
    if not(os.path.exists(f'./output/Conversation.txt')):
        prGreen("\nQuicking off the conversation:")
        print(f"Authors are: {paper.authors}")
        author = input("Author: ")
        firsq = query(f'Based on the content, print out the beginning of an engaign conversation between a podcast \
        host and a guest about {qq[0]}. Introduce the guest named {author} by its name and mention that they is one \
        of the authors of the paper we will talk about. The conversation has to be easy to understand for popular science communication. \
        You will use "Host: " and "Guest: " at the beginning of each line of the conversation. \
        Start the conversation with "Today, we have an special guest" \
        This part is the continuation after an introduction. Do not end the conversation at the end. AVOID LISTING NAMES.',1)
        save_output(f"Conversation.txt", f'{firsq}\n\n   ####   DIVISION   ####\n\n')
        for i in range(len(qq) - 1): 
            prGreen(f"topic {i + 1}: {qq[i + 1]}")
            conv = query(f'Based on the content, print out a segment of an engaign conversation, between \
            a podcast host and a guest about {qq[i + 1]}. The conversation has to be easy to understand for popular science communication. \
            The host will not introduce or welcome the guest, rather the host will ask a question right from the beginning. \
            End the conversation about this tipic so that so that the host can continue immediately the next topic of the conversation. \
            AVOID LISTING NAMES.',1)
            prGreen(f"We got topic {i + 1}")
            save_output("main.txt", f'{conv}\n\n')
            save_output(f"Conversation.txt", f'{conv}\n\n   ####   DIVISION   ####\n\n')
        subprocess.run(["open", f"./output/Conversation.txt"], check=True)
        prGreenIn(f"\nNOW REVIEW Conversation.txt AND PRESS ENTER TO CONTINUE")
    else:     
    	print(f"\nConversation.txt is already completed.")
    if not(os.path.exists(f'./output/Conversation_GPT4.txt')):
        with open(f'./output/Conversation.txt', 'r') as file:
        	conv = file.read()
        text = f'The provided TEXT represents a conversation. Please conduct a thorough review to identify and remove redundancies. \
        Additionally, carefully check for any grammar and spelling errors, addressing them as necessary. \
        Omit auxiliary phrases such as "Absolutely," "Fascinating," and similar expressions. Ensure a consistent flow throughout \
        the conversation, maintaining coherence. Also, simplify any responses that are overly intricate or lengthy. \
        The primary goal is to enhance TEXT to create a conversation that effectively conveys information, is consistent, answers queries, \
        and guides the audience. Begin by introducing the guest appropriately, \
        and conclude the conversation with a gracious note, allowing the guest to have the final say. \
        As you enhance TEXT, focus on retaining both engagement and readability. Try to retain the lenght of the conversation. \
        Below, you will find the provided TEXT for your assessment.\n{conv}'
        messages =  [  
        {'role':'system', 
         'content':"""You are an expert writer, content creator, and popular science communicator."""},    
        {'role':'user', 'content':text},  
        ]
        ans, token_dict = get_completion_gpt4(messages, temperature=0)
        prRed(f'\nNumber of tokens used are: {token_dict}\n')
        save_output("Conversation_GPT4.txt", f'{ans}')
        prGreen("Conversation edited by GPT4!\n")
        subprocess.run(["open", f"./output/Conversation_GPT4.txt"], check=True)
        prGreenIn(f"\nNOW REVIEW Conversation_GPT4.txt AGAIN AND PRESS ENTER TO CONTINUE")
    else:
        prGreen('Conversation_GPT4.txt Already exist.\n')
    with open(f'./output/Conversation_GPT4.txt', 'r') as file:
    	conv = file.read()
    hostd, guestd = separate(conv,"Host","Guest")
    combined = combine(hostd, guestd)
    combined.export(f"./output/Conversation.mp3", format="mp3")
else:
	print(f"\nConversation.mp3 is already completed.")

###The closure
if not(os.path.exists('./output/final_p2.mp3')):
    prGreen("\nThe ending:")
    text = f'Based on the TEXT enclosed below, print out an engaging outro for an episode of a podcast named "The PaperCast". \
    Start by extending an appreciation to the authors of the paper. Invite listeners to suggest papers for the next episode. \nTEXT:\n{paper.summary}'
    messages =  [  
    {'role':'system', 
     'content':"""You are a popular science communicator"""},    
    {'role':'user', 'content':text},  
    ]
    ans, token_dict = get_completion_and_token_count(messages, temperature=0)
    prGreen("The outro:")
    print(f"{ans}\n")
    save_output("main.txt", ans)
    save_output("outro.txt", f'{ans}')
    subprocess.run(["open", "./output/outro.txt"], check=True)
    prGreenIn("\nNOW REVIEW outro.txt AND PRESS ENTER TO CONTINUE")
    gen_audio("./output/outro.txt", 'Stephen', 2)
else:
	print(f"\nOutro is already completed.")

###Convining intro, conversation, and outro.
prGreen('\nPreparing final Audio')
intro = AudioSegment.from_file("./output/final_p1.mp3", format="mp3")
conversation = AudioSegment.from_file("./output/Conversation.mp3", format="mp3")
outro = AudioSegment.from_file("./output/final_p2.mp3", format="mp3")
silence_duration = 3000
silence_segment = AudioSegment.silent(duration=silence_duration)
combined = intro + silence_segment + conversation + silence_segment + outro
combined.export(f"./output/Audio_{paper_id}.mp3", format="mp3")
prRed(f"FIND THE FINAL AUDIO IN Audio_{paper_id}.mp3 FILE.")



#remove = prGreenIn("\nIf you are ok with the result we can start cleaning up the folder ./output (yes/no): ")
#if remove == 'yes':
#    output_file = "./output/Conversation.txt"
#    for i in range(10):
#        with open(output_file, "w") as outfile:
#            with open(f'./output/conversation_{i}.txt', "r") as infile:
#                outfile.write(infile.read())
#        os.remove(f'./output/conversation_{i}.mp3')
#    prGreen("\n./output folder cleaned up!!")





