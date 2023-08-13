import re
import os

#n is the number of lines to remove from the top, and m is the number of lines to remove from bottom.
def remove_text(text, guest="", n=0,m=0, ts=1):
    # Remove "VERONICA: or Veronica: "
    text = text.replace("Podcast Host: ", "")
    text = text.replace("Guest: ", "")
    text = text.replace(guest, "")

    # Remove words enclosed in [] and ()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    
    # Remove empty lines
    text = "\n".join(line for line in text.split("\n") if line.strip())
    
    if n != 0 or m != 0:
        # Remove the first n lines, and last m lines
        lines = text.split('\n')
        text = '\n'.join(lines[n:-m])

    # Remove empty lines
    text = "\n".join(line for line in text.split("\n") if line.strip())
    
    #Replace \n with ". "
    if ts == 1:
        text = text.replace('\n', '\n<break time="500ms"/>\n')
    
    return text

#Create the file where the output is saved
def save_output(filename: str, texto: str):
    directory = './output/'

    file_path = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'Directory {directory} doesnt exist. Created now!')

    if not os.path.exists(file_path):
        # File does not exist, create it
        with open(file_path, 'w') as file:
            print(f'File {file_path} doesnt exist. Created now!')  # This creates an empty file

    with open(file_path, 'a') as file:
        # Append text to the file
        file.write(texto)
    return f"Content saved to {file_path}."