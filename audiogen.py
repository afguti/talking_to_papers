from mutagen.mp3 import MP3 #pip install mutagen
from pydub import AudioSegment, silence #for mp3 manipulation. pip install pydub

from speech import tts
from bsoundprompt import prompt
from filehandle import remove_text, save_output

#Font colors
def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreenIn(skk): input("\033[92m {}\033[00m" .format(skk))

def audio(resp: str, voice = 'Ruth', ch = 1):
	with open(resp, 'r') as file:
		content = file.read()
	resp1 = remove_text(content,'',0,0,1) 
	resp1 = '<speak>\n'+resp1+'\n</speak>'
	save_output(f"resp{ch}_2.txt", resp1)
	prGreenIn(f"\nREVIEW resp{ch}_2.txt FOR tts AND PRESS ENTER TO CONTINUE")
	tts(engine="neural", region='ap-northeast-1', endpoint_url='https://polly.ap-northeast-1.amazonaws.com/', output_format='mp3', 
	bucket_name='podcast-wellness-e1', s3_key_prefix='prueba', voice_id=voice, text_file_path=f'./output/resp{ch}_2.txt', output_path=f'./output/part{ch}.mp3')
	audio = MP3(f"./output/part{ch}.mp3")
	audio_lenght=int(audio.info.length)+6
	prRed(f'\naudio lenght for Part{ch}: {audio_lenght} seconds\n')
	prompt1 = prompt(content)
	lines = prompt1.split('\n')
	last_two = lines[-3:] #How many lines of the prompt for background audio we will generate
	last_lines = "\n".join(last_two)
	prRed(f'\nPrompt to generate Part{ch} background sound: ')
	print(last_lines+"\n")
	#we use the Colab from https://github.com/facebookresearch/audiocraft to generate background audio
	prGreenIn(f"\nNOW BASED ON THE PROMPT ABOVE, GENERATE BACKGROUND SOUND, NAME IT background{ch}.mp4, AND PRESS ENTER TO CONTINUE") #Need to automate this part
	backg = AudioSegment.from_file(f"./output/background{ch}.mp4", format="mp4")
	backg = backg - 20 #Here we can adjust the volume of the background sound
	backg = backg * (int(audio_lenght)+1)
	backg = backg[0:audio_lenght*1000]
	faded_sound = backg.fade_out(3000) #Fade out the background audio the last 3 seconds
	talk = AudioSegment.from_file(f"./output/part{ch}.mp3", format="mp3")
	talk = talk + 8 #This is the volume of the speech
	overlay1 = faded_sound.overlay(talk, position=3000)
	file_handle = overlay1.export(f'./output/final_p{ch}.mp3', format='mp3')
	prRed(f"\nPART {ch} IS COMPLETED!")
	return None
