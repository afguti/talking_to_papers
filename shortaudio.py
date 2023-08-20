import boto3
import os

from pydub import AudioSegment, silence #for mp3 manipulation. pip install pydub

def voice(text = '', voice = 'Joanna', output='./output/speech.mp3'):
    polly = boto3.client('polly')
    response = polly.synthesize_speech(VoiceId=voice, OutputFormat='mp3', Text=text, Engine = 'neural')
    file = open(output, 'wb')
    file.write(response['AudioStream'].read())
    file.close()

def combine(hostd=[], guestd=[]):
    hostdl = []
    guestdl = []
    for i in range(len(hostd)):
        voice(hostd[i],'Stephen',f'./output/que{i+1}.mp3') #Voice of the host
        hostdl.append(AudioSegment.from_file(f'./output/que{i+1}.mp3', format="mp3") + 8)
        os.remove(f'./output/que{i+1}.mp3')
    for i in range(len(guestd)):
        voice(guestd[i],'Matthew',f'./output/ans{i+1}.mp3') #Voice of the guest: Matthew (20s), Joey (40s), Stephen (30s) Joanna (30s), Ruth (20s), Kendra (40s)
        guestdl.append(AudioSegment.from_file(f'./output/ans{i+1}.mp3', format="mp3") + 8)
        os.remove(f'./output/ans{i+1}.mp3')
    silence_duration = 1000
    silence_segment = AudioSegment.silent(duration=silence_duration)
    minlen = min(len(hostd), len(guestd))
    combined = hostdl[0] + silence_segment + guestdl[0] + silence_segment
    for i in range(minlen-1):
        combined = combined + hostdl[i+1] + silence_segment + guestdl[i+1] + silence_segment
    if len(hostd) != len(guestd) & max(len(hostd), len(guestd)) == len(hostd):
        combined = combined + hostdl[len(hostd)-1] + silence_segment
    elif len(hostd) != len(guestd) & max(len(hostd), len(guestd)) == len(guestd):
    #elif:
        combined = combined + guestdl[len(guestd)-1] + silence_segment
    return combined