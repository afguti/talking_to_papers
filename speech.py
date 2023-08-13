import boto3
#s3 = boto3.client('s3')
s3 = boto3.resource('s3')
import time

def start_polly_synthesis_task(texttype="ssml", engine="neural", region='ap-northeast-1', endpoint_url='https://polly.ap-northeast-1.amazonaws.com/', output_format='mp3', 
bucket_name='podcast-wellness-e1', s3_key_prefix='prueba', voice_id='Ruth', text_file_path='./input.txt'):
    # Create a Boto3 client for Polly
    polly_client = boto3.client('polly', region_name=region, endpoint_url=endpoint_url)
    # Read the text content from the file
    with open(text_file_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    # Start the speech synthesis task
    response = polly_client.start_speech_synthesis_task(
        TextType=texttype,
        Engine=engine,
        OutputFormat=output_format,
        OutputS3BucketName=bucket_name,
        OutputS3KeyPrefix=s3_key_prefix,
        VoiceId=voice_id,
        Text=text
    )
    return response['SynthesisTask']['TaskId'] 

def check_task_status(region, task_id):
    # Create a Boto3 client for Polly
    polly_client = boto3.client('polly', region_name=region)

    # Check the status of the synthesis task
    response = polly_client.get_speech_synthesis_task(TaskId=task_id)
    status = response['SynthesisTask']['TaskStatus']

    return status

def tts(texttype="ssml", engine="neural", region='ap-northeast-1', endpoint_url='https://polly.ap-northeast-1.amazonaws.com/', output_format='mp3', 
bucket_name='podcast-wellness-e1', s3_key_prefix='prueba', voice_id='Ruth', text_file_path='./input.txt', output_path='./output.mp3'):
    task_id = start_polly_synthesis_task(texttype, engine, region, endpoint_url, output_format, bucket_name, s3_key_prefix, voice_id, text_file_path)
    print(f'Started Polly speech synthesis task with ID: {task_id}')
    while True:
        status = check_task_status(region, task_id)
        if status == 'completed':
            print('Speech synthesis task is completed.')
            break
        elif status == 'failed':
            print('Speech synthesis task failed.')
            break
        elif status == 'scheduled':
            print('Speech synthesis task is in scheduled.')
            time.sleep(5) #wait
        elif status == 'inProgress':
            print('Speech synthesis task is still in progress. Waiting...')
            time.sleep(5)  # Wait for a few seconds before checking again
        else:
            print('Unexpected task status:', status)
            break
    your_bucket = s3.Bucket(bucket_name)
    for s3_file in your_bucket.objects.all():
        print(s3_file.key)
    your_bucket.download_file(f'{s3_key_prefix}.{task_id}.mp3',output_path)
    print(f'Successfully downloaded the output audio to: {output_path}')
    return None
    








