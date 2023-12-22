import os
import argparse


if 'TORTOISE_MODELS_DIR' not in os.environ:
	os.environ['TORTOISE_MODELS_DIR'] = os.path.realpath(os.path.join(os.getcwd(), './models/tortoise/'))

if 'TRANSFORMERS_CACHE' not in os.environ:
	os.environ['TRANSFORMERS_CACHE'] = os.path.realpath(os.path.join(os.getcwd(), './models/transformers/'))

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import pesq
import librosa
from functions import *
# from generate import *
def calc_pesq(original,cloned):
    # Load the reference and degraded audio signals
    ref, sr_ref = librosa.load(original, sr=None)
    deg, sr_deg = librosa.load(cloned, sr=None)
    # Ensure both signals have the same sample rate (common_sample_rate)
    common_sample_rate = 16000  # Set your desired sample rate
    # Calculate the PESQ score
    try:
        pesq_score = pesq.pesq(common_sample_rate, ref, deg, 'wb')
    except:
        pesq_score = None

    
    return pesq_score

if __name__ == "__main__":
	args = setup_args(cli=True)

	default_arguments = import_generate_settings()
	parser = argparse.ArgumentParser(allow_abbrev=False)
	parser.add_argument("--text", default=default_arguments['text'])
	parser.add_argument("--delimiter", default=default_arguments['delimiter'])
	parser.add_argument("--emotion", default=default_arguments['emotion'])
	parser.add_argument("--prompt", default=default_arguments['prompt'])
	parser.add_argument("--voice", default=default_arguments['voice'])
	parser.add_argument("--mic_audio", default=default_arguments['mic_audio'])
	parser.add_argument("--voice_latents_chunks", default=default_arguments['voice_latents_chunks'])
	parser.add_argument("--candidates", default=default_arguments['candidates'])
	parser.add_argument("--seed", default=default_arguments['seed'])
	parser.add_argument("--num_autoregressive_samples", default=default_arguments['num_autoregressive_samples'])
	parser.add_argument("--diffusion_iterations", default=default_arguments['diffusion_iterations'])
	parser.add_argument("--temperature", default=default_arguments['temperature'])
	parser.add_argument("--diffusion_sampler", default=default_arguments['diffusion_sampler'])
	parser.add_argument("--breathing_room", default=default_arguments['breathing_room'])
	parser.add_argument("--cvvp_weight", default=default_arguments['cvvp_weight'])
	parser.add_argument("--top_p", default=default_arguments['top_p'])
	parser.add_argument("--diffusion_temperature", default=default_arguments['diffusion_temperature'])
	parser.add_argument("--length_penalty", default=default_arguments['length_penalty'])
	parser.add_argument("--repetition_penalty", default=default_arguments['repetition_penalty'])
	parser.add_argument("--cond_free_k", default=default_arguments['cond_free_k'])

	args, unknown = parser.parse_known_args()

	voice_directory = '..\\voices'

	def find_folders_with_substring(directory, substring):
		matching_folders = []
		for root, dirs, files in os.walk(directory):
			for dir_name in dirs:
				if substring in dir_name:
					matching_folders.append(os.path.join(root, dir_name))
		return matching_folders
	
	
	
	data = {
		"voice":[
			'newyoushang',
		   'Obama',
		   'aayush',
		   'sai' ,
			 'harsha'],
		'text':[
		'How about this way? Can you try to',
		'In the desktop, not here, you have to finish, first finish in the desktop, then you can go back to update here. So in this case because it takes a while so that we don\'t know how. What is happening here, right?',
		  'The President and I have just concluded a very productive meeting in the Oval Office on the urgent and overriding challenges before us helping the people of Haiti as they recover and rebuild after one of the most devastating natural disasters.',
		  'The development is set to transform an old industrial site into an interconnected smart city.',
		  'The model here I can see in only JSON format professor not in the, with, format so it is showing us up we I have the training pretrained models',
		  'For a second I thought I could see through the roof, and the stars swarmed before me. It was as though I was at the vortex of a high whirlwind of dancing, shining specks of light.'],
		'ground_truth_path':[
			'C:\\Users\\yzhang21\\clone\\ozen-toolkit\\output\\newyoushang_2023_11_20-23_05\\wavs\\3.wav',
					   'C:\\Users\\yzhang21\\Downloads\\Barack_Obama_&_René_Préval_in_the_Rose_Garden_2010-03-10 (mp3cut.net).wav',
					   'C:\\Users\\yzhang21\\clone\\ozen-toolkit\\output\\aayush_2023_12_18-00_29\\wavs\\14.wav',
					   'C:\\Users\\yzhang21\\Downloads\\sai_input.wav',
					   'C:\\Users\\yzhang21\\clone\\try2\\ai-voice-cloning\\voices\\harsha\\13.wav']

	}

	df = pd.DataFrame(data)
	tts_score = pd.DataFrame(columns=['Speaker','Input file','Output file','PESQ score'])
	for i in range(len(df)):
		if df.loc[i,'text'] == '':
			pass
		else:
			print(df.loc[i,'text'],df.loc[i,'voice'],df.loc[i,'ground_truth_path'])
			kwargs = {
				'text': df.loc[i,'text'],
				'delimiter': args.delimiter,
				'emotion': args.emotion,
				'prompt': args.prompt,
				'voice': df.loc[i,'voice'],
				'mic_audio': args.mic_audio,
				'voice_latents_chunks': args.voice_latents_chunks,
				'candidates': args.candidates,
				'seed': args.seed,
				'num_autoregressive_samples': args.num_autoregressive_samples,
				'diffusion_iterations': args.diffusion_iterations,
				'temperature': args.temperature,
				'diffusion_sampler': args.diffusion_sampler,
				'breathing_room': args.breathing_room,
				'cvvp_weight': args.cvvp_weight,
				'top_p': args.top_p,
				'diffusion_temperature': args.diffusion_temperature,
				'length_penalty': args.length_penalty,
				'repetition_penalty': args.repetition_penalty,
				'cond_free_k': args.cond_free_k,
				'experimentals': default_arguments['experimentals'],
			}
			# Replace "/your/directory/path" with the actual path to the directory
			directory_path = "./training/"
			substring_to_find = df.loc[i,'voice']
			matching_folders = find_folders_with_substring(directory_path, substring_to_find)
			direc = matching_folders[-1]+'/finetune/models/'
			for root, dirs, files in os.walk(direc):
				ar_model = os.path.join(direc,files[0])
			tts = load_tts(autoregressive_model=ar_model)
			details = generate(**kwargs)

			audio_file_name = os.path.normpath(details[1][0])
			input_file = df.loc[i,'ground_truth_path']
			print(audio_file_name,input_file)
			pesq_score = calc_pesq(input_file,audio_file_name)
			print(f"PESQ Score: {pesq_score:.2f}")


	# csv_file_path = 'output_pesq1.csv'
	# tts_score.to_csv(csv_file_path, index=False) 
	#tts_score.to_cvs(cvs_file_path, index=False)
	#cvs_file_path = 'output_pesq1.cvs'
	