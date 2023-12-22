from tortoise.api import TextToSpeech as TorToise_TTS, MODELS, get_model_path, pad_or_truncate

import os
import argparse
import json



import time
import math
import json
import base64
import re
import urllib.request
import signal
import gc
import subprocess
import psutil
import yaml
import hashlib
import string
import random

from tqdm import tqdm
import torch
import torchaudio
import music_tag
import gradio as gr
import gradio.utils
import pandas as pd
import numpy as np

from glob import glob
from datetime import datetime
from datetime import timedelta

from tortoise.api import TextToSpeech as TorToise_TTS, MODELS, get_model_path, pad_or_truncate
from tortoise.utils.audio import load_audio, load_voice, load_voices, get_voice_dir, get_voices
from tortoise.utils.text import split_and_recombine_text
from tortoise.utils.device import get_device_name, set_device_name, get_device_count, get_device_vram, get_device_batch_size, do_gc

try:
	from whisper.normalizers.english import EnglishTextNormalizer
	from whisper.normalizers.basic import BasicTextNormalizer
	from whisper.tokenizer import LANGUAGES 

	print("Whisper detected")
except Exception as e:
	if VERBOSE_DEBUG:
		print(traceback.format_exc())
	pass

def pad(num, zeroes):
	return str(num).zfill(zeroes+1)

language = "en"
validation_text_length = 12
validation_audio_length = 1
skip_existings = None
slice_audio = None
trim_silence = None
slice_start_offset = 0
slice_end_offset = 0

whisper_model = None
MODELS['dvae.pth'] = "https://huggingface.co/jbetker/tortoise-tts-v2/resolve/3704aea61678e7e468a06d8eea121dba368a798e/.models/dvae.pth"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v1", "large-v2"]
WHISPER_SPECIALIZED_MODELS = ["tiny.en", "base.en", "small.en", "medium.en"]
WHISPER_BACKENDS = ["openai/whisper", "lightmare/whispercpp", "m-bain/whisperx"]
VOCODERS = ['univnet', 'bigvgan_base_24khz_100band', 'bigvgan_24khz_100band']
TTSES = ['tortoise']

INFERENCING = False
GENERATE_SETTINGS_ARGS = None

LEARNING_RATE_SCHEMES = {"Multistep": "MultiStepLR", "Cos. Annealing": "CosineAnnealingLR_Restart"}
LEARNING_RATE_SCHEDULE = [ 2, 4, 9, 18, 25, 33, 50 ]

RESAMPLERS = {}

MIN_TRAINING_DURATION = 0.6
MAX_TRAINING_DURATION = 11.6097505669
MAX_TRAINING_CHAR_LENGTH = 200

VALLE_ENABLED = False
BARK_ENABLED = False

VERBOSE_DEBUG = True

KKS = None
PYKAKASI_ENABLED = False

def setup_args(cli=False):
	global args

	default_arguments = {
		'share': False,
		'listen': None,
		'check-for-updates': False,
		'models-from-local-only': False,
		'low-vram': False,
		'sample-batch-size': None,
		'unsqueeze-sample-batches': False,
		'embed-output-metadata': True,
		'latents-lean-and-mean': True,
		'voice-fixer': False, # getting tired of long initialization times in a Colab for downloading a large dataset for it
		'use-deepspeed': False,
		'voice-fixer-use-cuda': True,

		
		'force-cpu-for-conditioning-latents': False,
		'defer-tts-load': False,
		'device-override': None,
		'prune-nonfinal-outputs': True,
		'concurrency-count': 2,
		'autocalculate-voice-chunk-duration-size': 10,

		'output-sample-rate': 44100,
		'output-volume': 1,
		'results-folder': "./results/",
		
		'hf-token': None,
		'tts-backend': TTSES[0],
		
		'autoregressive-model': None,
		'diffusion-model': None,
		'vocoder-model': VOCODERS[-1],
		'tokenizer-json': None,
		'phonemizer-backend': 'espeak',
		'valle-model': None,
		'whisper-backend': 'openai/whisper',
		'whisper-model': "base",
		'whisper-batchsize': 1,
		'training-default-halfp': False,
		'training-default-bnb': True,
		'websocket-listen-address': "127.0.0.1",
		'websocket-listen-port': 8069,
		'websocket-enabled': False
	}

	if os.path.isfile('./config/exec.json'):
		with open(f'./config/exec.json', 'r', encoding="utf-8") as f:
			try:
				overrides = json.load(f)
				for k in overrides:
					default_arguments[k] = overrides[k]
			except Exception as e:
				print(e)
				pass
	print('./models/tortoise/autoregressive.pth')
	# print(default_arguments['autoregressive-model'])
	parser = argparse.ArgumentParser(allow_abbrev=not cli)
	parser.add_argument("--share", action='store_true', default=default_arguments['share'], help="Lets Gradio return a public URL to use anywhere")
	parser.add_argument("--listen", default=default_arguments['listen'], help="Path for Gradio to listen on")
	parser.add_argument("--check-for-updates", action='store_true', default=default_arguments['check-for-updates'], help="Checks for update on startup")
	parser.add_argument("--models-from-local-only", action='store_true', default=default_arguments['models-from-local-only'], help="Only loads models from disk, does not check for updates for models")
	parser.add_argument("--low-vram", action='store_true', default=default_arguments['low-vram'], help="Disables some optimizations that increases VRAM usage")
	parser.add_argument("--no-embed-output-metadata", action='store_false', default=not default_arguments['embed-output-metadata'], help="Disables embedding output metadata into resulting WAV files for easily fetching its settings used with the web UI (data is stored in the lyrics metadata tag)")
	parser.add_argument("--latents-lean-and-mean", action='store_true', default=default_arguments['latents-lean-and-mean'], help="Exports the bare essentials for latents.")
	parser.add_argument("--voice-fixer", action='store_true', default=default_arguments['voice-fixer'], help="Uses python module 'voicefixer' to improve audio quality, if available.")
	parser.add_argument("--voice-fixer-use-cuda", action='store_true', default=default_arguments['voice-fixer-use-cuda'], help="Hints to voicefixer to use CUDA, if available.")
	parser.add_argument("--use-deepspeed", action='store_true', default=default_arguments['use-deepspeed'], help="Use deepspeed for speed bump.")
	parser.add_argument("--force-cpu-for-conditioning-latents", default=default_arguments['force-cpu-for-conditioning-latents'], action='store_true', help="Forces computing conditional latents to be done on the CPU (if you constantyl OOM on low chunk counts)")
	parser.add_argument("--defer-tts-load", default=default_arguments['defer-tts-load'], action='store_true', help="Defers loading TTS model")
	parser.add_argument("--prune-nonfinal-outputs", default=default_arguments['prune-nonfinal-outputs'], action='store_true', help="Deletes non-final output files on completing a generation")
	parser.add_argument("--device-override", default=default_arguments['device-override'], help="A device string to override pass through Torch")
	parser.add_argument("--sample-batch-size", default=default_arguments['sample-batch-size'], type=int, help="Sets how many batches to use during the autoregressive samples pass")
	parser.add_argument("--unsqueeze-sample-batches", default=default_arguments['unsqueeze-sample-batches'], action='store_true', help="Unsqueezes sample batches to process one by one after sampling")
	parser.add_argument("--concurrency-count", type=int, default=default_arguments['concurrency-count'], help="How many Gradio events to process at once")
	parser.add_argument("--autocalculate-voice-chunk-duration-size", type=float, default=default_arguments['autocalculate-voice-chunk-duration-size'], help="Number of seconds to suggest voice chunk size for (for example, 100 seconds of audio at 10 seconds per chunk will suggest 10 chunks)")
	parser.add_argument("--output-sample-rate", type=int, default=default_arguments['output-sample-rate'], help="Sample rate to resample the output to (from 24KHz)")
	parser.add_argument("--output-volume", type=float, default=default_arguments['output-volume'], help="Adjusts volume of output")
	parser.add_argument("--results-folder", type=str, default=default_arguments['results-folder'], help="Sets output directory")
	
	parser.add_argument("--hf-token", type=str, default=default_arguments['hf-token'], help="HuggingFace Token")
	parser.add_argument("--tts-backend", default=default_arguments['tts-backend'], help="Specifies which TTS backend to use.")

	parser.add_argument("--autoregressive-model", default=default_arguments['autoregressive-model'], help="Specifies which autoregressive model to use for sampling.")
	parser.add_argument("--diffusion-model", default=default_arguments['diffusion-model'], help="Specifies which diffusion model to use for sampling.")
	parser.add_argument("--vocoder-model", default=default_arguments['vocoder-model'], action='store_true', help="Specifies with vocoder to use")
	parser.add_argument("--tokenizer-json", default=default_arguments['tokenizer-json'], help="Specifies which tokenizer json to use for tokenizing.")

	parser.add_argument("--phonemizer-backend", default=default_arguments['phonemizer-backend'], help="Specifies which phonemizer backend to use.")
	
	parser.add_argument("--valle-model", default=default_arguments['valle-model'], help="Specifies which VALL-E model to use for sampling.")
	
	parser.add_argument("--whisper-backend", default=default_arguments['whisper-backend'], action='store_true', help="Picks which whisper backend to use (openai/whisper, lightmare/whispercpp)")
	parser.add_argument("--whisper-model", default=default_arguments['whisper-model'], help="Specifies which whisper model to use for transcription.")
	parser.add_argument("--whisper-batchsize", type=int, default=default_arguments['whisper-batchsize'], help="Specifies batch size for WhisperX")
	
	parser.add_argument("--training-default-halfp", action='store_true', default=default_arguments['training-default-halfp'], help="Training default: halfp")
	parser.add_argument("--training-default-bnb", action='store_true', default=default_arguments['training-default-bnb'], help="Training default: bnb")
	
	parser.add_argument("--websocket-listen-port", type=int, default=default_arguments['websocket-listen-port'], help="Websocket server listen port, default: 8069")
	parser.add_argument("--websocket-listen-address", default=default_arguments['websocket-listen-address'], help="Websocket server listen address, default: 127.0.0.1")
	parser.add_argument("--websocket-enabled", action='store_true', default=default_arguments['websocket-enabled'], help="Websocket API server enabled, default: false")

	if cli:
		args, unknown = parser.parse_known_args()
	else:
		args = parser.parse_args()

	args.embed_output_metadata = not args.no_embed_output_metadata

	if not args.device_override:
		set_device_name(args.device_override)

	if args.sample_batch_size == 0 and get_device_batch_size() == 1:
		print("!WARNING! Automatically deduced sample batch size returned 1.")

	args.listen_host = None
	args.listen_port = None
	args.listen_path = None
	if args.listen:
		try:
			match = re.findall(r"^(?:(.+?):(\d+))?(\/.*?)?$", args.listen)[0]

			args.listen_host = match[0] if match[0] != "" else "127.0.0.1"
			args.listen_port = match[1] if match[1] != "" else None
			args.listen_path = match[2] if match[2] != "" else "/"
		except Exception as e:
			pass

	if args.listen_port is not None:
		args.listen_port = int(args.listen_port)
		if args.listen_port == 0:
			args.listen_port = None
	
	return args

def notify_progress(message, progress=None, verbose=True):
	if verbose:
		print(message)

	if progress is None:
		tqdm.write(message)
	else:
		progress(0, desc=message)

args = setup_args()

def load_tts( restart=False, 
	# TorToiSe configs
	autoregressive_model=None, diffusion_model=None, vocoder_model=None, tokenizer_json=None,
	# VALL-E configs
	valle_model=None,
):
	global args
	global tts

	if restart:
		unload_tts()

	tts_loading = True
	if args.tts_backend == "tortoise":
		if autoregressive_model:
			args.autoregressive_model = autoregressive_model
		else:
			autoregressive_model = args.autoregressive_model

		if autoregressive_model == "auto":
			autoregressive_model = deduce_autoregressive_model()

		if diffusion_model:
			args.diffusion_model = diffusion_model
		else:
			diffusion_model = args.diffusion_model

		if vocoder_model:
			args.vocoder_model = vocoder_model
		else:
			vocoder_model = args.vocoder_model

		if tokenizer_json:
			args.tokenizer_json = tokenizer_json
		else:
			tokenizer_json = args.tokenizer_json

		if get_device_name() == "cpu":
			print("!!!! WARNING !!!! No GPU available in PyTorch. You may need to reinstall PyTorch.")

		print(f"Loading TorToiSe... (AR: {autoregressive_model}, diffusion: {diffusion_model}, vocoder: {vocoder_model})")
		tts = TorToise_TTS(minor_optimizations=not args.low_vram, autoregressive_model_path=autoregressive_model, diffusion_model_path=diffusion_model, vocoder_model=vocoder_model, tokenizer_json=tokenizer_json, unsqueeze_sample_batches=args.unsqueeze_sample_batches, use_deepspeed=args.use_deepspeed)
	elif args.tts_backend == "vall-e":
		if valle_model:
			args.valle_model = valle_model
		else:
			valle_model = args.valle_model

		print(f"Loading VALL-E... (Config: {valle_model})")
		tts = VALLE_TTS(config=args.valle_model)
	elif args.tts_backend == "bark":

		print(f"Loading Bark...")
		tts = Bark_TTS(small=args.low_vram)

	print("Loaded TTS, ready for generation.")
	tts_loading = False
	return tts



def unload_tts():
	global tts

	if tts:
		del tts
		tts = None
		print("Unloaded TTS")
	do_gc()
def reload_tts():
	unload_tts()
	load_tts()

whisper_align_model = None
voicefixer = None

def load_voicefixer(restart=False):
	global voicefixer

	if restart:
		unload_voicefixer()

	try:
		print("Loading Voicefixer")
		from voicefixer import VoiceFixer
		voicefixer = VoiceFixer()
		print("Loaded Voicefixer")
	except Exception as e:
		print(f"Error occurred while tring to initialize voicefixer: {e}")
		if voicefixer:
			del voicefixer
		voicefixer = None


def unload_voicefixer():
	global voicefixer

	if voicefixer:
		del voicefixer
		voicefixer = None
		print("Unloaded Voicefixer")

	do_gc()


def unload_whisper():
	global whisper_model
	global whisper_align_model

	if whisper_align_model:
		del whisper_align_model
		whisper_align_model = None

	if whisper_model:
		del whisper_model
		whisper_model = None
		print("Unloaded Whisper")

	do_gc()	


def import_generate_settings(file = None):
	if not file:
		file = "./config/generate.json"

	res = {
		'text': None,
		'delimiter': None,
		'emotion': None,
		'prompt': None,
		'voice': "random",
		'mic_audio': None,
		'voice_latents_chunks': None,
		'candidates': None,
		'seed': None,
		'num_autoregressive_samples': 16,
		'diffusion_iterations': 30,
		'temperature': 0.8,
		'diffusion_sampler': "DDIM",
		'breathing_room': 8  ,
		'cvvp_weight': 0.0,
		'top_p': 0.8,
		'diffusion_temperature': 1.0,
		'length_penalty': 1.0,
		'repetition_penalty': 2.0,
		'cond_free_k': 2.0,
		'experimentals': None,
	}

	settings, _ = read_generate_settings(file, read_latents=False)

	if settings is not None:
		res.update(settings)
	
	return res

def read_generate_settings(file, read_latents=True):
	j = None
	latents = None

	if isinstance(file, list) and len(file) == 1:
		file = file[0]

	try:
		if file is not None:
			if hasattr(file, 'name'):
				file = file.name

			if file[-4:] == ".wav":
					metadata = music_tag.load_file(file)
					if 'lyrics' in metadata:
						j = json.loads(str(metadata['lyrics']))
			elif file[-5:] == ".json":
				with open(file, 'r') as f:
					j = json.load(f)
	except Exception as e:
		pass

	if j is not None:
		if 'latents' in j:
			if read_latents:
				latents = base64.b64decode(j['latents'])
			del j['latents']
		

		if "time" in j:
			j["time"] = "{:.3f}".format(j["time"])



	return (
		j,
		latents,
	)

def generate(**kwargs):
	return generate_tortoise(**kwargs)


def resample( waveform, input_rate, output_rate=44100 ):
	# mono-ize
	waveform = torch.mean(waveform, dim=0, keepdim=True)

	if input_rate == output_rate:
		return waveform, output_rate

	key = f'{input_rate}:{output_rate}'
	if not key in RESAMPLERS:
		RESAMPLERS[key] = torchaudio.transforms.Resample(
			input_rate,
			output_rate,
			lowpass_filter_width=16,
			rolloff=0.85,
			resampling_method="kaiser_window",
			beta=8.555504641634386,
		)

	return RESAMPLERS[key]( waveform ), output_rate

def slice_waveform( waveform, sample_rate, start, end, trim ):
	start = int(start * sample_rate)
	end = int(end * sample_rate)

	if start < 0:
		start = 0
	if end >= waveform.shape[-1]:
		end = waveform.shape[-1] - 1

	sliced = waveform[:, start:end]

	error = validate_waveform( sliced, sample_rate, min_only=True )
	if trim and not error:
		sliced = torchaudio.functional.vad( sliced, sample_rate )

	return sliced, error

def validate_waveform( waveform, sample_rate, min_only=False ):
	if not torch.any(waveform < 0):
		return "Waveform is empty"

	num_channels, num_frames = waveform.shape
	duration = num_frames / sample_rate
	
	if duration < MIN_TRAINING_DURATION:
		return "Duration too short ({:.3f}s < {:.3f}s)".format(duration, MIN_TRAINING_DURATION)

	if not min_only:
		if duration > MAX_TRAINING_DURATION:
			return "Duration too long ({:.3f}s < {:.3f}s)".format(MAX_TRAINING_DURATION, duration)

	return


def slice_dataset( voice, trim_silence=True, start_offset=0, end_offset=0, results=None, progress=gr.Progress() ):
	indir = f'./training/{voice}/'
	infile = f'{indir}/whisper.json'
	messages = []

	if not os.path.exists(infile):
		message = f"Missing dataset: {infile}"
		print(message)
		return message

	if results is None:
		results = json.load(open(infile, 'r', encoding="utf-8"))

	TARGET_SAMPLE_RATE = 22050
	if args.tts_backend != "tortoise":
		TARGET_SAMPLE_RATE = 24000
	if tts:
		TARGET_SAMPLE_RATE = tts.input_sample_rate

	files = 0
	segments = 0
	for filename in results:
		path = f'./voices/{voice}/{filename}'
		extension = os.path.splitext(filename)[-1][1:]
		out_extension = extension # "wav"

		if not os.path.exists(path):
			path = f'./training/{voice}/{filename}'

		if not os.path.exists(path):
			message = f"Missing source audio: {filename}"
			print(message)
			messages.append(message)
			continue

		files += 1
		result = results[filename]
		waveform, sample_rate = torchaudio.load(path)
		num_channels, num_frames = waveform.shape
		duration = num_frames / sample_rate

		for segment in result['segments']: 
			file = filename.replace(f".{extension}", f"_{pad(segment['id'], 4)}.{out_extension}")
			
			sliced, error = slice_waveform( waveform, sample_rate, segment['start'] + start_offset, segment['end'] + end_offset, trim_silence )
			if error:
				message = f"{error}, skipping... {file}"
				print(message)
				messages.append(message)
				continue
		
			sliced, _ = resample( sliced, sample_rate, TARGET_SAMPLE_RATE )

			if waveform.shape[0] == 2:
				waveform = waveform[:1]
				
			kwargs = {}
			if file[-4:] == ".wav":
				kwargs['encoding'] = "PCM_S"
				kwargs['bits_per_sample'] = 16

			torchaudio.save(f"{indir}/audio/{file}", sliced, TARGET_SAMPLE_RATE, **kwargs)
			
			segments +=1

	messages.append(f"Sliced segments: {files} => {segments}.")
	return "\n".join(messages)

def cleanup_voice_name( name ):
	return name.split("/")[-1]

def compute_latents(voice=None, voice_samples=None, voice_latents_chunks=0, original_ar=False, original_diffusion=False):
	global tts
	global args
	
	unload_whisper()
	unload_voicefixer()

	if not tts:
		if tts_loading:
			raise Exception("TTS is still initializing...")
		load_tts()

	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	if args.tts_backend == "bark":
		tts.create_voice( voice )
		return

	if args.autoregressive_model == "auto":
		tts.load_autoregressive_model(deduce_autoregressive_model(voice))

	if voice:
		load_from_dataset = voice_latents_chunks == 0

		if load_from_dataset:
			dataset_path = f'./training/{voice}/train.txt'
			if not os.path.exists(dataset_path):
				load_from_dataset = False
			else:
				with open(dataset_path, 'r', encoding="utf-8") as f:
					lines = f.readlines()

				print("Leveraging dataset for computing latents")

				voice_samples = []
				max_length = 0
				for line in lines:
					filename = f'./training/{voice}/{line.split("|")[0]}'
					
					waveform = load_audio(filename, 22050)
					max_length = max(max_length, waveform.shape[-1])
					voice_samples.append(waveform)

				for i in range(len(voice_samples)):
					voice_samples[i] = pad_or_truncate(voice_samples[i], max_length)

				voice_latents_chunks = len(voice_samples)
				if voice_latents_chunks == 0:
					print("Dataset is empty!")
					load_from_dataset = True
		if not load_from_dataset:
			voice_samples, _ = load_voice(voice, load_latents=False)

	if voice_samples is None:
		return

	conditioning_latents = tts.get_conditioning_latents(voice_samples, return_mels=not args.latents_lean_and_mean, slices=voice_latents_chunks, force_cpu=args.force_cpu_for_conditioning_latents, original_ar=original_ar, original_diffusion=original_diffusion)

	if len(conditioning_latents) == 4:
		conditioning_latents = (conditioning_latents[0], conditioning_latents[1], conditioning_latents[2], None)
	
	outfile = f'{get_voice_dir()}/{voice}/cond_latents_{tts.autoregressive_model_hash[:8]}.pth'
	torch.save(conditioning_latents, outfile)
	print(f'Saved voice latents: {outfile}')

	return conditioning_latents

def generate_tortoise(**kwargs):
	parameters = {}
	parameters.update(kwargs)

	voice = parameters['voice']
	progress = parameters['progress'] if 'progress' in parameters else None
	if parameters['seed'] == 0:
		parameters['seed'] = None

	usedSeed = parameters['seed']

	global args
	global tts

	unload_whisper()
	unload_voicefixer()

	if not tts:
		# should check if it's loading or unloaded, and load it if it's unloaded
		if tts_loading:
			raise Exception("TTS is still initializing...")
		load_tts()
	
	if hasattr(tts, "loading") and tts.loading:
		raise Exception("TTS is still initializing...")

	do_gc()

	voice_samples = None
	conditioning_latents = None
	sample_voice = None

	voice_cache = {}
	def fetch_voice( voice ):
		cache_key = f'{voice}:{tts.autoregressive_model_hash[:8]}'
		if cache_key in voice_cache:
			return voice_cache[cache_key]

		print(f"Loading voice: {voice} with model {tts.autoregressive_model_hash[:8]}")
		sample_voice = None
		if voice == "microphone":
			if parameters['mic_audio'] is None:
				raise Exception("Please provide audio from mic when choosing `microphone` as a voice input")
			voice_samples, conditioning_latents = [load_audio(parameters['mic_audio'], tts.input_sample_rate)], None
		elif voice == "random":
			voice_samples, conditioning_latents = None, tts.get_random_conditioning_latents()
		else:
			if progress is not None:
				notify_progress(f"Loading voice: {voice}", progress=progress)

			voice_samples, conditioning_latents = load_voice(voice, model_hash=tts.autoregressive_model_hash)
			
		if voice_samples and len(voice_samples) > 0:
			if conditioning_latents is None:
				conditioning_latents = compute_latents(voice=voice, voice_samples=voice_samples, voice_latents_chunks=parameters['voice_latents_chunks'])
				
			sample_voice = torch.cat(voice_samples, dim=-1).squeeze().cpu()
			voice_samples = None

		voice_cache[cache_key] = (voice_samples, conditioning_latents, sample_voice)
		return voice_cache[cache_key]

	def get_settings( override=None ):
		settings = {
			'temperature': float(parameters['temperature']),

			'top_p': float(parameters['top_p']),
			'diffusion_temperature': float(parameters['diffusion_temperature']),
			'length_penalty': float(parameters['length_penalty']),
			'repetition_penalty': float(parameters['repetition_penalty']),
			'cond_free_k': float(parameters['cond_free_k']),

			'num_autoregressive_samples': parameters['num_autoregressive_samples'],
			'sample_batch_size': args.sample_batch_size,
			'diffusion_iterations': parameters['diffusion_iterations'],

			'voice_samples': None,
			'conditioning_latents': None,

			'use_deterministic_seed': parameters['seed'],
			'return_deterministic_state': True,
			'k': parameters['candidates'],
			'diffusion_sampler': parameters['diffusion_sampler'],
			'breathing_room': parameters['breathing_room'],
			'half_p': "Half Precision" in parameters['experimentals'],
			'cond_free': "Conditioning-Free" in parameters['experimentals'],
			'cvvp_amount': parameters['cvvp_weight'],
			
			'autoregressive_model': args.autoregressive_model,
			'diffusion_model': args.diffusion_model,
			'tokenizer_json': args.tokenizer_json,
		}

		# could be better to just do a ternary on everything above, but i am not a professional
		selected_voice = voice
		if override is not None:
			if 'voice' in override:
				selected_voice = override['voice']

			for k in override:
				if k not in settings:
					continue
				settings[k] = override[k]

		if settings['autoregressive_model'] is not None:
			if settings['autoregressive_model'] == "auto":
				settings['autoregressive_model'] = deduce_autoregressive_model(selected_voice)
			tts.load_autoregressive_model(settings['autoregressive_model'])

		if settings['diffusion_model'] is not None:
			if settings['diffusion_model'] == "auto":
				settings['diffusion_model'] = deduce_diffusion_model(selected_voice)
			tts.load_diffusion_model(settings['diffusion_model'])
		
		if settings['tokenizer_json'] is not None:
			tts.load_tokenizer_json(settings['tokenizer_json'])

		settings['voice_samples'], settings['conditioning_latents'], _ = fetch_voice(voice=selected_voice)

		# clamp it down for the insane users who want this
		# it would be wiser to enforce the sample size to the batch size, but this is what the user wants
		settings['sample_batch_size'] = args.sample_batch_size
		if not settings['sample_batch_size']:
			settings['sample_batch_size'] = tts.autoregressive_batch_size
		if settings['num_autoregressive_samples'] < settings['sample_batch_size']:
			settings['sample_batch_size'] = settings['num_autoregressive_samples']

		if settings['conditioning_latents'] is not None and len(settings['conditioning_latents']) == 2 and settings['cvvp_amount'] > 0:
			print("Requesting weighing against CVVP weight, but voice latents are missing some extra data. Please regenerate your voice latents with 'Slimmer voice latents' unchecked.")
			settings['cvvp_amount'] = 0
			
		return settings

	if not parameters['delimiter']:
		parameters['delimiter'] = "\n"
	elif parameters['delimiter'] == "\\n":
		parameters['delimiter'] = "\n"

	if parameters['delimiter'] and parameters['delimiter'] != "" and parameters['delimiter'] in parameters['text']:
		texts = parameters['text'].split(parameters['delimiter'])
	else:
		texts = split_and_recombine_text(parameters['text'])
 
	full_start_time = time.time()
 
	outdir = f"{args.results_folder}/{voice}/"
	os.makedirs(outdir, exist_ok=True)

	audio_cache = {}

	volume_adjust = torchaudio.transforms.Vol(gain=args.output_volume, gain_type="amplitude") if args.output_volume != 1 else None

	idx = 0
	idx_cache = {}
	for i, file in enumerate(os.listdir(outdir)):
		filename = os.path.basename(file)
		extension = os.path.splitext(filename)[-1][1:]
		if extension != "json" and extension != "wav":
			continue
		match = re.findall(rf"^{voice}_(\d+)(?:.+?)?{extension}$", filename)
		if match and len(match) > 0:
			key = int(match[0])
			idx_cache[key] = True

	if len(idx_cache) > 0:
		keys = sorted(list(idx_cache.keys()))
		idx = keys[-1] + 1

	idx = pad(idx, 4)

	def get_name(line=0, candidate=0, combined=False):
		name = f"{idx}"
		if combined:
			name = f"{name}_combined"
		elif len(texts) > 1:
			name = f"{name}_{line}"
		if parameters['candidates'] > 1:
			name = f"{name}_{candidate}"
		return name

	def get_info( voice, settings = None, latents = True ):
		info = {}
		info.update(parameters)

		info['time'] = time.time()-full_start_time
		info['datetime'] = datetime.now().isoformat()

		info['model'] = tts.autoregressive_model_path
		info['model_hash'] = tts.autoregressive_model_hash 

		info['progress'] = None
		del info['progress']

		if info['delimiter'] == "\n":
			info['delimiter'] = "\\n"

		if settings is not None:
			for k in settings:
				if k in info:
					info[k] = settings[k]

			if 'half_p' in settings and 'cond_free' in settings:
				info['experimentals'] = []
				if settings['half_p']:
					info['experimentals'].append("Half Precision")
				if settings['cond_free']:
					info['experimentals'].append("Conditioning-Free")

		if latents and "latents" not in info:
			voice = info['voice']
			model_hash = settings["model_hash"][:8] if settings is not None and "model_hash" in settings else tts.autoregressive_model_hash[:8]

			dir = f'{get_voice_dir()}/{voice}/'
			latents_path = f'{dir}/cond_latents_{model_hash}.pth'

			if voice == "random" or voice == "microphone":
				if latents and settings is not None and settings['conditioning_latents']:
					os.makedirs(dir, exist_ok=True)
					torch.save(conditioning_latents, latents_path)

			if latents_path and os.path.exists(latents_path):
				try:
					with open(latents_path, 'rb') as f:
						info['latents'] = base64.b64encode(f.read()).decode("ascii")
				except Exception as e:
					pass

		return info

	INFERENCING = True
	for line, cut_text in enumerate(texts):
		if should_phonemize():
			cut_text = phonemizer( cut_text )

		if parameters['emotion'] == "Custom":
			if parameters['prompt'] and parameters['prompt'].strip() != "":
				cut_text = f"[{parameters['prompt']},] {cut_text}"
		elif parameters['emotion'] != "None" and parameters['emotion']:
			cut_text = f"[I am really {parameters['emotion'].lower()},] {cut_text}"
		
		tqdm_prefix = f'[{str(line+1)}/{str(len(texts))}]'
		print(f"{tqdm_prefix} Generating line: {cut_text}")
		start_time = time.time()

		# do setting editing
		match = re.findall(r'^(\{.+\}) (.+?)$', cut_text) 
		override = None
		if match and len(match) > 0:
			match = match[0]
			try:
				override = json.loads(match[0])
				cut_text = match[1].strip()
			except Exception as e:
				raise Exception("Prompt settings editing requested, but received invalid JSON")

		settings = get_settings( override=override )
		gen, additionals = tts.tts(cut_text, **settings )

		parameters['seed'] = additionals[0]
		run_time = time.time()-start_time
		print(f"Generating line took {run_time} seconds")

		if not isinstance(gen, list):
			gen = [gen]

		for j, g in enumerate(gen):
			audio = g.squeeze(0).cpu()
			name = get_name(line=line, candidate=j)

			settings['text'] = cut_text
			settings['time'] = run_time
			settings['datetime'] = datetime.now().isoformat()
			if args.tts_backend == "tortoise":
				settings['model'] = tts.autoregressive_model_path
				settings['model_hash'] = tts.autoregressive_model_hash

			audio_cache[name] = {
				'audio': audio,
				'settings': get_info(voice=override['voice'] if override and 'voice' in override else voice, settings=settings)
			}
			# save here in case some error happens mid-batch
			torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', audio, tts.output_sample_rate)

	del gen
	do_gc()
	INFERENCING = False

	for k in audio_cache:
		audio = audio_cache[k]['audio']

		audio, _ = resample(audio, tts.output_sample_rate, args.output_sample_rate)
		if volume_adjust is not None:
			audio = volume_adjust(audio)

		audio_cache[k]['audio'] = audio
		torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{k}.wav', audio, args.output_sample_rate)

	output_voices = []
	for candidate in range(parameters['candidates']):
		if len(texts) > 1:
			audio_clips = []
			for line in range(len(texts)):
				name = get_name(line=line, candidate=candidate)
				audio = audio_cache[name]['audio']
				audio_clips.append(audio)
			
			name = get_name(candidate=candidate, combined=True)
			audio = torch.cat(audio_clips, dim=-1)
			torchaudio.save(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav', audio, args.output_sample_rate)

			audio = audio.squeeze(0).cpu()
			audio_cache[name] = {
				'audio': audio,
				'settings': get_info(voice=voice),
				'output': True
			}
		else:
			name = get_name(candidate=candidate)
			audio_cache[name]['output'] = True


	if args.voice_fixer:
		if not voicefixer:
			notify_progress("Loading voicefix...", progress=progress)
			load_voicefixer()

		try:
			fixed_cache = {}
			for name in tqdm(audio_cache, desc="Running voicefix..."):
				del audio_cache[name]['audio']
				if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
					continue

				path = f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav'
				fixed = f'{outdir}/{cleanup_voice_name(voice)}_{name}_fixed.wav'
				voicefixer.restore(
					input=path,
					output=fixed,
					cuda=get_device_name() == "cuda" and args.voice_fixer_use_cuda,
					#mode=mode,
				)
				
				fixed_cache[f'{name}_fixed'] = {
					'settings': audio_cache[name]['settings'],
					'output': True
				}
				audio_cache[name]['output'] = False
			
			for name in fixed_cache:
				audio_cache[name] = fixed_cache[name]
		except Exception as e:
			print(e)
			print("\nFailed to run Voicefixer")

	for name in audio_cache:
		if 'output' not in audio_cache[name] or not audio_cache[name]['output']:
			if args.prune_nonfinal_outputs:
				audio_cache[name]['pruned'] = True
				os.remove(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')
			continue

		output_voices.append(f'{outdir}/{cleanup_voice_name(voice)}_{name}.wav')

		if not args.embed_output_metadata:
			with open(f'{outdir}/{cleanup_voice_name(voice)}_{name}.json', 'w', encoding="utf-8") as f:
				f.write(json.dumps(audio_cache[name]['settings'], indent='\t') )

	if args.embed_output_metadata:
		for name in tqdm(audio_cache, desc="Embedding metadata..."):
			if 'pruned' in audio_cache[name] and audio_cache[name]['pruned']:
				continue

			metadata = music_tag.load_file(f"{outdir}/{cleanup_voice_name(voice)}_{name}.wav")
			metadata['lyrics'] = json.dumps(audio_cache[name]['settings'])
			metadata.save()
 
	if sample_voice is not None:
		sample_voice = (tts.input_sample_rate, sample_voice.numpy())

	info = get_info(voice=voice, latents=False)
	print(f"Generation took {info['time']} seconds, saved to '{output_voices[0]}'\n")

	info['seed'] = usedSeed
	if 'latents' in info:
		del info['latents']

	os.makedirs('./config/', exist_ok=True)
	with open(f'./config/generate.json', 'w', encoding="utf-8") as f:
		f.write(json.dumps(info, indent='\t') )

	stats = [
		[ parameters['seed'], "{:.3f}".format(info['time']) ]
	]

	return (
		sample_voice,
		output_voices,
		stats,
	)

def should_phonemize():
	if args.tts_backend == "vall-e":
		return False
		
	should = args.tokenizer_json is not None and args.tokenizer_json[-8:] == "ipa.json"
	if should:
		try:
			from phonemizer import phonemize
		except Exception as e:
			return False
	return should


def prepare_dataset( voice, use_segments=False, text_length=0, audio_length=0, progress=gr.Progress() ):
	indir = f'./training/{voice}/'
	infile = f'{indir}/whisper.json'
	if not os.path.exists(infile):
		message = f"Missing dataset: {infile}"
		print(message)
		return message

	results = json.load(open(infile, 'r', encoding="utf-8"))

	errored = 0
	messages = []
	normalize = False # True
	phonemize = should_phonemize()
	lines = { 'training': [], 'validation': [] }
	segments = {}

	quantize_in_memory = args.tts_backend == "vall-e"

	if args.tts_backend != "tortoise":
		text_length = 0
		audio_length = 0

	start_offset = -0.1
	end_offset = 0.1
	trim_silence = False

	TARGET_SAMPLE_RATE = 22050
	if args.tts_backend != "tortoise":
		TARGET_SAMPLE_RATE = 24000
	if tts:
		TARGET_SAMPLE_RATE = tts.input_sample_rate

	for filename in tqdm(results, desc="Parsing results"):
		use_segment = use_segments

		extension = os.path.splitext(filename)[-1][1:]
		out_extension = extension # "wav"
		result = results[filename]
		lang = result['language']
		language = LANGUAGES[lang] if lang in LANGUAGES else lang
		normalizer = EnglishTextNormalizer() if language and language == "english" else BasicTextNormalizer()

		# check if unsegmented text exceeds 200 characters
		if not use_segment:
			if len(result['text']) > MAX_TRAINING_CHAR_LENGTH:
				message = f"Text length too long ({MAX_TRAINING_CHAR_LENGTH} < {len(result['text'])}), using segments: {filename}"
				print(message)
				messages.append(message)
				use_segment = True

		# check if unsegmented audio exceeds 11.6s
		if not use_segment:
			path = f'{indir}/audio/{filename}'
			if not quantize_in_memory and not os.path.exists(path):
				messages.append(f"Missing source audio: {filename}")
				errored += 1
				continue

			duration = 0
			for segment in result['segments']:
				duration = max(duration, segment['end'])

			if duration >= MAX_TRAINING_DURATION:
				message = f"Audio too large, using segments: {filename}"
				print(message)
				messages.append(message)
				use_segment = True

		# implicitly segment
		if use_segment and not use_segments:
			exists = True
			for segment in result['segments']:
				duration = segment['end'] - segment['start']
				if duration <= MIN_TRAINING_DURATION or MAX_TRAINING_DURATION <= duration:
					continue

				path = f'{indir}/audio/' + filename.replace(f".{extension}", f"_{pad(segment['id'], 4)}.{out_extension}")
				if os.path.exists(path):
					continue
				exists = False
				break

			if not quantize_in_memory and not exists:
				tmp = {}
				tmp[filename] = result
				print(f"Audio not segmented, segmenting: {filename}")
				message = slice_dataset( voice, results=tmp )
				print(message)
				messages = messages + message.split("\n")
		
		waveform = None
		

		if quantize_in_memory:
			path = f'{indir}/audio/{filename}'
			if not os.path.exists(path):
				path = f'./voices/{voice}/{filename}'

			if not os.path.exists(path):
				message = f"Audio not found: {path}"
				print(message)
				messages.append(message)
				#continue
			else:
				waveform = torchaudio.load(path)
				waveform = resample(waveform[0], waveform[1], TARGET_SAMPLE_RATE)

		if not use_segment:
			segments[filename] = {
				'text': result['text'],
				'lang': lang,
				'language': language,
				'normalizer': normalizer,
				'phonemes': result['phonemes'] if 'phonemes' in result else None
			}

			if waveform:
				segments[filename]['waveform'] = waveform
		else:
			for segment in result['segments']:
				duration = segment['end'] - segment['start']
				if duration <= MIN_TRAINING_DURATION or MAX_TRAINING_DURATION <= duration:
					continue

				file = filename.replace(f".{extension}", f"_{pad(segment['id'], 4)}.{out_extension}")

				segments[file] = {
					'text': segment['text'],
					'lang': lang,
					'language': language,
					'normalizer': normalizer,
					'phonemes': segment['phonemes'] if 'phonemes' in segment else None
				}

				if waveform:
					sliced, error = slice_waveform( waveform[0], waveform[1], segment['start'] + start_offset, segment['end'] + end_offset, trim_silence )
					if error:
						message = f"{error}, skipping... {file}"
						print(message)
						messages.append(message)
						segments[file]['error'] = error
						#continue
					else:
						segments[file]['waveform'] = (sliced, waveform[1])

	jobs = {
		'quantize':  [[], []],
		'phonemize': [[], []],
	}

	for file in tqdm(segments, desc="Parsing segments"):
		extension = os.path.splitext(file)[-1][1:]
		result = segments[file]
		path = f'{indir}/audio/{file}'

		text = result['text']
		lang = result['lang']
		language = result['language']
		normalizer = result['normalizer']
		phonemes = result['phonemes']
		if phonemize and phonemes is None:
			phonemes = phonemizer( text, language=lang )
		
		normalized = normalizer(text) if normalize else text

		if len(text) > MAX_TRAINING_CHAR_LENGTH:
			message = f"Text length too long ({MAX_TRAINING_CHAR_LENGTH} < {len(text)}), skipping... {file}"
			print(message)
			messages.append(message)
			errored += 1
			continue

		# num_channels, num_frames = waveform.shape
		#duration = num_frames / sample_rate


		culled = len(text) < text_length
		if not culled and audio_length > 0:
			culled = duration < audio_length

		line = f'audio/{file}|{phonemes if phonemize and phonemes else text}'

		lines['training' if not culled else 'validation'].append(line) 

		if culled or args.tts_backend != "vall-e":
			continue
		
		os.makedirs(f'{indir}/valle/', exist_ok=True)
		#os.makedirs(f'./training/valle/data/{voice}/', exist_ok=True)

		phn_file = f'{indir}/valle/{file.replace(f".{extension}",".phn.txt")}'
		#phn_file = f'./training/valle/data/{voice}/{file.replace(f".{extension}",".phn.txt")}'
		if not os.path.exists(phn_file):
			jobs['phonemize'][0].append(phn_file)
			jobs['phonemize'][1].append(normalized)
			"""
			phonemized = valle_phonemize( normalized )
			open(f'{indir}/valle/{file.replace(".wav",".phn.txt")}', 'w', encoding='utf-8').write(" ".join(phonemized))
			print("Phonemized:", file, normalized, text)
			"""

		qnt_file = f'{indir}/valle/{file.replace(f".{extension}",".qnt.pt")}'
		#qnt_file = f'./training/valle/data/{voice}/{file.replace(f".{extension}",".qnt.pt")}'
		if 'error' not in result:
			if not quantize_in_memory and not os.path.exists(path):
				message = f"Missing segment, skipping... {file}"
				print(message)
				messages.append(message)
				errored += 1
				continue

		if not os.path.exists(qnt_file):
			waveform = None
			if 'waveform' in result:
				waveform, sample_rate = result['waveform']
			elif os.path.exists(path):
				waveform, sample_rate = torchaudio.load(path)
				error = validate_waveform( waveform, sample_rate )
				if error:
					message = f"{error}, skipping... {file}"
					print(message)
					messages.append(message)
					errored += 1
					continue

			if waveform is not None:
				jobs['quantize'][0].append(qnt_file)
				jobs['quantize'][1].append((waveform, sample_rate))
				"""
				quantized = valle_quantize( waveform, sample_rate ).cpu()
				torch.save(quantized, f'{indir}/valle/{file.replace(".wav",".qnt.pt")}')
				print("Quantized:", file)
				"""

	for i in tqdm(range(len(jobs['quantize'][0])), desc="Quantizing"):
		qnt_file = jobs['quantize'][0][i]
		waveform, sample_rate = jobs['quantize'][1][i]

		quantized = valle_quantize( waveform, sample_rate ).cpu()
		torch.save(quantized, qnt_file)
		#print("Quantized:", qnt_file)

	for i in tqdm(range(len(jobs['phonemize'][0])), desc="Phonemizing"):
		phn_file = jobs['phonemize'][0][i]
		normalized = jobs['phonemize'][1][i]

		if language == "japanese":
			language = "ja"

		if language == "ja" and PYKAKASI_ENABLED and KKS is not None:
			normalized = KKS.convert(normalized)
			normalized = [ n["hira"] for n in normalized ]
			normalized = "".join(normalized)

		try:
			phonemized = valle_phonemize( normalized )
			open(phn_file, 'w', encoding='utf-8').write(" ".join(phonemized))
			#print("Phonemized:", phn_file)
		except Exception as e:
			message = f"Failed to phonemize: {phn_file}: {normalized}"
			messages.append(message)
			print(message)


	training_joined = "\n".join(lines['training'])
	validation_joined = "\n".join(lines['validation'])

	with open(f'{indir}/train.txt', 'w', encoding="utf-8") as f:
		f.write(training_joined)

	with open(f'{indir}/validation.txt', 'w', encoding="utf-8") as f:
		f.write(validation_joined)

	messages.append(f"Prepared {len(lines['training'])} lines (validation: {len(lines['validation'])}, culled: {errored}).\n{training_joined}\n\n{validation_joined}")
	return "\n".join(messages)


def get_voice( name, dir=get_voice_dir(), load_latents=True, extensions=["wav", "mp3", "flac"] ):
	subj = f'{dir}/{name}/'
	if not os.path.isdir(subj):
		return
	files = os.listdir(subj)
	
	if load_latents:
		extensions.append("pth")

	voice = []
	for file in files:
		ext = os.path.splitext(file)[-1][1:]
		if ext not in extensions:
			continue

		voice.append(f'{subj}/{file}') 

	return sorted( voice )


def load_whisper_model(language=None, model_name=None, progress=None):
	global whisper_model
	global whisper_align_model

	if args.whisper_backend not in WHISPER_BACKENDS:
		raise Exception(f"unavailable backend: {args.whisper_backend}")

	if not model_name:
		model_name = args.whisper_model
	else:
		args.whisper_model = model_name
		save_args_settings()

	if language and f'{model_name}.{language}' in WHISPER_SPECIALIZED_MODELS:
		model_name = f'{model_name}.{language}'
		print(f"Loading specialized model for language: {language}")

	notify_progress(f"Loading Whisper model: {model_name}", progress=progress)

	if args.whisper_backend == "openai/whisper":
		import whisper
		try:
			#is it possible for model to fit on vram but go oom later on while executing on data?
			whisper_model = whisper.load_model(model_name)
		except:
			print("Out of VRAM memory. falling back to loading Whisper on CPU.")
			whisper_model = whisper.load_model(model_name, device="cpu")
	elif args.whisper_backend == "lightmare/whispercpp":
		from whispercpp import Whisper
		if not language:
			language = 'auto'

		b_lang = language.encode('ascii')
		whisper_model = Whisper(model_name, models_dir='./models/', language=b_lang)
	elif args.whisper_backend == "m-bain/whisperx":
		import whisper, whisperx
		device = "cuda" if get_device_name() == "cuda" else "cpu"
		whisper_model = whisperx.load_model(model_name, device)
		whisper_align_model = whisperx.load_align_model(model_name="WAV2VEC2_ASR_LARGE_LV60K_960H" if language=="en" else None, language_code=language, device=device)

	print("Loaded Whisper model")

# collapses short segments into the previous segment
def whisper_sanitize( results ):
	sanitized = json.loads(json.dumps(results))
	sanitized['segments'] = []

	for segment in results['segments']:
		length = segment['end'] - segment['start']
		if length >= MIN_TRAINING_DURATION or len(sanitized['segments']) == 0:
			sanitized['segments'].append(segment)
			continue

		last_segment = sanitized['segments'][-1]
		# segment already asimilitated it, somehow
		if last_segment['end'] >= segment['end']:
			continue
		"""
		# segment already asimilitated it, somehow
		if last_segment['text'].endswith(segment['text']):
			continue
		"""
		last_segment['text'] += segment['text']
		last_segment['end'] = segment['end']

	for i in range(len(sanitized['segments'])):
		sanitized['segments'][i]['id'] = i

	return sanitized

def whisper_transcribe( file, language=None ):
	# shouldn't happen, but it's for safety
	global whisper_model
	global whisper_align_model

	if not whisper_model:
		load_whisper_model(language=language)

	if args.whisper_backend == "openai/whisper":
		if not language:
			language = None

		return whisper_model.transcribe(file, language=language)

	if args.whisper_backend == "lightmare/whispercpp":
		res = whisper_model.transcribe(file)
		segments = whisper_model.extract_text_and_timestamps( res )

		result = {
			'text': [],
			'segments': []
		}
		for segment in segments:
			reparsed = {
				'start': segment[0] / 100.0,
				'end': segment[1] / 100.0,
				'text': segment[2],
				'id': len(result['segments'])
			}
			result['text'].append( segment[2] )
			result['segments'].append(reparsed)

		result['text'] = " ".join(result['text'])
		return result

	if args.whisper_backend == "m-bain/whisperx":
		import whisperx

		device = "cuda" if get_device_name() == "cuda" else "cpu"
		result = whisper_model.transcribe(file, batch_size=args.whisper_batchsize)
			
		align_model, metadata = whisper_align_model
		result_aligned = whisperx.align(result["segments"], align_model, metadata, file, device, return_char_alignments=False)

		result['segments'] = result_aligned['segments']
		result['text'] = []
		for segment in result['segments']:
			segment['id'] = len(result['text'])
			result['text'].append(segment['text'].strip())
		result['text'] = " ".join(result['text'])

		return result

def transcribe_dataset( voice, language=None, skip_existings=False, progress=None ):
	print(voice)
	unload_tts()

	global whisper_model
	if whisper_model is None:
		load_whisper_model(language=language)

	results = {}

	files = get_voice(voice, load_latents=False)
	print(files)
	indir = f'./training/{voice}/'
	infile = f'{indir}/whisper.json'

	quantize_in_memory = args.tts_backend == "vall-e"
	
	os.makedirs(f'{indir}/audio/', exist_ok=True)
	
	TARGET_SAMPLE_RATE = 22050
	if args.tts_backend != "tortoise":
		TARGET_SAMPLE_RATE = 24000
	if tts:
		TARGET_SAMPLE_RATE = tts.input_sample_rate

	if os.path.exists(infile):
		results = json.load(open(infile, 'r', encoding="utf-8"))

	for file in tqdm(files, desc="Iterating through voice files"):
		basename = os.path.basename(file)

		if basename in results and skip_existings:
			print(f"Skipping already parsed file: {basename}")
			continue

		try:
			result = whisper_transcribe(file, language=language)
		except Exception as e:
			print("Failed to transcribe:", file, e)
			continue

		results[basename] = result

		if not quantize_in_memory:
			waveform, sample_rate = torchaudio.load(file)
			# resample to the input rate, since it'll get resampled for training anyways
			# this should also "help" increase throughput a bit when filling the dataloaders
			waveform, sample_rate = resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
			if waveform.shape[0] == 2:
				waveform = waveform[:1]
			
			try:
				kwargs = {}
				if basename[-4:] == ".wav":
					kwargs['encoding'] = "PCM_S"
					kwargs['bits_per_sample'] = 16

				torchaudio.save(f"{indir}/audio/{basename}", waveform, sample_rate, **kwargs)
			except Exception as e:
				print(e)

		with open(infile, 'w', encoding="utf-8") as f:
			f.write(json.dumps(results, indent='\t'))

		do_gc()

	modified = False
	for basename in results:
		try:
			sanitized = whisper_sanitize(results[basename])
			if len(sanitized['segments']) > 0 and len(sanitized['segments']) != len(results[basename]['segments']):
				results[basename] = sanitized
				modified = True
				print("Segments sanizited: ", basename)
		except Exception as e:
			print("Failed to sanitize:", basename, e)
			pass

	if modified:
		os.rename(infile, infile.replace(".json", ".unsanitized.json"))
		with open(infile, 'w', encoding="utf-8") as f:
			f.write(json.dumps(results, indent='\t'))

	return f"Processed dataset to: {indir}"

def prepare_all_datasets( voices, language, validation_text_length, validation_audio_length, skip_existings, slice_audio, trim_silence, slice_start_offset, slice_end_offset, progress=None ):
	kwargs = locals()

	messages = []

	# for voice in voices:
	# 	print("Processing:", voice)
	# 	message = transcribe_dataset( voice=voice, language=language, skip_existings=skip_existings, progress=progress )
	# 	messages.append(message)

	if slice_audio:
		for voice in voices:
			print("Processing:", voice)
			message = slice_dataset( voice, trim_silence=trim_silence, start_offset=slice_start_offset, end_offset=slice_end_offset, results=None, progress=progress )
			messages.append(message)
			
	for voice in voices:
		print("Processing:", voice)
		message = prepare_dataset( voice, use_segments=slice_audio, text_length=validation_text_length, audio_length=validation_audio_length, progress=progress )
		messages.append(message)

	return "\n".join(messages)

