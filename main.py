import os
import queue
import re
import sys
import threading
from google.cloud import speech, texttospeech
import pyaudio
import signal
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINIAI_API_KEY = os.getenv('GEMINIAI_API_KEY')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./voice_ai_bot_service_account.json"
genai.configure(api_key=GEMINIAI_API_KEY)
RATE = 16000
CHUNK = int(RATE / 10)
stop_flag = False


def signal_handler(sig, frame):
    print("\nClose Session...")
    global stop_flag
    stop_flag = True
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)


def text_to_speech(text):
    global stop_flag
    tts_client = texttospeech.TextToSpeechClient()
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    synthesis_input = texttospeech.SynthesisInput(text=cleaned_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code='id-ID',
        name='id-ID-Wavenet-D'
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000
    )

    if stop_flag:
        return
    tts_response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    p_audio = pyaudio.PyAudio()
    tts_stream = p_audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        output=True
    )

    audio_data = tts_response.audio_content
    for chunk_start in range(0, len(audio_data), 1024):
        if stop_flag:
            break
        tts_stream.write(audio_data[chunk_start:chunk_start + 1024])

    tts_stream.stop_stream()
    tts_stream.close()
    p_audio.terminate()


def process_responses(responses):
    global stop_flag
    num_chars_printed = 0
    for response in responses:
        if stop_flag:
            print("Processing stopped.")
            break
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            print(transcript + overwrite_chars)
            if re.search(r"\b(stop|exit|quit)\b", transcript, re.I):
                print("Stopping the program...")
                stop_flag = True
                break

            # NLP processing
            try:
                if stop_flag:
                    break
                model_genai = genai.GenerativeModel("gemini-1.5-flash")
                response_genai = model_genai.generate_content(transcript)
                print("GeminiAI Response:", response_genai.text)

                # Text to Speech
                if not stop_flag:
                    text_to_speech(response_genai.text)
            except Exception as error:
                print(error)

            num_chars_printed = 0


if __name__ == "__main__":
    text_to_speech("Hello, how can I help you?")
    language_code = "id-ID"
    stt_client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream_gcloud:
        audio_generator = stream_gcloud.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = stt_client.streaming_recognize(streaming_config, requests)

        response_thread = threading.Thread(
            target=process_responses,
            args=(responses,),
            daemon=True
        )
        response_thread.start()

        while not stop_flag:
            response_thread.join(timeout=0.1)
