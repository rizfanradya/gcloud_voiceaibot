import os
import queue
import re
import sys
import signal
from google.cloud import speech, texttospeech
import pyaudio
import threading
from pyaudio import PyAudio, paInt16
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINIAI_API_KEY = os.environ.get('GEMINIAI_API_KEY')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./voice_ai_bot_service_account.json"
genai.configure(api_key=GEMINIAI_API_KEY)
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms
stop_flag = False


def signal_handler(sig, frame):
    print("\nClose Session...")
    global stop_flag
    stop_flag = True
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


class MicrophoneStream:
    def __init__(self: object, rate: int = RATE, chunk: int = CHUNK) -> None:
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self: object) -> object:
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

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
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
    tts_client = texttospeech.TextToSpeechClient()
    cleaned_text = re.sub(r"[^\w\s]", "", text)
    synthesis_input = texttospeech.SynthesisInput(
        text=cleaned_text
    )
    voice = texttospeech.VoiceSelectionParams(
        language_code='id-ID',
        name='id-ID-Wavenet-D'
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000
    )
    tts_response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    p_audio = PyAudio()
    tts_stream = p_audio.open(
        format=paInt16,
        channels=1,
        rate=16000,
        output=True
    )
    tts_stream.write(tts_response.audio_content)
    tts_stream.stop_stream()
    tts_stream.close()
    p_audio.terminate()


def process_responses(responses):
    num_chars_printed = 0
    for response in responses:
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

            # nlp
            try:
                model_genai = genai.GenerativeModel("gemini-1.5-flash")
                response_genai = model_genai.generate_content(transcript)
                print("GeminiAI Response:", response_genai.text)

                # text to speech
                try:
                    text_to_speech(response_genai.text)
                except Exception as error:
                    print(error)
            except Exception as error:
                print(error)

            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break
            num_chars_printed = 0


if __name__ == "__main__":
    text_to_speech("Hallo ada yang bisa saya bantu")
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
