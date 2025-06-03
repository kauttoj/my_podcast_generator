# Configuration parameters (easy to modify)
BACKGROUND_FILE = "input_data/raportti_tekoälyn_käytöstä_mediayhtiöissä_äänipalveluissa_2024–2025.txt"
META_OUTPUT_FILE = "output_script/demo_podcast_metadata.json"
SCRIPT_OUTPUT_FILE = "output_script/demo_podcast_script.json"
PODCAST_OUTPUT_FILE = "output_script/demo_podcast_audio.wav"
SPEECH_OUTPUT_PATH = "output_script/demo_speech_files"
MUSIC_OUTPUT_PATH = "output_script/demo_music_files"

NUM_GUESTS = 2  # 1-3 guests
TARGET_LENGTH_MINUTES = 10 # polished final length will be approximately ~80% of this, while final audio could be more.
PODCAST_TOPIC = "Miten generatiivinen tekoäly mullistaa media-alaa äänituotannossa, kuten radiossa, podcasteissa ja äänimainonnassa. Miten tekoälyä käytetään juuri nyt ja mitä kaikkien alalla olevien tulisi tietää."
# Optional host description, leave None for LLM generated
HOST_DESCRIPTION = None
#HOST_DESCRIPTION = {'name':"Anna Hietanen",'description':'Erittäin kokenut esiintyjä, puhuja ja tieteen popularisoija, jolla on syvä tuntemus yritysmaailmasta, johtamisesta ja uusien teknologioiden hyödyntämisestä yritystoiminnassa. Tunnettu kiinnostavista ja helppokäyttöisistä selityksistään monimutkaisiin aiheisiin. Hän on isännöinyt useita populaaritieteellisiä podcasteja ja hänellä on kyky esittää oivaltavia kysymyksiä, jotka paljastavat miten teknologia ja tieteellinen tutkimus valjastetaan osaksi tuottavaa yritystoimintaa.','gender':'female'}

LANGUAGE = "finnish"  # enforce langauge of podcast

UTTERANCE_GAP = 0.80 # gap between utterances in seconds

# Optional TTS voice configuration - set to a dict to override automatic voice selection
# Example: {"host_name": "nova", "guest_name": "echo"}

# LLM configuration (easy to swap models)
if 1:
    PLANNER_LLM = 'openai/gpt-4.1'
    HOST_LLM = 'openai/gpt-4.1'
    GUEST_LLM = 'openai/gpt-4.1'
    FINALIZING_LLM = 'openai/gpt-4.1'
    SELECTOR_LLM = 'openai/gpt-4.1'
    AUDIO_LLM = 'openai/gpt-4.1'
    MUSIC_LLM = 'replicate'
    TEXT_TO_SPEECH = "gpt-4o-mini-tts"
    #TEXT_TO_SPEECH = "eleven_flash_v2_5"  # "eleven_multilingual_v2",
    #TEXT_TO_SPEECH = "eleven_multilingual_v2"
else:
    PLANNER_LLM = "gpt-4o-mini"
    HOST_LLM = "gpt-4o-mini"
    GUEST_LLM = "gpt-4o-mini"
    FINALIZING_LLM = 'o3-mini'
    SELECTOR_LLM = 'o3-mini'
    AUDIO_LLM = "gpt-4o"
    TEXT_TO_SPEECH = "gpt-4o-mini-tts"

TTS_VOICE_CONFIG = None

# Local LLM configuration (set USE_LOCAL_LLM to True to use local LLM)
USE_LOCAL_LLM = 0  # Set to False to use OpenAI/Anthropic instead of local LLM
LOCAL_LLM_BASE_URL = "http://localhost:1234/v1"
LOCAL_LLM_API_KEY = "lm-studio"
LOCAL_LLM_MODEL = "gemma-3-12b-instruct" #"deepthinkers-phi4"
LOCAL_LLM_MAX_TOKENS = 42672
LOCAL_LLM_TEMPERATURE = 0.60

# Word count to time conversion
WORDS_PER_MINUTE = {
    "english": 150,
    "finnish": 120
}

INTRO_OUTRO_MUSIC_LENGTH = 8
INTERMEDIATE_MUSIC_LENGTH = 5

# Valid stages for the podcast generation process
VALID_STAGES = ['metadata', 'raw_script', 'final_script', 'audio_speech', 'audio_ambient',"finalize_podcast"]
