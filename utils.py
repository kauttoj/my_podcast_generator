import math
import datetime
import os
import json
from markitdown import MarkItDown
import re
import keyword
import unicodedata
import random
import ffmpeg

import moviepy

from moviepy import ImageClip, AudioFileClip, TextClip, CompositeVideoClip
from moviepy.video.fx.Resize import Resize
from moviepy.video.fx.Crop   import Crop

# Import aisuite for single LLM calls
import aisuite as ai

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

import params

def update_stage_status(stage_name, status="completed"):
    """Update a stage's status in SCRIPT_OUTPUT_FILE."""
    if stage_name not in params.VALID_STAGES:
        print(f"Warning: Invalid stage name '{stage_name}'. Valid stages are: {params.VALID_STAGES}")
        return False
        
    try:
        data = {}
        if os.path.exists(params.SCRIPT_OUTPUT_FILE):
            with open(params.SCRIPT_OUTPUT_FILE, 'r', encoding='utf-8') as file:
                data = json.load(file)
        
        # Initialize stages if not present
        if "stages" not in data:
            data["stages"] = {}
        
        # Update stage status
        data["stages"][stage_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update last_updated timestamp
        data["last_updated"] = datetime.now().isoformat()
        
        # Write back to file
        with open(params.SCRIPT_OUTPUT_FILE, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=2)
            
        return True
    except Exception as e:
        print(f"Warning: Could not update stage status: {e}")
        return False

def check_stage_status(stage_name):
    """Check if a stage is completed in SCRIPT_OUTPUT_FILE."""
    if stage_name not in params.VALID_STAGES:
        raise Exception(f"Invalid stage name '{stage_name}'. Valid stages are: {params.VALID_STAGES}")
                
    try:
        if os.path.exists(params.SCRIPT_OUTPUT_FILE):
            with open(params.SCRIPT_OUTPUT_FILE, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if "stages" in data and stage_name in data["stages"]:
                    return data["stages"][stage_name]["status"] == "completed"
    except Exception as e:
        print(f"Warning: Could not check stage status: {e}")
    return False

def can_proceed_to_stage(stage_name):
    """Check if we can proceed to a stage based on dependencies."""
    if stage_name not in params.VALID_STAGES:
        print(f"Warning: Invalid stage name '{stage_name}'. Valid stages are: {params.VALID_STAGES}")
        return False
        
    # Define stage dependencies in code
    dependencies = {
        "metadata": [],
        "raw_script": ["metadata"],
        "final_script": ["raw_script"],
        "audio_speech": ["final_script"],
        "audio_ambient": ["audio_speech"],
        "finalize_podcast": ["audio_ambient"]
    }
    
    # Check if all dependencies are completed
    for dep in dependencies.get(stage_name, []):
        if not check_stage_status(dep):
            print(f"Cannot proceed to {stage_name}: {dep} is not completed")
            return False
    return True

def to_valid_variable(name: str) -> str:
    """Convert any string into a valid Python variable name (ASCII letters, digits, and underscores).

    Steps:
      1. Strip leading/trailing whitespace.
      2. Normalize Unicode (NFKD) and drop non-ASCII.
      3. Replace any non-word characters with underscores.
      4. Trim extra underscores.
      5. Ensure it starts with a letter or underscore (prepend “_” if not).
      6. If it’s a Python keyword, append an underscore.
      7. If it still isn’t a valid identifier (unlikely), force it by replacing invalid chars and prefixing “_”.
      8. Fallback to "_var" if empty.

    Examples:
      >>> to_valid_variable("Ångström-värde")
      'Angstrom_varde'
      >>> to_valid_variable("123count")
      '_123count'
      >>> to_valid_variable("class")
      'class_'
      >>> to_valid_variable("$$$")
      '_var'
    """
    # 1. Strip whitespace
    name = name.strip()

    # Early fallback
    if not name:
        return "_var"

    # 2. Unicode → ASCII
    name = unicodedata.normalize('NFKD', name)
    name = name.encode('ascii', 'ignore').decode('ascii')

    # 3. Non-word → underscore
    name = re.sub(r'\W+', '_', name)

    # 4. Trim underscores
    name = name.strip('_')

    # 5. Must start with letter or underscore
    if not name or not (name[0].isalpha() or name[0] == '_'):
        name = '_' + name

    # 6. Avoid keywords
    if keyword.iskeyword(name):
        name += '_'

    # 7. Guarantee a valid identifier
    if not name.isidentifier():
        # replace any remaining invalids
        name = re.sub(r'[^0-9A-Za-z_]', '_', name)
        # ensure start
        if not (name[0].isalpha() or name[0] == '_'):
            name = '_' + name

    # 8. Final fallback
    return name or ("temp_var_" + str(random.randint(1,1000)))

def generate_podcast_video(output_file, bg_path, audio_path, transcript,
                           video_size=(1920, 1080), fps=24,
                           font="Arial-Bold", font_size=48, font_color="white",
                           text_width_ratio=0.8, scroll_speed=100,
                           codec="libx265", audio_codec="aac",
                           preset="medium", crf=23):
    """
    Generate a video with a static, cropped-to-fill background and smoothly scrolling transcript.
    Returns True on success, False on failure.
    """
    try:
        # 1. Load audio
        audio_clip = AudioFileClip(audio_path)
        duration   = audio_clip.duration

        # 2. Prepare background: scale to cover, then center-crop
        bg     = ImageClip(bg_path)
        bw, bh = bg.size
        vw, vh = video_size
        scale  = max(vw/bw, vh/bh)

        # Resize via FX class (lowercase fx modules don’t exist in v2.x) :contentReference[oaicite:0]{index=0}
        bg_resized = Resize(bg, width=int(bw*scale), height=int(bh*scale))

        # Crop then immediately assign a duration with .with_duration() (replacing .set_duration) :contentReference[oaicite:1]{index=1}
        bg_clip    = (
            Crop(
                bg_resized,
                x_center=int(bw*scale/2),
                y_center=int(bh*scale/2),
                width=vw,
                height=vh
            )
            .with_duration(duration)  # ← replaced set_duration() with with_duration() :contentReference[oaicite:2]{index=2}
        )

        # 3. Build full transcript text block
        full_text = "\n\n".join(text for (_, _, text) in transcript)
        text_clip = TextClip(
            txt=full_text,
            fontsize=font_size,
            font=font,
            color=font_color,
            method="caption",
            size=(int(vw * text_width_ratio), None),
            align="West"
        )

        # 4. Animate vertical scrolling via .with_position() (replacing .set_pos()) :contentReference[oaicite:3]{index=3}
        def scrolling_pos(t):
            return ("center", vh - scroll_speed * t)

        scrolling = (
            text_clip
            .with_position(scrolling_pos)
            .with_duration(duration)  # ensure caption clip matches audio length :contentReference[oaicite:4]{index=4}
        )

        # 5. Composite layers and attach audio via .with_audio() (replacing .set_audio()) :contentReference[oaicite:5]{index=5}
        final = (
            CompositeVideoClip([bg_clip, scrolling], size=video_size)
            .with_audio(audio_clip)
        )

        # 6. Render with HEVC (x265)
        final.write_videofile(
            output_file,
            fps=fps,
            codec=codec,
            audio_codec=audio_codec,
            preset=preset,
            ffmpeg_params=["-crf", str(crf)]
        )

        return True

    except Exception as e:
        print(f"Error generating video: {e}")
        return False

def get_merged_audio(output_file: str, input_files: list[str], gap_lims = None) -> str:
    """
    Merge a list of WAV (or MP3) files into one WAV, inserting `gap` seconds
    of silence between each, and normalizing all audio to 16-bit PCM at
    the highest sample rate and channel count found.

    Args:
        output_file (str): Path where the merged WAV will be saved.
        input_files (list[str]): Ordered list of source audio file paths.
        gap (float): Duration in seconds of silence between each file.

    Returns:
        str: The `output_file` path.
    """
    if not input_files:
        raise ValueError("`input_files` must contain at least one file")

    # 1. Probe each file for audio specs
    specs = []
    for path in input_files:
        info = ffmpeg.probe(path)
        stream = next(s for s in info['streams'] if s['codec_type']=='audio')
        specs.append({
            'sr': int(stream['sample_rate']),
            'ch': int(stream['channels']),
            'cl': stream.get('channel_layout', 'stereo'),
        })

    # 2. Choose highest-quality sample rate and channels
    max_sr = max(s['sr'] for s in specs)
    max_ch = max(s['ch'] for s in specs)
    # Prefer a layout matching max_ch, else default
    layouts = [s['cl'] for s in specs if s['ch']==max_ch]
    max_cl = layouts[0] if layouts else ('stereo' if max_ch>1 else 'mono')

    # 3. Build normalized inputs and silence segments
    streams = []
    for i, path in enumerate(input_files):
        # Normalize real audio to s16 @ max_sr, max_cl
        inp = (
            ffmpeg.input(path)
            .filter(
                'aformat',
                sample_fmts='s16',
                sample_rates=max_sr,
                channel_layouts=max_cl
            )
        )  # :contentReference[oaicite:5]{index=5}
        streams.append(inp)

        # Insert silence between files
        if i < len(input_files) - 1 and gap_lims:

            gap = gap_lims[0] + random.random() * (gap_lims[1] - gap_lims[0])

            sil = (
                ffmpeg.input(
                    f'anullsrc=channel_layout={max_cl}:sample_rate={max_sr}',
                    format='lavfi',
                    t=gap
                )
                .filter(
                    'aformat',
                    sample_fmts='s16',
                    sample_rates=max_sr,
                    channel_layouts=max_cl
                )
            )  # :contentReference[oaicite:6]{index=6}
            streams.append(sil)

    # 4. Concatenate all normalized streams (audio only)
    merged = ffmpeg.concat(*streams, v=0, a=1)  # :contentReference[oaicite:7]{index=7}

    # 5. Output with enforced WAV specs (pcm_s16le)
    (
        merged
        .output(
            output_file,
            ar=max_sr,             # set sample rate
            ac=max_ch,             # set channel count
            sample_fmt='s16'       # enforce 16-bit PCM
        )
        .run(overwrite_output=True)
    )

    return output_file

def get_faded_wav(input_path: str, start: float = 0.0, end: float = 0.0) -> str:
    """
    Apply linear fade-in over the first `start` seconds and fade-out
    over the last `end` seconds of a WAV file. Writes the result to
    a new file with "_edited" before the extension.

    Args:
        input_path (str): Path to the source WAV file.
        start (float): Duration in seconds for fade-in at the beginning.
        end (float): Duration in seconds for fade-out at the end.

    Returns:
        str: Path to the newly created WAV file, e.g.
             "my_original_file.wav" → "my_original_file_edited.wav"
    """
    # 1. Probe input WAV to get duration
    probe_info = ffmpeg.probe(input_path)
    audio_stream = next(
        s for s in probe_info['streams']
        if s['codec_type'] == 'audio'
    )
    duration = float(audio_stream['duration'])

    # 2. Compute fade-out start time
    fade_out_start = max(0.0, duration - end)

    # 3. Construct output file path
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_edited{ext}"

    # 4. Build filter graph: fade-in then fade-out
    stream = ffmpeg.input(input_path)
    if start > 0.0:
        stream = stream.filter('afade', t='in', st=0, d=start)
    if end > 0.0:
        stream = stream.filter('afade', t='out', st=fade_out_start, d=end)
    (
        stream
        .output(output_path)
        .run(overwrite_output=True)
    )

    return output_path

def get_aisuite_model_name(model_name: str, use_local_llm=False, local_llm_model=None) -> str:
    """Convert internal model name to aisuite format.
    
    For aisuite, we need to prefix the model with the provider:
    - OpenAI models: "openai:gpt-4o"
    - Anthropic models: "anthropic:claude-3-5-sonnet-20240620"
    - Local LLM: Use the configured LOCAL_LLM_MODEL directly
    """
    if use_local_llm:
        # For local LLM, return the model name directly
        # aisuite client is already configured with the correct base_url
        return local_llm_model
    elif model_name.startswith("gpt") or model_name.startswith("o3"):
        return f"openai:{model_name}"
    elif model_name.startswith("gemini"):
        return f"google:{model_name}"    
    elif model_name.startswith("claude"):
        return f"anthropic:{model_name}"
    else:
        raise ValueError(f"Unsupported model for aisuite: {model_name}")

# Initialize AI client for single LLM calls
def get_ai_client(use_local_llm=False):
    """Get the AI client for single LLM calls."""
    if use_local_llm:
        # For local LLM, we'll use OpenAI client directly instead of aisuite
        # We'll configure this in the respective methods
        return None
    else:
        # Use default OpenAI/Anthropic configuration with aisuite
        return ai.Client()

def read_background_material(background_file) -> str:

    def clean_markdown_links_safely(text, max_url_length=300):
        """
        Removes markdown hyperlinks [text](url) where url is a valid http/https link and not unreasonably long.
        Returns the cleaned text and the list of removed hyperlinks.
        """
        pattern = re.compile(r'\[([^\]]{1,200})\]\((https?://[^)\s]{1,' + str(max_url_length) + r'})\)')

        removed_links = []

        def replacement(match):
            link_text, url = match.groups()
            removed_links.append({'text': link_text, 'url': url})
            return link_text  # retain the human-readable text only

        cleaned_text = pattern.sub(replacement, text)
        return cleaned_text, removed_links

    """Read background material from file."""
    try:

        if 1:
            md = MarkItDown(enable_plugins=True)  # Set to True to enable plugins
            result = md.convert(background_file)
            background_material_raw = result.text_content
        else:

            with open(background_file, 'r', encoding='utf-8') as file:
                background_material_raw = file.read()

        cleaned_text, removed_links = clean_markdown_links_safely(background_material_raw)

        background_material = cleaned_text

        if not background_material:
            raise Exception(f"Background material file '{background_file}' is empty")
            
        return background_material
    except Exception as e:
        error_msg = f"Error reading background material: {e}"
        print(error_msg)
        raise Exception(error_msg)


def estimate_duration(text: str, language="english", words_per_minute=None) -> int:
    """Estimate the duration of spoken text in seconds."""
    # Count words
    words = len(re.findall(r'\w+', text))
    
    # Calculate duration based on language
    if words_per_minute is None:
        words_per_minute = 150 if language.lower() == "english" else 120
    
    duration_minutes = words / words_per_minute
    
    # Convert to seconds and round up
    return math.ceil(duration_minutes * 60)


def _get_model_client(model_name: str, use_local_llm=False, local_llm_model=None, 
                     local_llm_api_key=None, local_llm_base_url=None, 
                     local_llm_max_tokens=None, local_llm_temperature=None):
    """Get the appropriate model client based on the model name.
    
    This is used for group chat functionality which still uses AutoGen.
    For single LLM calls, use aisuite directly.
    """
    if use_local_llm:
        # Use local LLM for autogen with built-in configuration
        return OpenAIChatCompletionClient(
            model=local_llm_model,
            api_key=local_llm_api_key,
            base_url=local_llm_base_url,
            api_type="openai",
            max_tokens=local_llm_max_tokens,
            temperature=local_llm_temperature,
            model_info = {
                "name": local_llm_model,   # Model identifier
                "max_context": 32000,       # Maximum context length in tokens
                "vision": False,           # Whether the model supports vision (required)
                "function_calling": False, # Whether the model supports function calling (required)
                "json_output": False, 
                'structured_output':True,
                "family": "gpt-4o",
                "token_limit": 15000
            }    
        )
    elif "gpt" in model_name or "o3" in model_name:

        return OpenAIChatCompletionClient(
            model=model_name.replace('openai/', ''),
            api_key=os.getenv("OPENAI_API_KEY")
        )

    elif "claude" in model_name:

        from autogen_core.models import ModelFamily

        return AnthropicChatCompletionClient(
            model=model_name.replace('anthropic/', ''),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            # model_info = {
            #       "vision": False,
            #       "function_calling": True,
            #       "json_output": False,
            #      'structured_output': True,
            #     'family':ModelFamily.CLAUDE_3_5_SONNET
            # }
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")
