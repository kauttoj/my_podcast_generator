#!/usr/bin/env python3

"""
Podcast Simulator - Creates simulated podcast discussions using LLMs

This script generates a simulated podcast discussion based on background materials,
using Microsoft Autogen 0.5+ to manage the conversation between LLM agents.
"""
import re
import os
import wave
import json
import time
import math
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from openai import OpenAI
import ffmpeg
from tokencost import count_message_tokens, count_string_tokens
import re
import openai
from elevenlabs import ElevenLabs
import replicate
from dotenv import load_dotenv
from litellm import completion

# Import functions from utils.py
from utils import (
    to_valid_variable, get_ai_client, get_faded_wav, get_merged_audio,
    read_background_material, estimate_duration, update_stage_status, check_stage_status, can_proceed_to_stage,
)

# Import functions from dialogue_simulation.py
from dialogue_simulation import get_podcast_dialogue

import params

# Load environment variables from .env file
load_dotenv()

def get_podcast_metadata():
    """Generate podcast title, description, and participant profiles."""
    try:
        # Initialize script object
        script_obj = {"metadata":{"target_duration": params.TARGET_LENGTH_MINUTES * 60}}
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(params.META_OUTPUT_FILE), exist_ok=True)
        
        # Read background material
        script_obj["metadata"]["background_material"] = read_background_material(params.BACKGROUND_FILE)
        
        # Get AI client
        ai_client = get_ai_client(params.USE_LOCAL_LLM)
        
        # Prepare the prompt for metadata generation
        topic_info = f"on the topic of {params.PODCAST_TOPIC}" if params.PODCAST_TOPIC else ""
        host_info = f"with the following host: {params.HOST_DESCRIPTION}" if params.HOST_DESCRIPTION else ""
        
        # Create messages for the LLM call
        messages = [
            {
                "role": "system", 
                "content": "You are a podcast planning assistant that creates detailed podcast metadata and participant profiles."
            },
            {
                "role": "user", 
                "content": f"""Based on the following background material {topic_info} {host_info}, generate:

1. A catchy podcast title
2. A brief podcast overall description (3-6 sentences), what is it all about?   
3. List of 10 relevant questions to be discussed with guests
4. Detailed profiles for {params.NUM_GUESTS} guests (name, gender, background, expertise, personality)
{"4. A profile for the podcast host (name, background, style)" if not params.HOST_DESCRIPTION else ""}

Format your response as a JSON object with the following structure:
{{
    "title": "Podcast Title",
    "description": "4-5 sentence description",
    "questions": "10 potential questions to be discussed",
    {'' if params.HOST_DESCRIPTION else '"host": {{' + '''
        "name": "Host Name",
        "gender: "Host gender (male or female)",
        "description": "Host background and style"
    }},'''}
    "guests": [
        {{
            "name": "Guest 1 Name",
            "gender: "Guest 1 gender (male or female)",
            "description": "Guest 1 background, expertise, and personality"
        }},
        ...
    ]
}}

Background Material:
{script_obj["metadata"]["background_material"]}"""
            }
        ]
        
        # Make the API call
        if params.USE_LOCAL_LLM:
            # For local LLM, use OpenAI client directly
            client = openai.OpenAI(
                api_key=params.LOCAL_LLM_API_KEY,
                base_url=params.LOCAL_LLM_BASE_URL
            )
            response = client.chat.completions.create(
                model=params.LOCAL_LLM_MODEL,
                messages=messages,
                temperature=0.7
            )
        else:

            response = completion(model=params.PLANNER_LLM, messages=messages)

            # # Use aisuite for OpenAI/Anthropic
            # response = ai_client.chat.completions.create(
            #     model=get_aisuite_model_name(params.PLANNER_LLM, params.USE_LOCAL_LLM, params.LOCAL_LLM_MODEL),
            #     messages=messages,
            #     temperature=0.7
            # )
        
        # Extract content from the response
        content = response.choices[0].message.content
        
        # Extract JSON from the response
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if not json_match:
                raise Exception("Failed to extract JSON from LLM response")
            json_str = json_match.group(1)
        
        script_obj["metadata"] = script_obj["metadata"] | json.loads(json_str)
        
        # Format participants list
        if params.HOST_DESCRIPTION:
            # Parse the provided HOST_DESCRIPTION to extract name and description
            # Assuming HOST_DESCRIPTION is in format "Name: Description"
            host_name = params.HOST_DESCRIPTION['name']
            host_gender = params.HOST_DESCRIPTION['gender']
            host_description = params.HOST_DESCRIPTION['description']
               
            script_obj["metadata"]["host"] = {
                    "name": host_name,
                    "description": host_description,
                    "gender":host_gender,
                }
            
        script_obj["metadata"]["host"]['name_agent'] = to_valid_variable(script_obj["metadata"]["host"]['name'])

        for k in range(len(script_obj["metadata"]["guests"])):
            script_obj["metadata"]["guests"][k]["name_agent"] = to_valid_variable(script_obj["metadata"]["guests"][k]["name"])

        script_obj["metadata"]["participants"] = []
        script_obj["metadata"]["participants"].append({
            "name": script_obj["metadata"]["host"]['name'],
            "gender": script_obj["metadata"]["host"]['gender'],
            "name_agent": script_obj["metadata"]["host"]['name_agent'],
            "role": "host",
            "description": script_obj["metadata"]["host"]["description"]
        })

        for guest in script_obj["metadata"]["guests"]:
            script_obj["metadata"]["participants"].append({
                "name": guest['name'],
                "name_agent": guest['name_agent'],
                "gender": guest['gender'],
                "role": "guest",
                "description": guest["description"]
            })

        script_obj["metadata"]['target_length_min']=params.TARGET_LENGTH_MINUTES
        
        # Save metadata
        save_metadata(script_obj)
        
        return script_obj
    except Exception as e:
        error_msg = f"Error generating podcast metadata: {e}"
        print(error_msg)
        raise Exception(error_msg)

def save_metadata(script_obj):
    """Save the podcast metadata to SCRIPT_OUTPUT_FILE."""
    # Validate required fields
    if not script_obj.get("metadata"):
        error_msg = "Missing podcast_metadata in script object"
        print(error_msg)
        raise Exception(error_msg)

    # Initialize output data structure
    output = {}
    
    # Check if the file exists and load its content
    try:
        if os.path.exists(params.SCRIPT_OUTPUT_FILE):
            with open(params.SCRIPT_OUTPUT_FILE, 'r', encoding='utf-8') as file:
                output = json.load(file)
    except Exception as e:
        print(f"Warning: Could not load existing file {params.SCRIPT_OUTPUT_FILE}: {e}")
        # Start fresh if there's an error loading the file
        output = {}
    
    # Update metadata
    output["metadata"] = script_obj["metadata"]

    # Update stage status
    if "stages" not in output:
        output["stages"] = {}
    
    output["stages"]["metadata"] = {
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }
    
    # Update last_updated timestamp
    output["last_updated"] = datetime.now().isoformat()
    
    # Write to file
    try:
        with open(params.SCRIPT_OUTPUT_FILE, 'w', encoding='utf-8') as file:
            json.dump(output, file, indent=2)
    except Exception as e:
        error_msg = f"Error saving metadata: {e}"
        print(error_msg)
        raise Exception(error_msg)

    print(f"Metadata saved to {params.SCRIPT_OUTPUT_FILE}")
    return params.SCRIPT_OUTPUT_FILE

def load_metadata():
    """Load podcast metadata from SCRIPT_OUTPUT_FILE."""
    script_obj = {}
    
    try:
        # First try to load from SCRIPT_OUTPUT_FILE
        if os.path.exists(params.SCRIPT_OUTPUT_FILE):
            with open(params.SCRIPT_OUTPUT_FILE, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
                # Check if metadata is present in SCRIPT_OUTPUT_FILE
                if "metadata" in data:

                    script_obj["metadata"] = data["metadata"]

                    # Validate that we have at least one participant
                    if not script_obj["metadata"]["participants"]:
                        raise Exception(f"No participants found in {params.SCRIPT_OUTPUT_FILE}")
                    
                    # Validate that we have a host
                    if not any(p.get("role") == "host" for p in script_obj["metadata"]["participants"]):
                        raise Exception(f"No host found in participants list in {params.SCRIPT_OUTPUT_FILE}")
                    
                    print(f"Metadata loaded from {params.SCRIPT_OUTPUT_FILE}")
                    return script_obj
        
        # If not found in SCRIPT_OUTPUT_FILE, try the original metadata file for backward compatibility
        metadata_file = params.META_OUTPUT_FILE
        with open(metadata_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Validate required fields
            if "metadata" not in data:
                raise Exception(f"Missing 'metadata' field in {metadata_file}")

            script_obj["metadata"] = data["metadata"]

            # Validate that we have at least one participant
            if not script_obj["metadata"]["participants"]:
                raise Exception(f"No participants found in {metadata_file}")
            
            # Validate that we have a host
            if not any(p.get("role") == "host" for p in script_obj["metadata"]["participants"]):
                raise Exception(f"No host found in participants list in {metadata_file}")
            
            print(f"Metadata loaded from {metadata_file}")
            
            # Save to SCRIPT_OUTPUT_FILE for future use
            update_stage_status("metadata", "completed")
            
            return script_obj
    except Exception as e:
        error_msg = f"Error loading metadata: {e}"
        print(error_msg)
        raise Exception(error_msg)

def clean_podcast_dialogue(script_obj):
    """Process the script to add speech styles and finalize timestamps."""
    try:
        # Validate input
        if not script_obj.get("script"):
            raise Exception("Missing script in script object")
        
        # Get AI client
        ai_client = get_ai_client()
        
        # Prepare the raw script for processing
        clean_script = []
        for i, message in enumerate(script_obj["script"]):
            # Validate message data
            if "speaker" not in message:
                raise Exception(f"Message {i+1} missing speaker field")
            
            if "text" not in message:
                raise Exception(f"Message {i+1} missing text field")
                
            clean_script.append({'speaker': message['speaker'], 'text': message['text'], 'style': '???'})

        N_raw = len(clean_script)
        raw_script = json.dumps(clean_script, indent=2)
        
        # Create messages for the LLM call
        messages = [
            {
                "role": "system", 
                "content": "You are a professional podcast script editor that enhances raw podcast scripts with improved utterances and appropriate speech styles. Aim is to produce a polished and production-ready podcast script."
            },
            {
                "role": "user", 
                "content": f"""
The podcast is in {params.LANGUAGE} language.

Please analyze this raw podcast script and enhance it by:

1. Verifying that utterances don't contain any non-verbal texts or cues that are NOT meant to be spoken out loud. The final utterances must only contain spoken texts tokens.                
2. Fix any grammar and discontinuity issues and typos, but do not modify original meaning of utterances.
3. Try to make utterance appear as realistic, relaxed podcast style discussion with valid dialogue between real people with emotions. Avoid long and boring monologues.
4. Replace ???'s with detailed speech style descriptions for each utterance to guide text-to-speech model in speech generation. Keep descriptions consistent for each person, while adapting to discussion.

Speech style descriptions can contain following elements:
    -Emotional range
    -Intonation
    -Impressions
    -Speed of speech
    -Tone                                                

** Examples of speech style descriptions **:      
"High-energy and enthusiastic voice, projecting enthusiasm and strong motivation"
"Short, punchy sentences with strategic pauses to maintain excitement and clarity" 
"Fast-paced and dynamic voice, with rising intonation to build momentum and keep engagement high"
"Positive, energetic, and empowering tone, creating an atmosphere of encouragement and achievement"
"Friendly and reassuring tone, creating a calm atmosphere and making the listener feel confident and comfortable, maintaining a natural and conversational flow."
"Fast paced speech with hints of frustration in to make own point clear."

Think which style is suitable for each utterance so that it makes sense in the discussion and sounds natural.

Below is the raw script that your must update:

<raw_script>
{raw_script}
</raw_script>

Return the refined and finalized script as a JSON array with the same exact structure with updated "style" fields.
Format your response as a code block with ONLY the JSON array.
""".strip()
            }
        ]
        
        try:
            total_tokens = count_string_tokens(prompt=messages[0]["content"] + messages[1]["content"], model=params.FINALIZING_LLM if not(params.USE_LOCAL_LLM) else 'azure_ai/Llama-3.3-70B-Instruct')
            print(f'\n...[The raw script has approximately {total_tokens} tokens]\n')
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}")
        
        # Make the API call
        if params.USE_LOCAL_LLM:
            # For local LLM, use OpenAI client directly
            client = openai.OpenAI(
                api_key=params.LOCAL_LLM_API_KEY,
                base_url=params.LOCAL_LLM_BASE_URL
            )
            response = client.chat.completions.create(
                model=params.LOCAL_LLM_MODEL,
                messages=messages,
                temperature=0.5
            )
        else:
            # Use aisuite for OpenAI/Anthropic
            try:
                response = completion(model=params.FINALIZING_LLM, messages=messages,temperature=0.5)
                # response = ai_client.chat.completions.create(
                #     model=get_aisuite_model_name(params.FINALIZING_LLM),
                #     messages=messages,
                #     temperature=0.3
                # )
            except Exception as e:
                print(f"Warning: Error with primary model, trying without temperature: {e}")
                response = completion(model=params.FINALIZING_LLM, messages=messages,max_completion_tokens=20000)
                # response = ai_client.chat.completions.create(
                #     model=get_aisuite_model_name(params.FINALIZING_LLM),
                #     messages=messages,
                # )
        
        # Extract content from the response
        content = response.choices[0].message.content
        
        # Extract JSON from the response
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'(\[.*\])', content, re.DOTALL)
            if not json_match:
                raise Exception("Failed to extract JSON array from LLM response")
            json_str = json_match.group(1)
        
        # Load the updated script
        updated_script = json.loads(json_str)
        
        # Validate the updated script
        if not updated_script:
            raise Exception("Empty script returned from LLM")
        
        N_clean = len(updated_script)
        if N_clean != N_raw:
            print(f"!!!! Number of messages changed during finalization from {N_raw} to {N_clean} !!!")
            # If the number of messages changed, we need to validate that we still have all required fields
            for i, entry in enumerate(updated_script):
                if "speaker" not in entry:
                    raise Exception(f"Updated message {i+1} missing speaker field")
                
                if "text" not in entry:
                    raise Exception(f"Updated message {i+1} missing text field")
                    
                if "style" not in entry:
                    raise Exception(f"Updated message {i+1} missing style field")
        
        final_script = []
        # Recalculate timestamps based on the new utterance lengths
        current_time = 0
        for entry in updated_script:
            # Estimate the duration of the updated utterance
            text = entry["text"].strip()
            duration = estimate_duration(text)
            
            # Update timestamps
            start_time = current_time
            end_time = current_time + duration
            
            final_script.append({
                'speaker': entry['speaker'],
                'text': text,
                'style': entry['style'],
                "start_time": start_time,
                "end_time": end_time,
                'duration': duration
            })

            # Move to the next utterance
            current_time += duration
        
        # Update the total duration
        script_obj["metadata"]["total_duration"] = current_time
        
        # Update the script
        script_obj["script"] = final_script
        
        # Save the final script
        save_script(script_obj, stage='final_script')
        
        return script_obj
    except Exception as e:
        error_msg = f"Error cleaning podcast dialogue: {e}"
        print(error_msg)
        raise Exception(error_msg)

def get_audio_descriptions_with_LLM(script_obj):

    max_index = len(script_obj['script']) - 1

    script = "\n".join(
        [('%i: ' % i) + x['speaker'] + ': ' + x['text'].replace('\n', ' ') for i, x in enumerate(script_obj['script'])])

    prompt = f'''
        You are an expert audio designer and composer. Your task is to design a short intro and intermediate music for a podcasts. Given a podcast script your task is to. Aim is to enhance listener experience and provide a comprehensive, production-ready podcast.
    
        Here is the podcast script:
        <podcast_script>
        {script}
        </podcast_script>
    
        ### Task
    
        You must design suitable 10s intro, outro and intermediate music pieces for this podcast. You write your designs as prompts for a music-generating AI model. For intermediate music piece, you must decide a proper location suitable for the podcast.
    
        ### Instructions
    
        Follow these guidelines:
        - All music are instrumental, with no vocals
        - Write detailed prompts of each of the three music pieces, including style, mood, tempo in BPM and used instruments
        - All three music should have similar overall style
        - Intermediate music piece is located approximately at the midpoint of the podcast in a suitable location that feels a natural break, if such exists. If podcast is very short or there is no suitable breakpoint for music, you indicate this in the "location" field.
    
        ### Output:
    
        Format your response as a valid JSON object in the following format. Locations to be completed are marked with parentheses [...].
    
        ```json
        [
            {{
                "type": "intro",
                "prompt": [detailed prompt of instrumental music style, mood, tempo, used instruments and other relevant music information]
            }},
            {{
                "type": "intermediate",
                "location": [Python index of the script utterance which music follows between 0 and {max_index}. If no suitable pause location is available, set value -1],
                "prompt": [detailed prompt of instrumental music style, mood, tempo in BPM, used instruments and other relevant music information]
            }},
            {{
                "type": "outro",
                "prompt": [detailed prompt of instrumental music style, mood, tempo in BPM, used instruments and other relevant music information]
            }}
        ]'''.strip()

    messages = [{
        "role": "user",
        "content": prompt
    }]

    ai_client = get_ai_client()

    try:
        # Make the API call
        if params.USE_LOCAL_LLM:
            # For local LLM, use OpenAI client directly
            client = openai.OpenAI(
                api_key=params.LOCAL_LLM_API_KEY,
                base_url=params.LOCAL_LLM_BASE_URL
            )
            response = client.chat.completions.create(
                model=params.LOCAL_LLM_MODEL,
                messages=messages,
                temperature=0.3
            )
        else:
            # Use aisuite for OpenAI/Anthropic
            response = completion(model=params.AUDIO_LLM, messages=messages, temperature=0.3)
            # response = ai_client.chat.completions.create(
            #     model=get_aisuite_model_name(params.AUDIO_LLM),
            #     messages=messages,
            #     temperature=0.3
            # )

        # Extract content from the response
        content = response.choices[0].message.content

        # Extract JSON from the response
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if not json_match:
                raise Exception("Failed to extract JSON from LLM response")
            json_str = json_match.group(1)

        # Load the voice selections from LLM
        audio_prompts = json.loads(json_str)

    except Exception as e:
        error_msg = f"Error designing voice prompt LLM: {e}"
        print(error_msg)
        raise Exception(error_msg)

    if not (1 < len(audio_prompts) < 4):
        raise Exception("Audio prompts must have 2-3 items")
    if len(audio_prompts) == 3 and not (2 < audio_prompts[1]['location'] < max_index - 2):
        audio_prompts = [audio_prompts[0]] + [audio_prompts[2]]
    if len(audio_prompts) == 2:
        assert audio_prompts[0]['type'] == 'intro' and audio_prompts[1]['type'] == 'outro'

    # Example placeholder data - in a real implementation, this would be generated based on content analysis
    script_obj["audio_ambient"] = audio_prompts
    # Save the ambient audio information
    save_script(script_obj, stage='audio_ambient', status='partial')

    return script_obj

def add_ambient_audio(script_obj):

    print("Adding ambient audio to the podcast")

    # Initialize ambient audio data if not present
    if "audio_ambient" not in script_obj:

        script_obj = get_audio_descriptions_with_LLM(script_obj)

    output_path = params.MUSIC_OUTPUT_PATH + '_' + params.MUSIC_LLM
    os.makedirs(output_path, exist_ok=True)

    if 'replicate' in params.MUSIC_LLM:

        for k,prompt in enumerate(script_obj["audio_ambient"]):

            input = {
                "prompt": prompt["prompt"],
                "model_version": "stereo-large",
                "output_format": "wav",
                "normalization_strategy": "rms",
                "duration": params.INTERMEDIATE_MUSIC_LENGTH if 'intermediate' in prompt['type'] else params.INTRO_OUTRO_MUSIC_LENGTH
            }

            output = replicate.run(
                "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb",
                input=input
            )

            filename = f"music_{prompt['type']}.wav"
            music_file_path = os.path.join(output_path, filename)

            with open(music_file_path, "wb") as file:
                file.write(output.read())

            script_obj["audio_ambient"][k]['audio_file'] = music_file_path

    elif 'elevenlabs' in params.MUSIC_LLM:

        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        client = ElevenLabs(api_key=elevenlabs_api_key)

        for k,prompt in enumerate(script_obj["audio_ambient"]):

            audio_bytes = client.text_to_sound_effects.convert(
                text=prompt["prompt"],
                duration_seconds = params.INTERMEDIATE_MUSIC_LENGTH if 'intermediate' in prompt['type'] else params.INTRO_OUTRO_MUSIC_LENGTH,
                output_format="pcm_24000")

            filename = f"music_{prompt['type']}.wav"
            music_file_path = os.path.join(output_path, filename)

            with open(music_file_path.replace('.wav', '.pcm'), "wb") as f:
                for chunk in audio_bytes:
                    f.write(chunk)

            # Read raw PCM data
            with open(music_file_path.replace('.wav', '.pcm'), "rb") as pcm_file:
                pcm_data = pcm_file.read()

            # Write WAV file
            with wave.open(music_file_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(pcm_data)

            script_obj["audio_ambient"][k]['audio_file'] = music_file_path

    new_music_file_path = get_faded_wav(script_obj["audio_ambient"][0]['audio_file'],start=1,end=2.5)
    script_obj["audio_ambient"][0]['audio_file'] = new_music_file_path

    new_music_file_path = get_faded_wav(script_obj["audio_ambient"][-1]['audio_file'],start=2.5,end=1)
    script_obj["audio_ambient"][-1]['audio_file'] = new_music_file_path

    if len(script_obj["audio_ambient"])==3:
        new_music_file_path = get_faded_wav(script_obj["audio_ambient"][1]['audio_file'], start=1.5, end=1.5)
        script_obj["audio_ambient"][1]['audio_file'] = new_music_file_path

    # Save the ambient audio information
    save_script(script_obj, stage='audio_ambient')

    return script_obj

def save_script(script_obj, stage, output_file=None,status='completed'):
    """Save the podcast script to SCRIPT_OUTPUT_FILE with stage tracking.
    
    Args:
        script_obj: The script object containing the podcast script
        stage: Processing stage identifier (one of VALID_STAGES)
        output_file: Optional custom output file path (defaults to SCRIPT_OUTPUT_FILE)
    """
    # Default output file
    if output_file is None:
        output_file = params.SCRIPT_OUTPUT_FILE
    
    # Initialize output data structure
    output = {}
    
    # Check if the file exists and load its content
    try:
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as file:
                output = json.load(file)
    except Exception as e:
        print(f"Warning: Could not load existing file {output_file}: {e}")
        # Start fresh if there's an error loading the file
        output = {}
    
    # Update basic metadata
    output["metadata"]={}
    output["metadata"]["title"] = script_obj["metadata"].get("title", "")
    output["metadata"]["description"] = script_obj["metadata"].get("description", "")
    output["metadata"]["language"] = params.LANGUAGE
    output["metadata"]["total_duration"] = script_obj["metadata"]["total_duration"]
    output["metadata"]["participants"] = script_obj["metadata"]["participants"]
    
    # Store script data based on stage
    if stage == 'raw_script':
        output[stage] = script_obj["script"]
    elif stage == 'final_script':
        output[stage] = script_obj["script"]
    elif stage == 'audio_speech':
        # Store voice selections if available
        output["audio_info"] = {}
        for k in script_obj["audio_info"].keys():
            output["audio_info"][k] = script_obj["audio_info"][k]
        output[stage] = script_obj["script"]
    elif stage == 'audio_ambient':
        # Store ambient audio information
        output[stage] = script_obj[stage]
    
    # Update stage status directly in the output object
    if "stages" not in output:
        output["stages"] = {}
    
    output["stages"][stage] = {
        "status": status,
        "timestamp": datetime.now().isoformat()
    }
    
    # Update last_updated timestamp
    output["last_updated"] = datetime.now().isoformat()
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=2)
    
    print(f"Script saved to {output_file} (stage: {stage})")
    return output_file

def load_script(stage, input_file=None):
    """Load a previously saved podcast script from SCRIPT_OUTPUT_FILE.
    
    Args:
        stage: The stage to load (one of VALID_STAGES)
        input_file: Optional custom input file path (defaults to SCRIPT_OUTPUT_FILE)
    
    Returns:
        dict: Script object with loaded data
    """
    # Initialize script object
    script_obj = {}
    
    # Default input file
    if input_file is None:
        input_file = params.SCRIPT_OUTPUT_FILE
    
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Load the script based on stage
            if stage in data:
                # If the stage data exists directly, use it
                script_obj["script"] = data[stage]
            elif stage == "final_script" and "styled" in data:
                # For backward compatibility
                script_obj["script"] = data["styled"]
            elif "script" in data:
                # Fallback to the main script field
                script_obj["script"] = data["script"]
            else:
                raise Exception(f"No script data found for stage '{stage}' in {input_file}")

            if "metadata" in data:
                script_obj["metadata"] = data["metadata"]

            # Load voice selections if available
            if "audio_info" in data and "voice_selections" in data["audio_info"]:
                script_obj["voice_selections"] = data["audio_info"]["voice_selections"]

            # Load stage-specific data
            if "audio_ambient" in data:
                script_obj["audio_ambient"] = data["audio_ambient"]

            # Load stage-specific data
            if "audio_speech" in data:
                script_obj["audio_speech"] = data["audio_speech"]
            
            print(f"Script (stage: {stage}) loaded from {input_file}")
            print(f"Loaded {len(script_obj['script'])} messages with total duration of {script_obj['metadata']['total_duration']} seconds")
            return script_obj
    except Exception as e:
        error_msg = f"Error loading script: {e}"
        print(error_msg)
        raise Exception(error_msg)

def select_voices_with_llm(participants_info, available_voices):
    """Use LLM to select appropriate voices for participants based on their characteristics.
    
    Args:
        participants_info: List of dictionaries containing participant information
        available_voices: Dictionary mapping voice IDs to voice descriptions
        ai_client: Optional AI client for making LLM calls
        
    Returns:
        dict: Mapping of participant names to selected voices
    """
    
    # Create messages for the LLM call
    messages = [
        {
            "role": "system", 
            "content": "You are an audio production assistant that selects appropriate voices for podcast participants."
        },
        {
            "role": "user", 
            "content": f"""
Based on the following participant information, select the most appropriate voice for each participant from the available options.

Participants:
{json.dumps(participants_info, indent=2)}

Available Voices:
{json.dumps(available_voices, indent=2)}

For each participant, select ONE voice that best matches their characteristics (gender, personality, role). Each participant must have DIFFERENT voice, you cannot use same voice for two persons!.

Format your response as a JSON object with participant names as keys and selected voice names as values:
{{
    "Participant Name 1": "voice_name 1",
    "Participant Name 2": "voice_name 2",
    ...
}}

Return ONLY the JSON object, no additional text.""".strip()
        }
    ]

    ai_client = get_ai_client()
    
    try:
        # Make the API call
        if params.USE_LOCAL_LLM:
            # For local LLM, use OpenAI client directly
            client = openai.OpenAI(
                api_key=params.LOCAL_LLM_API_KEY,
                base_url=params.LOCAL_LLM_BASE_URL
            )
            response = client.chat.completions.create(
                model=params.LOCAL_LLM_MODEL,
                messages=messages,
                temperature=0.3
            )
        else:
            # Use aisuite for OpenAI/Anthropic
            response = completion(model=params.AUDIO_LLM, messages=messages, temperature=0.3)
            # response = ai_client.chat.completions.create(
            #     model=get_aisuite_model_name(params.AUDIO_LLM),
            #     messages=messages,
            #     temperature=0.3
            # )
        
        # Extract content from the response
        content = response.choices[0].message.content
        
        # Extract JSON from the response
        json_match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if not json_match:
                raise Exception("Failed to extract JSON from LLM response")
            json_str = json_match.group(1)
        
        # Load the voice selections from LLM
        llm_voice_selections = json.loads(json_str)
        
        # Validate voice selections
        if not llm_voice_selections:
            raise Exception("Empty voice selections returned from LLM")
            
        # Check that all participants have a voice assigned
        for participant in participants_info:
            if participant["name"] not in llm_voice_selections:
                raise Exception(f"No voice assigned for participant: {participant['name']}")
                
        # Check that all assigned voices are valid
        for name, voice in llm_voice_selections.items():
            if voice not in available_voices:
                raise Exception(f"Invalid voice '{voice}' assigned to participant '{name}'")
        
        return llm_voice_selections
    except Exception as e:
        error_msg = f"Error selecting voices with LLM: {e}"
        print(error_msg)
        raise Exception(error_msg)

def select_voices_for_participants_openai(script_obj, voice_config=None):
    """Select appropriate OpenAI voices for each participant based on their characteristics.
    
    Args:
        script_obj: The script object containing participant information
        voice_config: Optional dictionary mapping participant names to voice preferences
        
    Returns:
        dict: Mapping of participant names to selected voices
    """

    # Get AI client
    ai_client = get_ai_client()
    
    # Prepare participant information for the LLM
    participants_info = []
    for participant in script_obj["metadata"]["participants"]:
        # Validate participant data
        if not participant.get("name_agent"):
            error_msg = "Participant missing name"
            print(error_msg)
            raise Exception(error_msg)
        if not participant.get("role"):
            error_msg = "Participant missing role"
            print(error_msg)
            raise Exception(error_msg)
        if not participant.get("description"):
            error_msg = "Participant missing description"
            print(error_msg)
            raise Exception(error_msg)
            
        participants_info.append({
            "name": participant["name_agent"],"description": participant["gender"] + '. ' + participant["description"]
        })

    voice_descriptions = {
        "alloy": "A warm and engaging female voice",
        "echo": "A strong and expressive male voice",
        "fable": "A storytelling female voice with a dramatic flair",
        "onyx": "A rich and deep male voice",
        "nova": "A bright and cheerful female voice",
        "shimmer": "A soft and soothing female voice",
        "coral": "A light and airy female voice",
        "verse": "A young and clear male voice",
        "ballad": "A high male voice with slight british tone",
        "ash": "A strong and low male voice",
        "sage": "A young and bright female voice"
    }

    # If voice_config is provided, use it for the specified participants
    if voice_config:
        # Start with the provided configuration
        voice_selections = {}
        for participant in participants_info:
            if participant["name"] in voice_config:
                voice_selections[participant["name"]] = voice_config[participant["name"]]
                assert voice_config[participant["name"]] in voice_descriptions
        
        # If all participants have voices assigned, return the selections
        if len(voice_selections) == len(participants_info):
            return voice_selections
    else:
        voice_selections = select_voices_with_llm(participants_info,voice_descriptions)

    return voice_selections
    
def select_voices_for_participants_elevenlabs(script_obj, voice_config=None):
    """Select appropriate ElevenLabs voices for each participant based on their characteristics.
    
    Args:
        script_obj: The script object containing participant information
        voice_config: Optional dictionary mapping participant names to voice preferences
        
    Returns:
        dict: Mapping of participant names to selected voices
    """
    # Prepare participant information for the LLM
    participants_info = []
    for participant in script_obj["metadata"]["participants"]:
        # Validate participant data
        if not participant.get("name_agent"):
            error_msg = "Participant missing name"
            print(error_msg)
            raise Exception(error_msg)
        if not participant.get("role"):
            error_msg = "Participant missing role"
            print(error_msg)
            raise Exception(error_msg)
        if not participant.get("description"):
            error_msg = "Participant missing description"
            print(error_msg)
            raise Exception(error_msg)

        participants_info.append({
            "name": participant["name_agent"], "description": participant["gender"] + '. ' + participant["description"]
        })

    elevenlabs_voices = {
        "Anna": {"description":'female, clear and professional host voice','id':"OtosILLvrVZJYq1plzeO"},  #
        "Janne": {"description":'male, suitable for technical and researcher talk',"id":"JvN11VCYqsU2kmIoLprq"},       #
        "Henry": {"description":'male,older deep and authoritative tone','id':"Dkbbg7k9Ir9TNzn5GYLp"}, #
        "Aurora":{"description":'female, young and inspiring voice','id':"YSabzCJMvEHDduIDMdwV"},
        "Lumi": {"description": 'female, young slow-paced', 'id': "YSabzCJMvEHDduIDMdwV"},
        "Ville": {"description": 'male, young 30s fresh voice', 'id': "YSabzCJMvEHDduIDMdwV"},
        "Cristoffer": {"description": 'male, professional male presenter voice', 'id': "3OArekHEkHv5XvmZirVD"}
    }

    # If voice_config is provided, use it for the specified participants
    if voice_config:
        # Start with the provided configuration
        voice_selections = {}
        for participant in script_obj["metadata"]["participants"]:
            if participant["name_agent"] in voice_config:
                voice_selections[participant["name_agent"]] = voice_config[participant["name_agent"]]
        
        # If all participants have voices assigned, return the selections
        if len(voice_selections) == len(script_obj["participants"]):
            return voice_selections
    else:
        voice_descriptions = {k:v["description"] for k,v in elevenlabs_voices.items()}
        voice_selections = select_voices_with_llm(participants_info, voice_descriptions)
        voice_selections = {k:elevenlabs_voices[v]['id'] for k,v in voice_selections.items()}
    
    return voice_selections

def select_voices_for_participants(script_obj):
    """Select appropriate voices for each participant based on their characteristics and the TTS engine.
    
    Args:
        script_obj: The script object containing participant information
        
    Returns:
        dict: Mapping of participant names to selected voices
    """
    # Get TTS engine from script_obj or global config
    tts_engine = script_obj.get("audio_info", {}).get("TTS_engine", params.TEXT_TO_SPEECH)
    
    # Get voice config if provided
    voice_config = params.TTS_VOICE_CONFIG
    
    if 'gpt' in tts_engine:
        return select_voices_for_participants_openai(script_obj, voice_config)
    else:
        return select_voices_for_participants_elevenlabs(script_obj, voice_config)

def get_dialogue_audio(script_obj):
    """Generate audio files for each utterance in the podcast script."""

    # Validate input
    if not script_obj.get("script"):
        raise Exception("Missing script in script object")
    
    if not script_obj["metadata"].get("participants"):
        raise Exception("Missing participants in script object")

    if 'gpt' in params.TEXT_TO_SPEECH:
        client = OpenAI()
        model_str = '_openai'
    else:
        # Initialize the ElevenLabs client
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        client = ElevenLabs(api_key=elevenlabs_api_key)
        model_str = '_elevenlabs'

    # Ensure output directory exists
    output_path = params.SPEECH_OUTPUT_PATH + model_str
    os.makedirs(output_path, exist_ok=True)
    
    if "audio_info" not in script_obj:
        # Select voices for participants
        voice_selections = select_voices_for_participants(script_obj)
        script_obj["audio_info"] = {"voice_selections":voice_selections}
        save_script(script_obj, stage='audio_speech',status='partial') 
    else:
        voice_selections = script_obj["audio_info"]["voice_selections"]
       
    # Set TTS engine
    TTS_engine = params.TEXT_TO_SPEECH
    script_obj["audio_info"]['TTS_engine'] = TTS_engine
    
    total_duration = 0
    audio_speech = []

    # Process each utterance
    for i, utterance in enumerate(script_obj["script"], start=1):
        # Validate utterance data
        if "speaker" not in utterance:
            raise Exception(f"Utterance {i+1} missing speaker field")
        
        if "text" not in utterance:
            raise Exception(f"Utterance {i+1} missing text field")
        
        if "style" not in utterance:
            raise Exception(f"Utterance {i+1} missing style field")
        
        # Get the participant name from the speaker agent name
        participant_name = next(
            (p["name_agent"] for p in script_obj["metadata"]["participants"] if p["name_agent"] == utterance["speaker"]),
            None
        )
        
        if not participant_name:
            raise Exception(f"No participant found with name_agent '{utterance['speaker']}'")
        
        if participant_name not in voice_selections:
            raise Exception(f"No voice selected for participant '{participant_name}'")
        
        voice = voice_selections[participant_name]
        
        # Generate filename
        filename = f"utterance_{i}_{utterance['speaker']}.wav"
        speech_file_path = os.path.join(output_path, filename)
        
        utterance["voice"] = voice
        
        try_count = 0
        success = 0

        if 'gpt' in params.TEXT_TO_SPEECH:
            # Initialize OpenAI client

            while try_count < 3:
                try:                
                    # Generate audio with OpenAI
                    with client.audio.speech.with_streaming_response.create(
                        model=params.TEXT_TO_SPEECH,
                        voice=voice,
                        input=utterance["text"],
                        instructions=utterance["style"],
                    ) as response:
                        response.stream_to_file(speech_file_path)
                        success = 1
                        break
                    
                except Exception as e:                
                    error_msg = f"Error generating audio with OpenAI: {e}"
                    print(error_msg)
                    try_count += 1
                    
        else:
            # Get the ElevenLabs voice ID

            while try_count < 3:
                try:
                    # Generate audio with ElevenLabs
                    audio_bytes = client.text_to_speech.convert(
                        voice_id=voice,
                        output_format="pcm_24000",
                        text=utterance["text"],
                        model_id=params.TEXT_TO_SPEECH,
                        voice_settings={
                            "stability": 0.5,
                            "similarity_boost": 0.80,
                            'use_speaker_boost':True,
                            "speed": 1.07,
                            'style':0.3
                        }
                    )

                    with open(speech_file_path.replace('.wav','.pcm'), "wb") as f:
                        for chunk in audio_bytes:
                            f.write(chunk)

                    # Read raw PCM data
                    with open(speech_file_path.replace('.wav','.pcm'), "rb") as pcm_file:
                        pcm_data = pcm_file.read()

                    # Write WAV file
                    with wave.open(speech_file_path, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(24000)
                        wav_file.writeframes(pcm_data)

                    success = 1

                    break
                    
                except Exception as e:
                    error_msg = f"Error generating audio with ElevenLabs: {e}"
                    print(error_msg)
                    try_count += 1

        if success == 0:
            raise Exception(f'Failed to generate audio with {params.TEXT_TO_SPEECH}!')
            
        # Add audio_file field to utterance
        utterance["audio_file"] = speech_file_path

        info = ffmpeg.probe(speech_file_path)

        # Extract the duration from the metadata and convert it to a float
        duration = float(info['format']['duration'])

        utterance['start_time'] = total_duration
        utterance['end_time'] = total_duration + duration
        utterance["duration"] = duration

        total_duration+=duration+params.UTTERANCE_GAP

        audio_speech.append(utterance)

        print(f"...generated audio for utterance {i} of {len(script_obj['script'])}, participant '{participant_name}', using voice '{voice}, duration so far {total_duration}s'")

    script_obj["script"] = audio_speech

    script_obj["metadata"]["total_duration"] = total_duration

    save_script(script_obj, stage='audio_speech')

    return script_obj

def construct_podcast_file(script_obj):

    files_to_merge=[]

    files_to_merge.append({'file':script_obj['audio_ambient'][0]['audio_file'],'text':'[MUSIC]'} )

    intermediate_audio=None
    if len(script_obj['audio_ambient'])==3:
        intermediate_audio = script_obj['audio_ambient'][1]

    for k,utterance in enumerate(script_obj["audio_speech"]):

        if intermediate_audio:
            if intermediate_audio['location']==k:
                files_to_merge.append({'file':intermediate_audio['audio_file'],'text':'[MUSIC]'})

        files_to_merge.append({'file':utterance['audio_file'],'text':utterance['text']})

    files_to_merge.append({'file':script_obj['audio_ambient'][-1]['audio_file'],'text':'[MUSIC]'})

    files_to_merge = [x['file'] for x in files_to_merge]

    final_audio = get_merged_audio(params.PODCAST_OUTPUT_FILE,files_to_merge,gap_lims=(0.70,1.25))

    print(f'Final audio saved as {final_audio}')

# def construct_podcast_video(script_obj):
#
#     transcript = []
#
#     for k,utterance in enumerate(script_obj["audio_speech"]):
#
#         generate_podcast_video(params.VIDEO_OUTPUT_FILE, bg_path, audio_path, transcript)

if __name__ == "__main__":
    # To use a local LLM, uncomment and modify these lines as needed:
    # USE_LOCAL_LLM = True
    # LOCAL_LLM_BASE_URL = "http://localhost:1234/v1"  # Update with your local LLM server URL
    # LOCAL_LLM_API_KEY = "lm-studio"                  # Update with your local LLM API key
    # LOCAL_LLM_MODEL = "deepthinkers-phi4"            # Update with your local model name
    # LOCAL_LLM_MAX_TOKENS = 500                       # Adjust as needed
    # LOCAL_LLM_TEMPERATURE = 0.5                      # Adjust as needed
    
    # Step 1: Generate or load podcast metadata
    if check_stage_status('metadata'):
        print("podcast metadata exist in SCRIPT_OUTPUT_FILE...")
    else:
        print("Generating new podcast metadata...")
        get_podcast_metadata()
    
    # Step 2: Generate or load podcast dialogue
    if 1 and can_proceed_to_stage('raw_script'):
        if 1 and check_stage_status('raw_script'):
            print("podcast raw script exist in SCRIPT_OUTPUT_FILE...")
            # No action needed, data is already in SCRIPT_OUTPUT_FILE
        else:
            print(f"Starting podcast simulation with {params.NUM_GUESTS} guests...")
            # Load metadata first
            script_obj = load_metadata()
            # Generate dialogue and save to file
            get_podcast_dialogue(script_obj,save_script)
            # Save the raw script

    # Step 3: Clean and finalize the dialogue
    if 1 and can_proceed_to_stage('final_script'):
        if 1 and check_stage_status('final_script'):
            print("Loading finalized script from SCRIPT_OUTPUT_FILE...")
            # No action needed, data is already in SCRIPT_OUTPUT_FILE
        else:
            print("Finalizing script with speech styles and timestamps...")
            # Load raw script
            script_obj = load_script('raw_script')
            # Clean and finalize
            clean_podcast_dialogue(script_obj)

    # Step 4: Generate audio for each utterance
    if 1 and can_proceed_to_stage('audio_speech'):
        if 1 and check_stage_status('audio_speech'):
            print("Audio speech generation already completed.")
            # No action needed, data is already in SCRIPT_OUTPUT_FILE
        else:
            print("Generating audio files for each utterance...")
            # Load final script
            script_obj = load_script('final_script')
            # Generate audio
            get_dialogue_audio(script_obj)
            # Save is handled within get_dialogue_audio

    # Step 5: Add ambient audio
    if 1 and can_proceed_to_stage('audio_ambient'):
        if 1 and check_stage_status('audio_ambient'):
            print("Ambient audio already added.")
            # No action needed, data is already in SCRIPT_OUTPUT_FILE
        else:
            print("Adding ambient audio to the podcast...")
            # Load audio speech script
            script_obj = load_script('final_script')
            # Add ambient audio
            add_ambient_audio(script_obj)

    if 1 and can_proceed_to_stage('finalize_podcast'):
        if 0 and os.path.exists(params.PODCAST_OUTPUT_FILE):
            print("Podcast file created.")
            # No action needed, data is already in SCRIPT_OUTPUT_FILE
        else:
            print("Combining podcast file...")
            # Load audio speech script
            script_obj = load_script('final_script')
            # Add ambient audio
            construct_podcast_file(script_obj)

    print(f"Podcast simulation complete!")
