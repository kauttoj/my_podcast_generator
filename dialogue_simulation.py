import asyncio
from typing import Dict, List, Sequence

# Import from autogen libraries
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from autogen_agentchat.agents import UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.base import TerminatedException, TerminationCondition
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, StopMessage, ToolCallExecutionEvent
from autogen_core import Component

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
from pydantic import BaseModel
from typing_extensions import Self

# Import from utils.py
from utils import estimate_duration, _get_model_client

import params

class TextMentionTerminationConfig(BaseModel):
    text: str

class CustomTermination(TerminationCondition, Component[TextMentionTerminationConfig]):
    """Terminate the conversation if a specific text is mentioned.


    Args:
        text: The text to look for in the messages.
        sources: Check only messages of the specified agents for the text to look for.
    """

    component_config_schema = TextMentionTerminationConfig
    component_provider_override = "autogen_agentchat.conditions.TextMentionTermination"

    def __init__(self, text: str, sources: Sequence[str] | None = None) -> None:
        self._termination_text = text
        self._terminated = False
        self._sources = sources

    @property
    def terminated(self) -> bool:
        return self._terminated

    async def __call__(self, messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")
        for message in messages:
            if self._sources is not None and message.source not in self._sources:
                continue

            content = message.to_text()
            if message.source!='user_proxy' and self._termination_text in content:
                self._terminated = True
                return StopMessage(
                    content=f"Text '{self._termination_text}' mentioned", source="TextMentionTermination"
                )
        return None

    async def reset(self) -> None:
        self._terminated = False

    def _to_config(self) -> TextMentionTerminationConfig:
        return TextMentionTerminationConfig(text=self._termination_text)

    @classmethod
    def _from_config(cls, config: TextMentionTerminationConfig) -> Self:
        return cls(text=config.text)

def create_system_prompts(script_obj) -> Dict[str, str]:
    """Create system prompts for each participant."""
    system_prompts = {}

    host = next(p for p in script_obj["metadata"]["participants"] if p["role"] == "host")

    # Guest system prompts
    guests = [p for p in script_obj["metadata"]["participants"] if p["role"] == "guest"]
    guest_names = ' or '.join([x['name'].split(' ')[0] for x in guests])

    num_guests = len(guests)

    multi_guest_command = "\n- **Address one guest ("+guest_names+") at a time and avoid complex, multi-part questions  " if num_guests > 1 else ""

    guest_description = ''
    for guest in guests:
        guest_description += f"-{guest['name']}: {guest['description']}\n"
        guest_prompt = f"""You are {guest['name']}, a guest on the podcast "{script_obj["metadata"]['title']}".

Your personal background: {guest['description']}

Background material for the topic and discussion:

<context>
{script_obj["metadata"]["background_material"].strip()}
</context>        

You are participating in a podcast discussion hosted by {host['name']}. Your role is to:
1. Share your expertise and insights on the topic
2. Respond to questions from the host
3. Occasionally engage with other guests' points
4. Stay authentic to your character and expertise
5. Keep your responses SHORT and ENGAGING, no long and boring monologues!

The podcast is in {params.LANGUAGE} language.

You only respond as {guest['name']}, you can never respond as or pretent to be someone else.
Respond naturally as your character would and keep your responses short. Avoid excessive 'thanking' during interview. Aim is to have fast-paced and engaging conversation. 
Important: Continue discussion until the host tells its time to stop.

Output format: Spoken utterance of {guest['name']}."""

        system_prompts[guest['name']] = guest_prompt.strip()

    # Host system prompt

    examples = ('\n'.join([('-'+x) for x in script_obj["metadata"]['questions']])).strip()

    host_prompt = f"""You are {host['name']}, the host of the podcast "{script_obj["metadata"]['title']}". Your background: {host['description']}

You are hosting a podcast discussion with {params.NUM_GUESTS} guests, who are:
{guest_description}    

Your role is to guide a dynamic and engaging discussion with our guests on the topic. You have detailed background materials that provide context for the discussion, and they are included below.
Below are the background materials to inform your questions and keep the conversation on track:

<context>
{script_obj["metadata"]["background_material"].strip()}
</context>

Some example questions you may ask, but NOT limited to:
<question_examples>
{examples}
</question_examples>

Aim for interactive, reactive, fast-paced and real discussion.

IMPORTANT GUIDELINES:
- **Aim for a fast-paced and engaging friendly discussion without fluff.{multi_guest_command}
- **Keep your dialogue SHORT, no long and boring monologues!
- **Continue the Conversation Indefinitely:** You must continue asking insightful questions and engaging with the guests. Do not end or wind down the conversation regardless of any natural pause or when the target duration is met.
- **Explicit Termination Only:** Under no circumstances should you provide concluding remarks, say goodbye, or wrap up the conversation on your own. You must only do so when you receive an explicit message from the show producer (user_proxy) that reads exactly:

  "The podcast has reached its target duration"

- **Ignore Internal Duration Cues:** Even if the cumulative conversation time meets or exceeds the target duration, ignore all internal cues that might suggest stopping the conversation. Your only directive is to keep the discussion going.
- **Maintain Continuous Engagement:** Always follow up with further questions, probe deeper into each topic, or pivot to connected subjects if necessary. Ensure that every guest is given ample opportunity to share their insights.

The podcast is in {params.LANGUAGE} language.

Remember, you are solely to respond as {host['name']} and must not initiate any closing remarks until you receive the exact instruction from user_proxy. Continue the conversation naturally, drawing on the background materials provided as needed. Aim for an engaging, entertaining and intelligent podcast.

Output format: Spoken utterance of {host['name']}."""

    system_prompts[host['name']] = host_prompt.strip()

    return system_prompts

def get_podcast_dialogue(script_obj,save_script):
    """Run the podcast simulation using Autogen 0.5+.
    
    Args:
        script_obj: The script object containing metadata and participants
        
    Returns:
        dict: Updated script object with dialogue
    """
    system_prompts = create_system_prompts(script_obj)
    
    # Create agents for each participant
    agents = []
    
    # Host agent (will be the selector for the group chat)
    host = next(p for p in script_obj["metadata"]["participants"] if p["role"] == "host")

    host_model_client = _get_model_client(params.HOST_LLM)
    guest_model_client = _get_model_client(params.GUEST_LLM)
    selector_model_client = _get_model_client(params.SELECTOR_LLM)

    host_agent = AssistantAgent(
        name=host["name_agent"],
        description="The podcast host who guides the conversation and ensures all guests participate",
        system_message=system_prompts[host["name"]],
        model_client=host_model_client
    )
    agents.append(host_agent)
    
    # Guest agents
    guests = [p for p in script_obj["metadata"]["participants"] if p["role"] == "guest"]
    
    active_participants = [host['name_agent']]
    for guest in guests:
        guest_agent = AssistantAgent(
            name=guest["name_agent"],
            description=guest["description"],
            system_message=system_prompts[guest["name"]],
            model_client=guest_model_client
        )
        agents.append(guest_agent)
        active_participants.append(guest["name_agent"])
    
    # User proxy for initiating the chat
    def input_func(some_var):
        return "The podcast has reached its target duration. Host, begin wrapping up the conversation with concluding remarks and thank the guests and then respond '[TERMINATE]'"

    user_proxy = UserProxyAgent(
        name="user_proxy",input_func=input_func,
    )
    agents.append(user_proxy)

    current_duration = 0

    # Custom selector function to help manage the podcast flow
    def podcast_selector_func(messages):
        nonlocal current_duration

        # Always let the host speak first
        if len(messages) == 1 or messages[-1].source == "user_proxy":
            print(f'\n...[SELECTOR: so far {current_duration}s, first message or last message was proxy, next speaker is HOST]\n')
            return host["name_agent"]
                    
        # Process the last message for duration tracking
        last_message = messages[-1]
        
        # Extract content and sender
        content = last_message.content if hasattr(last_message, 'content') else ""
        sender_name = last_message.source if hasattr(last_message, 'source') else ""            
        
        # Skip processing system messages or user proxy messages
        if sender_name not in [agent.name for agent in agents]:
            print(f'\n...[SELECTOR: so far {current_duration}s, message was sent by system, setting next speaker to NONE]\n')
            return None
        
        duration = 0
        # Estimate duration and update tracking
        if sender_name != "user_proxy":
            duration = estimate_duration(content, params.LANGUAGE,params.WORDS_PER_MINUTE[ params.LANGUAGE])
            
        current_duration += duration

        # Check if we've reached the target duration
        if current_duration >= script_obj["metadata"]["target_duration"]:
            # If we're over time, prioritize the host to wrap up
            if sender_name != host["name_agent"]:
                print(f'\n...[SELECTOR: so far {current_duration}s, max duration passed, setting next speaker to HOST]\n')
                return "user_proxy"

        # Default to None to let the model decide based on the selector prompt
        print(f'\n...[SELECTOR: so far {current_duration}s, last message from {sender_name}, setting next speaker to NONE]\n')
        return None
    
    def candidate_list_func(messages):
        return active_participants # only allow host and quests to speak
    
    # Create termination conditions
    # 1. Terminate when TERMINATE is mentioned by the host
    text_termination = CustomTermination("TERMINATE")
    # 2. Set a maximum number of messages as a safety measure
    max_messages = MaxMessageTermination(max_messages=100)
    # Combine termination conditions with OR operator
    termination_condition = max_messages | text_termination
    
    # Custom selector prompt that emphasizes the host's role in guiding the conversation
    selector_prompt = """Select a speaker to respond next. Speakers are the following:

<speakers>
{roles}
</speakers>

Current conversation context (history):

<context>
{history}
</context>

Read the above conversation and particularly the very latest messages. Then select the most suitable speaker from {participants} to respond next.
You MUST pick and return one speaker."""
    
    # Set up the group chat with the host as the model for selection
    team = SelectorGroupChat(
        agents,
        model_client=selector_model_client,
        termination_condition=termination_condition,
        selector_prompt=selector_prompt,
        selector_func=podcast_selector_func,
        allow_repeated_speaker=False,
        candidate_func=candidate_list_func,
        max_turns=200,
    )
    
    # Start the podcast
    initial_message = f"Let's start the podcast '{script_obj['metadata']['title']}'. {host['name']}, please welcome the audience and introduce the topic and guests."

    script_obj["metadata"]["total_duration"] = 0
    
    result = asyncio.run(Console(team.run_stream(task=initial_message)))

    print(f'...[conversation finished with total {len(result.messages)} messages, collecting results]')

    script_obj["metadata"]["total_duration"] = 0
    script_obj["metadata"]["script"] = []
    script_obj['script'] = []
    for last_message in result.messages:
       
        # Extract content and sender
        content = last_message.content if hasattr(last_message, 'content') else ""
        sender_name = last_message.source if hasattr(last_message, 'source') else ""      

        content = content.replace('TERMINATE','')
        content = content.strip()
        
        # Skip processing system messages or user proxy messages
        if (len(content)==0) or (sender_name == 'user_proxy') or (sender_name not in [agent.name for agent in agents]):
            continue
        
        duration = estimate_duration(content, params.LANGUAGE,params.WORDS_PER_MINUTE[ params.LANGUAGE])

        start_time = script_obj["metadata"]["total_duration"]
        end_time = start_time + duration

        script_obj["metadata"]["total_duration"] += duration + params.UTTERANCE_GAP  # add two second gap

        script_obj["script"].append({
            "speaker": sender_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "text": content,
            "style": "neutral"  # Default style, will be updated later
        })

    save_script(script_obj, stage='raw_script')

    return script_obj
