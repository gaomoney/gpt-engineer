"""
This module provides a implementation of the `AI` class,
which is used to interact with POE's robot.

Key Features:
- Integration with POE's robot.
- Token usage logging.
- Seamless fallback to default models in case the desired model is not available.
- Serialization and deserialization of POE message.


Class:
- PoeAI: Represents an AI that interacts with POE's robot.

Dependencies:
- robot-api: For interacting with POE's robot, API Layer.
- backoff: For retrying failed requests.
- typing: For type hinting.
"""

import logging
import os
import json
from typing import List, Optional, Union
from asyncio import run as asyncio_run
import backoff
import openai
from robot_api.core import PoeAI
from robot_api.protocol import Models
from robot_api import protocol
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    messages_to_dict,
    messages_from_dict
)
from langchain.callbacks.manager import (
    Callbacks
)
from gpt_engineer.core.ai import AI
from gpt_engineer.core.token_usage import TokenUsageLog

# Type hint for a chat message
Message = Union[AIMessage, HumanMessage, SystemMessage]

# setup logging
logger = logging.getLogger(__name__)


class PAI(AI):
    """
    AI class for interacting with POE's robot.

    Args:
        AI (_type_): AI interface
    """

    def __init__(self, model_name: str = "GPT-4",
                 temperature=0.1, poe_endpoint: str = "https://api.poe.com/bot/"):
        self.temperature = temperature
        self.poe_endpoint = poe_endpoint
        self.model_name = self._check_model_access_and_fallback(model_name)
        self.llm = self._create_chat_model()
        self.token_usage_log = TokenUsageLog(model_name)

        logger.debug(f"Using model {self.model_name}")

    def _check_model_access_and_fallback(self, model_name) -> str:
        """
        Checks if the model is available, and if not, falls back to a default model.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            str: The name of the model to use.
        """
        if not Models.has_value(model_name):
            if Models.has_value(model_name.upper()):
                return model_name.upper()
            logger.warning(
                f"Model {model_name} is not available. Falling back to default model."
            )
            model_name = Models.GPT_3_5_TURBO.value
        return model_name

    def start(self, system: str, user: str, step_name: str) -> List[Message]:
        """
        Start the conversation with a system message and a user message.

        Parameters
        ----------
        system : str
            The content of the system message.
        user : str
            The content of the user message.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The list of messages in the conversation.
        """
        messages: List[Message] = [
            SystemMessage(content=system),
            HumanMessage(content=user),
        ]
        return self.next(messages, step_name=step_name)

    def next(
        self,
        messages: List[Message],
        prompt: Optional[str] = None,
        *,
        step_name: str,
    ) -> List[Message]:
        """
        Advances the conversation by sending message history
        to LLM and updating with the response.

        Parameters
        ----------
        messages : List[Message]
            The list of messages in the conversation.
        prompt : Optional[str], optional
            The prompt to use, by default None.
        step_name : str
            The name of the step.

        Returns
        -------
        List[Message]
            The updated list of messages in the conversation.
        """
        """
        Advances the conversation by sending message history
        to LLM and updating with the response.
        """
        if prompt:
            messages.append(HumanMessage(content=prompt))
        logger.debug(f"Sending messages to LLM: {messages}")
        callbacks = [StreamingStdOutCallbackHandler()]
        response = self.backoff_inference(messages, callbacks)

        self.token_usage_log.update_log(
            messages=messages, answer=response.content, step_name=step_name
        )
        messages.append(response)
        logger.debug(f"Chat completion finished: {messages}")
        return messages

    @backoff.on_exception(
        backoff.expo, openai.error.RateLimitError, max_tries=7, max_time=45
    )
    def backoff_inference(self, messages, callbacks):
        """
        Perform inference using the language model while implementing an exponential backoff strategy.

        This function will retry the inference in case of a rate limit error from the OpenAI API.
        It uses an exponential backoff strategy, meaning the wait time between retries increases
        exponentially. The function will attempt to retry up to 7 times within a span of 45 seconds.

        Parameters
        ----------
        messages : List[Message]
            A list of chat messages which will be passed to the language model for processing.

        callbacks : List[Callable]
            A list of callback functions that are triggered after each inference. These functions
            can be used for logging, monitoring, or other auxiliary tasks.

        Returns
        -------
        Any
            The output from the language model after processing the provided messages.

        Raises
        ------
        openai.error.RateLimitError
            If the number of retries exceeds the maximum or if the rate limit persists beyond the
            allotted time, the function will ultimately raise a RateLimitError.

        Example
        -------
        >>> messages = [SystemMessage(content="Hello"), HumanMessage(content="How's the weather?")]
        >>> callbacks = [some_logging_callback]
        >>> response = backoff_inference(messages, callbacks)
        """
        return self.llm(messages, callbacks=callbacks)  # type: ignore

    def _create_chat_model(self) -> PoeAI:
        """
        Create a chat model with the specified model name and temperature.

        Parameters
        ----------
        model : str
            The name of the model to create.
        temperature : float
            The temperature to use for the model.

        Returns
        -------
        BaseChatModel
            The created chat model.
        """
        api_key = os.environ.get("POE_ACCESS_KEY")
        assert api_key is not None, "POE_ACCESS_KEY environment variable not set"
        poe_ai = PoeAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key)

        def sync_model(messages: List[Message], callbacks: List[Callbacks]) -> Message:
            return asyncio_run(model(poe_ai, messages, callbacks=callbacks))

        return sync_model

    @staticmethod
    def serialize_messages(messages: List[Message]) -> str:
        """
        Serialize a list of messages to a JSON string.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to serialize.

        Returns
        -------
        str
            The serialized messages as a JSON string.
        """
        return json.dumps(messages_to_dict(messages))

    @staticmethod
    def deserialize_messages(jsondictstr: str) -> List[Message]:
        """
        Deserialize a JSON string to a list of messages.

        Parameters
        ----------
        jsondictstr : str
            The JSON string to deserialize.

        Returns
        -------
        List[Message]
            The deserialized list of messages.
        """
        data = json.loads(jsondictstr)
        # Modify implicit is_chunk property to ALWAYS false
        # since Langchain's Message schema is stricter
        prevalidated_data = [
            {**item, "data": {**item["data"], "is_chunk": False}} for item in data
        ]
        return list(messages_from_dict(prevalidated_data))  # type: ignore


async def model(
        ai: PoeAI,
        messages: List[Message],
        callbacks: List[Callbacks] = None,
        **kwargs):
    poe_messages = []
    for message in messages:
        # if message.
        if message.type == "system":
            poe_messages.append(
                protocol.SystemMessage(content=message.content))
        if message.type == "human":
            poe_messages.append(protocol.HumanMessage(content=message.content))
        if message.type == "ai":
            poe_messages.append(protocol.AIMessage(content=message.content))
    logging.debug(f"Sending messages to POE LLM: {poe_messages}")
    outputs = []
    async for msg in ai.chat(poe_messages):
        msg_text = ""
        if msg.choices and len(msg.choices) > 0:
            msg_text = msg.choices[0].message.content
            outputs.append(msg_text)
            if callbacks:
                for callback in callbacks:
                    callback.on_llm_new_token(msg_text)
    return AIMessage(content="".join(outputs))
