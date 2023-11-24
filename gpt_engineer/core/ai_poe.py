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

import backoff
from robot_api.core import PoeAI
from robot_api.protocol import Message, HumanMessage, AIMessage, \
    SystemMessage, Models

from gpt_engineer.core.ai import AI

# setup logging
logger = logging.getLogger(__name__)


class PAI(AI):
    """
    AI class for interacting with POE's robot.

    Args:
        AI (_type_): AI interface
    """

    def __init__(self, model_name: str = "GPT-4", temperature=0.1, poe_endpoint: str = "https://api.poe.com/bot/"):
        self.temperature = temperature
        self.poe_endpoint = poe_endpoint
        self.model_name = check
        pass

    def _check_model_access_and_fallback(self, model_name) -> str:
        """
        Checks if the model is available, and if not, falls back to a default model.

        Args:
            model_name (str): The name of the model to check.

        Returns:
            str: The name of the model to use.
        """
        if Models.has_value(model_name):
            logger.warning(
                f"Model {model_name} is not available. Falling back to default model."
            )
            model_name = Models.GPT_3_5_TURBO.value
        return model_name
