import logging
from typing import List, Dict, Tuple, Any
from retry import retry
import json
import openai
import base64
import imghdr
import httpx
import asyncio
from openai import AsyncOpenAI, RateLimitError, APIError
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type

import clemcore.backends as backends
from clemcore.backends.utils import ensure_messages_format

logger = logging.getLogger(__name__)

NAME = "openai"


class OpenAI(backends.RemoteBackend):

    def _make_api_client(self):
        creds = backends.load_credentials(NAME)
        api_key = creds[NAME]["api_key"]
        organization = creds[NAME]["organisation"] if "organisation" in creds[NAME] else None
        return openai.OpenAI(api_key=api_key, organization=organization)

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get an OpenAI model instance based on a model specification.
        Args:
            model_spec: A ModelSpec instance specifying the model.
        Returns:
            An OpenAI model instance based on the passed model specification.
        """
        return OpenAIModel(self.client, model_spec)


class OpenAIModel(backends.Model):
    """Model class accessing the OpenAI remote API."""
    def __init__(self, client: openai.OpenAI, model_spec: backends.ModelSpec):
        """
        Args:
            client: An OpenAI library OpenAI client class.
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        self.client = client
        creds = backends.load_credentials(NAME)
        api_key = creds[NAME]["api_key"]
        organization = creds[NAME]["organisation"] if "organisation" in creds[NAME] else None
        self.async_client = AsyncOpenAI(api_key=api_key, organization=organization)

    def encode_image(self, image_path):
        """Encode an image to allow sending it to the OpenAI remote API.
        Args:
            image_path: Path to the image to be encoded.
        Returns:
            A tuple with a bool, True if encoding was successful, False otherwise, the image encoded as base64 string
            and a string containing the image type.
        """
        if image_path.startswith('http'):
            image_bytes = httpx.get(image_path).content
            image_type = imghdr.what(None, image_bytes)
            return True, image_path, image_type
        with open(image_path, "rb") as image_file:
            image_type = imghdr.what(image_path)
            return False, base64.b64encode(image_file.read()).decode('utf-8'), 'image/'+str(image_type)

    def encode_messages(self, messages) -> list:
        """Encode a message history containing images to allow sending it to the OpenAI remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The message history list with encoded images.
        """
        encoded_messages = []

        for message in messages:
            if "image" not in message.keys():
                encoded_messages.append(message)
            else:
                this = {"role": message["role"],
                        "content": [
                            {
                                "type": "text",
                                "text": message["content"].replace(" <image> ", " ")
                            }
                        ]}

                if "image" in message.keys() and 'multimodality' not in self.model_spec.model_config:
                    logger.info(
                        f"The backend {self.model_spec.__getattribute__('model_id')} does not support multimodal inputs!")
                    raise Exception(
                        f"The backend {self.model_spec.__getattribute__('model_id')} does not support multimodal inputs!")

                if 'multimodality' in self.model_spec.model_config:
                    if "image" in message.keys():

                        if not self.model_spec['model_config']['multimodality']['multiple_images'] and len(message['image']) > 1:
                            logger.info(f"The backend {self.model_spec.__getattribute__('model_id')} does not support multiple images!")
                            raise Exception(f"The backend {self.model_spec.__getattribute__('model_id')} does not support multiple images!")
                        else:
                            # encode each image
                            for image in message['image']:
                                is_url, loaded, image_type = self.encode_image(image)
                                if is_url:
                                    this["content"].append(dict(type="image_url", image_url={
                                        "url": loaded
                                    }))
                                else:
                                    this["content"].append(dict(type="image_url", image_url={
                                        "url": f"data:{image_type};base64,{loaded}"
                                    }))
                encoded_messages.append(this)
        return encoded_messages

    @retry(tries=3, delay=90, logger=logger)
    @ensure_messages_format
    def generate_response(self, messages: List[Dict]) -> Tuple[str, Any, str]:
        """Request a generated response from the OpenAI remote API.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
        Returns:
            The generated response message returned by the OpenAI remote API.
        """
        prompt = self.encode_messages(messages)

        if 'reasoning_model' in self.model_spec.model_config:
            api_response = self.client.chat.completions.create(model=self.model_spec.model_id,
                                                               messages=prompt,
                                                               temperature=1)
        else:
            api_response = self.client.chat.completions.create(model=self.model_spec.model_id,
                                                               messages=prompt,
                                                               temperature=self.get_temperature(),
                                                               max_tokens=self.get_max_tokens())
        message = api_response.choices[0].message
        if message.role != "assistant":  # safety check
            raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
        response_text = message.content.strip()
        response = json.loads(api_response.json())

        return prompt, response, response_text

    async def _generate_single_async(self, prompt: List[Dict], temperature: float) -> Tuple[Any, Any, str]:
        """Generate a single response asynchronously with retry logic."""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=90),
            retry=retry_if_exception_type((RateLimitError, APIError)),
            reraise=True
        ):
            with attempt:
                if 'reasoning_model' in self.model_spec.model_config:
                    api_response = await self.async_client.chat.completions.create(
                        model=self.model_spec.model_id,
                        messages=prompt,
                        temperature=temperature,
                        timeout=120.0  # 2 minute timeout per request
                    )
                else:
                    api_response = await self.async_client.chat.completions.create(
                        model=self.model_spec.model_id,
                        messages=prompt,
                        temperature=temperature,
                        max_tokens=self.get_max_tokens(),
                        timeout=25.0  # 2 minute timeout per request
                    )

                message = api_response.choices[0].message
                if message.role != "assistant":  # safety check
                    raise AttributeError("Response message role is " + message.role + " but should be 'assistant'")
                response_text = message.content.strip()
                response = json.loads(api_response.json())

                return prompt, response, response_text

    def batch_generate(self, batch_messages: List[List[Dict]], **kwargs) -> List[Tuple[Any, Any, str]]:
        """
        Generate responses for a batch of message histories concurrently.
        The order of responses matches the order of input messages.

        Args:
            batch_messages: A batch of message histories. Each message history is a list of dictionaries.
                Example:
                [
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Who won the world series in 2020?"},
                        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                        {"role": "user", "content": "Where was it played?"}
                    ],
                    [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Tell me a joke."}
                    ]
                ]

        Returns:
            A list of tuples, where each tuple contains:
                - The prompt used for generation.
                - The response object containing metadata.
                - The generated response text.
            The order matches the input batch_messages order.
        """
        temperature = kwargs.get('temp', self.get_temperature())
        print(f"Temperature: {temperature}")
        
        batch_prompts = [self.encode_messages(messages) for messages in batch_messages]
        
        # Run all requests concurrently (order is preserved by asyncio.gather)
        return asyncio.run(self._batch_generate_async(batch_prompts, temperature))
    
    async def _batch_generate_async(self, batch_prompts: List[List[Dict]], temperature: float) -> List[Tuple[Any, Any, str]]:
        """Generate responses for all prompts concurrently while preserving order."""
        # Limit concurrent requests to avoid rate limiting (adjust based on your API tier)
        semaphore = asyncio.Semaphore(5)  # Reduced to 5 concurrent requests
        
        async def limited_generate(prompt):
            async with semaphore:
                return await self._generate_single_async(prompt, temperature)
        
        tasks = [limited_generate(prompt) for prompt in batch_prompts]
        
        # Add overall timeout and better error handling
        try:
            # Total timeout: 25 seconds per request * 3 retries + buffer
            total_timeout = len(batch_prompts) * 25 / 5 + 60  # Adjust based on semaphore size
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=total_timeout
            )
            
            # Check for exceptions in results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Request {i} failed: {result}")
                    raise result
            
            return results
        except asyncio.TimeoutError:
            logger.error(f"Batch generation timed out after {total_timeout} seconds")
            raise
