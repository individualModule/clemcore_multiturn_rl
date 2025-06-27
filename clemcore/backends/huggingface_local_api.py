"""Backend using HuggingFace transformers models.
Uses HF tokenizers instruct/chat templates for proper input format per model.
"""
from dataclasses import dataclass
import logging
from typing import List, Dict, Tuple, Any, Union
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StaticCache
from peft import PeftModel
from jinja2 import TemplateError

import clemcore.backends as backends
from clemcore.backends.utils import ensure_alternating_roles

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("clemcore.cli")

FALLBACK_CONTEXT_SIZE = 256

@dataclass
class GenerationOutputs:
    """Class to mimic the output structure of HuggingFace's generate() function."""
    sequences: torch.Tensor
    logits: list = None

def load_config_and_tokenizer(model_spec: backends.ModelSpec) -> Tuple[AutoTokenizer, AutoConfig, int]:
    """Load a HuggingFace model's standard config and tokenizer, and get context token limit from config.
    If the model config does not contain the context limit, it is set to 256 as fallback. Does not load the model
    weights, allowing for prototyping on non-GPU systems.
    Args:
        model_spec: The ModelSpec for the model.
    Returns:
        Tokenizer, model config and context token limit (int).
    """
    logger.info(f'Loading huggingface model config and tokenizer: {model_spec.model_name}')

    use_api_key = False
    api_key = None
    if 'requires_api_key' in model_spec.model_config:
        if model_spec['model_config']['requires_api_key']:
            # load HF API key:
            creds = backends.load_credentials("huggingface")
            api_key = creds["huggingface"]["api_key"]
            use_api_key = True
        else:
            requires_api_key_info = (f"{model_spec['model_name']} registry setting has requires_api_key, "
                                     f"but it is not 'true'. Please check the model entry.")
            print(requires_api_key_info)
            logger.info(requires_api_key_info)

    hf_model_str = model_spec['huggingface_id']

    # use 'slow' tokenizer for models that require it:
    if 'slow_tokenizer' in model_spec.model_config:
        if model_spec['model_config']['slow_tokenizer']:
            tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                      verbose=False, use_fast=False)
        else:
            tokenizer = None
            slow_tokenizer_info = (f"{model_spec['model_name']} registry setting has slow_tokenizer, "
                                   f"but it is not 'true'. Please check the model entry.")
            print(slow_tokenizer_info)
            logger.info(slow_tokenizer_info)
    elif use_api_key:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_str, token=api_key, device_map="auto",
                                                  torch_dtype="auto", verbose=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto",
                                                  verbose=False)

    # apply proper chat template:
    if not model_spec['model_config']['premade_chat_template']:
        if 'custom_chat_template' in model_spec.model_config:
            tokenizer.chat_template = model_spec['model_config']['custom_chat_template']
        else:
            logger.info(
                f"No custom chat template for {model_spec.model_name} found in model settings from model registry "
                f"while model has no pre-made template! Generic template will be used, likely leading to "
                f"bad results.")

    if use_api_key:
        model_config = AutoConfig.from_pretrained(hf_model_str, token=api_key)
    else:
        model_config = AutoConfig.from_pretrained(hf_model_str)

    # get context token limit for model:
    if hasattr(model_config, 'max_position_embeddings'):  # this is the standard attribute used by most
        context_size = model_config.max_position_embeddings
    elif hasattr(model_config, 'n_positions'):  # some models may have their context size under this attribute
        context_size = model_config.n_positions
    else:  # few models, especially older ones, might not have their context size in the config
        context_size = FALLBACK_CONTEXT_SIZE

    # stopping transformers pad_token_id warnings
    # check if tokenizer has no set pad_token_id:
    if not tokenizer.pad_token_id:  # if not set, pad_token_id is None
        # preemptively set pad_token_id to eos_token_id as automatically done to prevent warning at each generation:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model_config, context_size


def load_model(model_spec: backends.ModelSpec) -> Any:
    """Load Huggingface model weights, into VRAM if available.
    Weights are distributed over all available GPUs for maximum speed - make sure to limit the available GPUs using
    environment variables if only a subset is to be used.
    Args:
        model_spec: The ModelSpec for the model.
    Returns:
        The transformers model class instance of the loaded model.
    """
    logger.info(f'Start loading huggingface model weights: {model_spec.model_name}')

    model_args = dict(device_map="auto", torch_dtype="auto")
    if "load_in_8bit" in model_spec.model_config:
        model_args["load_in_8bit"] = model_spec.model_config["load_in_8bit"]
    if "load_in_4bit" in model_spec.model_config:
        model_args["load_in_4bit"] = model_spec.model_config["load_in_4bit"]
    if 'requires_api_key' in model_spec.model_config and model_spec['model_config']['requires_api_key']:
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        model_args["token"] = creds["huggingface"]["api_key"]

    hf_model_str = model_spec['huggingface_id']
    model = AutoModelForCausalLM.from_pretrained(hf_model_str, **model_args)

    if "peft_model" in model_spec.model_config:
        adapter_model = model_spec.model_config["peft_model"]  # can be a path or name
        stdout_logger.info(f"Load PeftModel adapters from {adapter_model}")
        model = PeftModel.from_pretrained(model, adapter_model)

    logger.info(f"Finished loading huggingface model: {model_spec.model_name}")
    logger.info(f"Model device map: {model.hf_device_map}")

    return model


class HuggingfaceLocal(backends.Backend):
    """Model/backend handler class for locally-run Huggingface models."""
    def __init__(self):
        super().__init__()

    def get_model_for(self, model_spec: backends.ModelSpec) -> backends.Model:
        """Get a HuggingFaceLocalModel instance with the passed model and settings.
        Will load all required data for using the model upon initialization.
        Args:
            model_spec: The ModelSpec for the model.
        Returns:
            The Model class instance of the model.
        """
        torch.set_num_threads(1)
        return HuggingfaceLocalModel(model_spec)


class HuggingfaceLocalModel(backends.Model):
    """Class for loaded HuggingFace transformers models ready for generation."""
    def __init__(self, model_spec: backends.ModelSpec):
        """
        Args:
            model_spec: A ModelSpec instance specifying the model.
        """
        super().__init__(model_spec)
        # fail-fast
        self.tokenizer, self.config, self.context_size = load_config_and_tokenizer(model_spec)
        self.model = load_model(model_spec)
        self.softmax = torch.nn.Softmax(dim=-1)  # softmax over vocabulary dimension   

        # check if model's generation_config has pad_token_id set:
        if not self.model.generation_config.pad_token_id:
            # set pad_token_id to tokenizer's eos_token_id to prevent excessive warnings:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def calculate_logprobs(self, prompt_ids, prompt_mask, completion_ids, completion_mask):
        """
        Inspired by the _forward class in TRL online DPO trainer. 
        Double forward pass to obtain logprobs: 
        1 - generate() to get completions
        2 - forward pass to get logprobs


        The padded tokens (based on completion mask) are assigned logprob 0 so they don't impact the loss
        """    

        # Get the number of tokens to truncate from prompt - 
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.context_size, 0)

        # Truncate left to avoid OOM
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]

        # Concatenate the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Forward pass through the model
        output = self.model(prompt_completion_ids, attention_mask=prompt_completion_mask)

        # There is 1 offset because the model predicts the next token
        logits = output.logits[:, prompt_ids.size(1) - 1 : -1]

        # Take the completion tokens log probabilities
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        # Mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        logprobs = logprobs * ~padding_mask  # Set logprobs for padding tokens to 0

        return logprobs

    def generate_action_and_logprobs(self,
                                    observations: Union[List[dict], List[List[dict]]],
                                    return_logprobs = True
                                    ) -> Tuple[List[List[dict]], torch.Tensor]:
        """
        https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo

        Generate policy actions and calculate their log probabilities in a single forward pass.
        Mask out special tokens so that it doesn't interfere with the gradient.
        Args:
            observations: A single observation or a batch of observations.

        Returns:
            Tuple containing:
                - Generated actions in the format List[List[dict]].
                - Log probabilities of the generated actions as a torch.Tensor.
        """
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        
        # Ensure observations are in batch format
        if isinstance(observations[0], dict):
            observations = [observations]  # Wrap single observation in a list
        print()
        print(observations)
        print()
        # Apply chat template and tokenize observations
        obs_template = self.tokenizer.apply_chat_template(observations, add_generation_prompt=True, tokenize=False)
        obs_tokens = self.tokenizer(obs_template, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Greedy decoding or sampling
        do_sample: bool = False
        if self.get_temperature() > 0.0:
            do_sample = True

        generate_kwargs = {
            "input_ids": obs_tokens['input_ids'],
            "attention_mask": obs_tokens['attention_mask'],
            "max_new_tokens": self.get_max_tokens(),
            "do_sample": do_sample,
            "return_dict_in_generate": True
        }

        if do_sample:
            generate_kwargs["temperature"] = self.get_temperature()

        outputs = self.model.generate(**generate_kwargs) # custom generation fn to get logprobs
        # Extract generated token IDs
        completion_ids = outputs.sequences[:, obs_tokens['input_ids'].size(1):]
        completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)
        generated_texts = self.tokenizer.batch_decode(outputs.sequences[:,:], skip_special_tokens=True)

        logprobs = None
        if return_logprobs:
            logprobs = self.calculate_logprobs(obs_tokens['input_ids'],
                                            obs_tokens['attention_mask'],
                                            completion_ids,
                                            completion_mask)
            
        actions = []

        for i, (text, generated_id_seq) in enumerate(zip(generated_texts, completion_ids)):
            # Remove special tokens from the generated text
            prompt_text = self.tokenizer.decode(obs_tokens['input_ids'][i], skip_special_tokens=True).strip()
            response_text = text.replace(prompt_text, '').strip()

            if 'output_split_prefix' in self.model_spec.model_config:
                response_text = response_text.rsplit(self.model_spec['model_config']['output_split_prefix'], maxsplit=1)[1]

            eos_to_cull = self.model_spec['model_config']['eos_to_cull']
            response_text = re.sub(eos_to_cull, "", response_text)

            actions.append([{"role": "assistant", "content": response_text}])

        print('---------------------')
        print(len(actions))
        print('+++++++++++==========++++++++')
        print(actions)
        print('=========================')
        print(actions[0])
        print()
        print(actions[0][0]['content'])
        print()
        print('-------------------')
        # Pad filtered log probabilities to ensure consistent tensor shape
        if return_logprobs:
            assert logprobs.size(0) == len(observations), "Log probabilities batch size mismatch."
            assert torch.isfinite(logprobs).all(), "Log probabilities contain NaN or Inf values."
            
        # --- Sanity Check ---
        assert len(actions) == len(observations), "Mismatch between number of actions and observations."
        assert all(isinstance(action[0]["content"], str) and action[0]["content"] for action in actions), \
            "Generated actions are not valid strings."

        return actions, logprobs

    def batch_generate(self, batch_messages: List[List[Dict]], return_full_text: bool = False, log_messages: bool = False):
        """
        Generate responses for a batch of message histories.

        Args:
            batch_messages: A batch of message histories. Each message history is a list of dictionaries.
            return_full_text: If True, return the full input context along with the response.
            log_messages: If True, log the raw and cleaned messages passed.

        Returns:
            A list of tuples, where each tuple contains:
                - The prompt used for generation.
                - The response object containing metadata.
                - The generated response text.
        """
        # Ensure batch_messages is a list of lists
        # print(f"Input batch_messages: {batch_messages}")
        assert isinstance(batch_messages, list) and all(isinstance(messages, list) for messages in batch_messages), \
            "batch_messages must be a list of message histories (lists of dictionaries)."

        # Log raw messages if requested
        if log_messages:
            for i, messages in enumerate(batch_messages):
                logger.info(f"Raw messages for batch {i}: {messages}")

        # Flatten and clean messages for each batch
        batch_cleaned_messages = [ensure_alternating_roles(messages) for messages in batch_messages]

        # Log cleaned messages if requested
        if log_messages:
            for i, messages in enumerate(batch_cleaned_messages):
                logger.info(f"Cleaned messages for batch {i}: {messages}")

        # Apply chat template and tokenize for the batch
        batch_prompt_template = self.tokenizer.apply_chat_template(batch_cleaned_messages, add_generation_prompt=True, tokenize=False)
        batch_prompt_tokens = self.tokenizer(batch_prompt_template, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Decode the prompts for logging
        batch_prompt_texts = self.tokenizer.batch_decode(batch_prompt_tokens["input_ids"], skip_special_tokens=True)

        # Check context limits for each batch
        for i, prompt_tokens in enumerate(batch_prompt_tokens["input_ids"]):
            context_check = _check_context_limit(self.context_size, prompt_tokens, max_new_tokens=self.get_max_tokens())
            if not context_check[0]:  # If context limit exceeded
                logger.info(f"Context token limit for batch {i} exceeded: {context_check[1]}/{context_check[3]}")
                raise backends.ContextExceededError(
                    f"Context token limit for batch {i} exceeded",
                    tokens_used=context_check[1],
                    tokens_left=context_check[2],
                    context_size=context_check[3]
                )

        # Perform generation for the batch
        do_sample = self.get_temperature() > 0.0
        generate_kwargs = {
            "input_ids": batch_prompt_tokens["input_ids"],
            "attention_mask": batch_prompt_tokens["attention_mask"],
            "max_new_tokens": self.get_max_tokens(),
            "do_sample": do_sample,
            "temperature": self.get_temperature() if do_sample else None
        }
        batch_model_output_ids = self.model.generate(**generate_kwargs)

        # Decode the generated outputs
        batch_model_outputs = self.tokenizer.batch_decode(batch_model_output_ids, skip_special_tokens=True)

        # Prepare the responses
        batch_responses = []
        for i, model_output in enumerate(batch_model_outputs):
            prompt_text = batch_prompt_texts[i]
            if not return_full_text:
                response_text = model_output.replace(prompt_text, "").strip()
                if "output_split_prefix" in self.model_spec.model_config:
                    response_text = response_text.rsplit(self.model_spec["model_config"]["output_split_prefix"], maxsplit=1)[1]
                eos_to_cull = self.model_spec["model_config"]["eos_to_cull"]
                response_text = re.sub(eos_to_cull, "", response_text)
            else:
                response_text = model_output.strip()

            response_object = {"response": model_output}
            batch_responses.append((prompt_text, response_object, response_text))

            # Log the response if requested
            if log_messages:
                logger.info(f"Response for batch {i}: {response_text}")
        # print("Output batch_responses (response_text only):")
        # for response in batch_responses:
        #     print(response[2])  # Print only the response_text

        return batch_responses

    def generate_response(self, messages: List[Dict],
                          return_full_text: bool = False,
                          log_messages: bool = False) -> Tuple[Any, Any, str]:
        """Generate a response with the loaded HuggingFace transformers model.
        Args:
            messages: A message history. For example:
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Who won the world series in 2020?"},
                    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                    {"role": "user", "content": "Where was it played?"}
                ]
            return_full_text: If True, whole input context is returned.
            log_messages: If True, raw and cleaned messages passed will be logged.
        Returns:
            The response message generated by the loaded HuggingFace transformers model.
        """
        # log current given messages list:
        if log_messages:
            logger.info(f"Raw messages passed: {messages}")

        current_messages = ensure_alternating_roles(messages)

        # log current flattened messages list:
        if log_messages:
            logger.info(f"Flattened messages: {current_messages}")

        # apply chat template & tokenize:
        prompt_tokens = self.tokenizer.apply_chat_template(current_messages, add_generation_prompt=True,
                                                           return_tensors="pt")
        prompt_tokens = prompt_tokens.to(self.device)

        prompt_text = self.tokenizer.batch_decode(prompt_tokens)[0]
        prompt = {"inputs": prompt_text, "max_new_tokens": self.get_max_tokens(),
                  "temperature": self.get_temperature(), "return_full_text": return_full_text}

        # check context limit:
        context_check = _check_context_limit(self.context_size, prompt_tokens[0],
                                             max_new_tokens=self.get_max_tokens())
        if not context_check[0]:  # if context is exceeded, context_check[0] is False
            logger.info(f"Context token limit for {self.model_spec.model_name} exceeded: "
                        f"{context_check[1]}/{context_check[3]}")
            # fail gracefully:
            raise backends.ContextExceededError(f"Context token limit for {self.model_spec.model_name} exceeded",
                                                tokens_used=context_check[1], tokens_left=context_check[2],
                                                context_size=context_check[3])

        # greedy decoding:
        do_sample: bool = False
        if self.get_temperature() > 0.0:
            do_sample = True

        if do_sample:
            model_output_ids = self.model.generate(
                prompt_tokens,
                temperature=self.get_temperature(),
                max_new_tokens=self.get_max_tokens(),
                do_sample=do_sample
            )
        else:
            model_output_ids = self.model.generate(
                prompt_tokens,
                max_new_tokens=self.get_max_tokens(),
                do_sample=do_sample
            )

        model_output = self.tokenizer.batch_decode(model_output_ids)[0]

        response = {'response': model_output}

        # cull input context; equivalent to transformers.pipeline method:
        if not return_full_text:
            response_text = model_output.replace(prompt_text, '').strip()

            if 'output_split_prefix' in self.model_spec.model_config:
                response_text = model_output.rsplit(self.model_spec['model_config']['output_split_prefix'], maxsplit=1)[1]

            # remove eos token string:
            eos_to_cull = self.model_spec['model_config']['eos_to_cull']
            response_text = re.sub(eos_to_cull, "", response_text)
        else:
            response_text = model_output.strip()

        if log_messages:
            logger.info(f"Response message: {response_text}")

        return prompt, response, response_text


def _check_context_limit(context_size, prompt_tokens, max_new_tokens: int = 100) -> Tuple[bool, int, int, int]:
    """Internal context limit check to run in generate_response.
    Args:
        prompt_tokens: List of prompt token IDs.
        max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
    Returns:
        Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    prompt_size = len(prompt_tokens)
    tokens_used = prompt_size + max_new_tokens  # context includes tokens to be generated
    tokens_left = context_size - tokens_used
    fits = tokens_used <= context_size
    return fits, tokens_used, tokens_left, context_size


def check_messages(messages: List[Dict], model_spec: backends.ModelSpec) -> bool:
    """Message checking for clemgame development.
    This checks if the model's chat template accepts the given messages as passed, before the standard flattening done
    for generation. This allows clemgame developers to construct message lists that are sound as-is and are not affected
    by the indiscriminate flattening of the generation method. Deliberately verbose.
    Args:
        model_spec: The ModelSpec for the model.
        messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
    Returns:
        True if messages are sound as-is, False if messages are not compatible with the model's template.
    """
    tokenizer, _, _ = load_config_and_tokenizer(model_spec)

    # bool for message acceptance:
    messages_accepted: bool = True

    # check for system message:
    has_system_message: bool = False
    if messages[0]['role'] == "system":
        print("System message detected.")
        has_system_message = True
        if not messages[0]['content']:
            print(f"Initial system message is empty. It will be removed when generating responses.")
        else:
            print(f"Initial system message has content! It will not be removed when generating responses. This "
                  f"will lead to issues with models that do not allow system messages.")
        """
        print("Checking model system message compatibility...")
        # unfortunately Mistral models, which do not accept system message, currently do not raise a distinct 
        # exception for this...
        try:
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except TemplateError:
            print("The model's chat template does not allow for system message!")
            messages_accepted = False
        """

    # check for message order:
    starts_with_assistant: bool = False
    double_user: bool = False
    double_assistant: bool = False
    ends_with_assistant: bool = False

    for msg_idx, message in enumerate(messages):
        if not has_system_message:
            if msg_idx == 0 and message['role'] == "assistant":
                starts_with_assistant = True
        else:
            if msg_idx == 1 and message['role'] == "assistant":
                starts_with_assistant = True
        if msg_idx > 0 and message['role'] == "user" and messages[msg_idx - 1]['role'] == "user":
            double_user = True
        elif msg_idx > 0 and message['role'] == "assistant" and messages[msg_idx - 1]['role'] == "assistant":
            double_assistant = True
    if messages[-1]['role'] == "assistant":
        ends_with_assistant = True

    if starts_with_assistant or double_user or double_assistant or ends_with_assistant:
        print("Message order issue(s) found:")
        if starts_with_assistant:
            print("First message has role:'assistant'.")
        if double_user:
            print("Messages contain consecutive user messages.")
        if double_assistant:
            print("Messages contain consecutive assistant messages.")
        if ends_with_assistant:
            print("Last message has role:'assistant'.")

    # proper check of chat template application:
    try:
        tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    except TemplateError:
        print(f"The {model_spec.model_name} chat template does not accept these messages! "
              f"Cleaning applied before generation might still allow these messages, but is indiscriminate and "
              f"might lead to unintended generation inputs.")
        messages_accepted = False
    else:
        print(
            f"The {model_spec.model_name} chat template accepts these messages. Cleaning before generation is still "
            f"applied to these messages, which is indiscriminate and might lead to unintended generation inputs.")

    return messages_accepted


def check_context_limit(messages: List[Dict], model_spec: backends.ModelSpec,
                        max_new_tokens: int = 100, clean_messages: bool = False,
                        verbose: bool = True) -> Tuple[bool, int, int, int]:
    """Externally-callable context limit check for clemgame development.
    Args:
        messages: for example
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Who won the world series in 2020?"},
                {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                {"role": "user", "content": "Where was it played?"}
            ]
        model_spec: The ModelSpec for the model.
        max_new_tokens: How many tokens to generate ('at most', but no stop sequence is defined).
        clean_messages: If True, the standard cleaning method for message lists will be applied.
        verbose: If True, prettyprint token counts.
    Returns:
        Tuple with
            Bool: True if context limit is not exceeded, False if too many tokens
            Number of tokens for the given messages and maximum new tokens
            Number of tokens of 'context space left'
            Total context token limit
    """
    tokenizer, _, context_size = load_config_and_tokenizer(model_spec)

    # optional messages processing:
    if clean_messages:
        current_messages = ensure_alternating_roles(messages)
    else:
        current_messages = messages
    # the actual tokens, including chat format:
    prompt_tokens = tokenizer.apply_chat_template(current_messages, add_generation_prompt=True)
    context_check_tuple = _check_context_limit(context_size, prompt_tokens, max_new_tokens=max_new_tokens)
    tokens_used = context_check_tuple[1]
    tokens_left = context_check_tuple[2]
    if verbose:
        print(f"{tokens_used} input tokens, {tokens_left} tokens of {context_size} left.")
    fits = context_check_tuple[0]
    return fits, tokens_used, tokens_left, context_size



def truncate_right(
    input_ids: torch.Tensor, stop_token_id: int, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Truncates the input tensor from the right side after the first occurrence of the stop token.

    Args:
        input_ids (`torch.Tensor`):
            The tensor containing the responses to be truncated
        stop_token_id (`int`):
            The token ID representing the stop token where truncation occurs
        pad_token_id (`int`):
            The token ID representing the pad token used to fill the truncated responses

    Returns:
        tuple:
            - `output_ids` (`torch.Tensor`):
                The truncated responses tensor with pad tokens filled after the stop token
            - `mask` (`torch.Tensor`):
                The mask tensor to indicate the padding tokens
    """
    if isinstance(input_ids, torch.Tensor):
        print("input_ids is a tensor.")
    else:
        print("input_ids is not a tensor.")

    trunc_idxs = first_true_indices(input_ids == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(input_ids.size()) - 1) + [input_ids.shape[1]]
    idxs = torch.arange(input_ids.shape[1], device=input_ids.device).view(*new_size)
    output_ids = torch.masked_fill(input_ids, idxs > trunc_idxs, pad_token_id)
    mask = torch.masked_fill(torch.ones_like(input_ids), idxs > trunc_idxs, 0)

    return output_ids, mask

def first_true_indices(bools: torch.Tensor, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.

    Args:
        bools (`torch.Tensor`):
            An N-dimensional boolean tensor.
        dtype (`torch.dtype`, optional):
            The desired data type of the output tensor. Defaults to `torch.long`.

    Returns:
        `torch.Tensor`:
            An (N-1)-dimensional tensor of integers indicating the position of the first True
            in each row. If no True value is found in a row, returns the length of the row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values