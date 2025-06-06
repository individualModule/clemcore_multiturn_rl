"""Backend using HuggingFace transformers models.
Uses HF tokenizers instruct/chat templates for proper input format per model.
"""
import logging
from typing import List, Dict, Tuple, Any, Union
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from jinja2 import TemplateError

import clemcore.backends as backends
from clemcore.backends.utils import ensure_alternating_roles

logger = logging.getLogger(__name__)

FALLBACK_CONTEXT_SIZE = 256


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

    hf_model_str = model_spec['huggingface_id']
    if 'requires_api_key' in model_spec.model_config and model_spec['model_config']['requires_api_key']:
        # load HF API key:
        creds = backends.load_credentials("huggingface")
        api_key = creds["huggingface"]["api_key"]
        # load model using its default configuration:
        model = AutoModelForCausalLM.from_pretrained(hf_model_str, token=api_key, device_map="auto", torch_dtype="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_model_str, device_map="auto", torch_dtype="auto")

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
        
    def calculate_logprobs(self, observation: Union[List[dict], List[List[dict]]],
                                 action: Union[List[dict], List[List[dict]]]) -> torch.Tensor:
        
        """Calculate log probabilities for observation-action pairs.
        
        !!! Probably does not work well !!!
        Args:
            observation: Single observation or batch of observations
            action: Single action or batch of actions
            
        Returns:
            Tensor containing log probabilities for each sequence
        """

        # assert that both entries are the same format
        # if isinstance(observation[0], list):
        #     # if the item inside observation is a list - we are dealing with batches. 
        assert len(observation) == len(action), "Observation and action batches must have same length"
        #     assert isinstance(action[0], list)

        # do we need padding? 
        obs_template = self.tokenizer.apply_chat_template(observation, tokenize=False)
        action_template = self.tokenizer.apply_chat_template(action, tokenize=False)

        obs_tokens = self.tokenizer(obs_template, padding=True, truncation=True, return_tensors="pt").to(self.device)
        action_tokens = self.tokenizer(action_template, padding=True, truncation=True, return_tensors="pt").to(self.device)


        # Get the full sequence by concatenating
        full_input_ids = torch.cat([obs_tokens['input_ids'], action_tokens['input_ids']], dim=1)
        attention_mask = torch.cat([obs_tokens['attention_mask'], action_tokens['attention_mask']], dim=1)

        # Get model outputs
        outputs = self.model(input_ids=full_input_ids, attention_mask=attention_mask)
        
        # Calculate probabilities
        logits = outputs.logits
        probs = self.softmax(logits)
        
        # Get probabilities of the actual tokens that were generated
        # We look at the logits for positions corresponding to the action tokens
        action_probs = torch.gather(
            probs[:, obs_tokens['input_ids'].size(1)-1:-1], 
            2,
            action_tokens['input_ids'].unsqueeze(2)
        ).squeeze(2)
        
        # Calculate log probabilities and mask
        log_probs = torch.log(action_probs) * action_tokens['attention_mask']     

        return log_probs
    
    
    def generate_action_and_logprobs(self, observations: Union[List[dict], List[List[dict]]]) -> Tuple[List[List[dict]], torch.Tensor]:
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

        # Define the tokens to mask out
        special_tokens = [
            self.tokenizer.pad_token_id,  # Padding token
            128006,  # start_header_id
            128007,  # end_header_id
            128009   # eot_id
        ]

        # Ensure observations are in batch format
        if isinstance(observations[0], dict):
            observations = [observations]  # Wrap single observation in a list

        # Apply chat template and tokenize observations
        obs_template = self.tokenizer.apply_chat_template(observations, add_generation_prompt=True, tokenize=False)
        obs_tokens = self.tokenizer(obs_template, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # Greedy decoding or sampling
        do_sample: bool = False
        if self.get_temperature() > 0.0:
            do_sample = True

        if do_sample:
            outputs = self.model.generate(
                input_ids=obs_tokens['input_ids'],
                attention_mask=obs_tokens['attention_mask'],
                max_new_tokens=self.get_max_tokens(),
                temperature=self.get_temperature(),
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_logits=True
            )
        else:
            outputs = self.model.generate(
                input_ids=obs_tokens['input_ids'],
                attention_mask=obs_tokens['attention_mask'],
                max_new_tokens=self.get_max_tokens(),
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_logits=True
            )

        # Extract generated token IDs
        generated_ids = outputs.sequences[:, obs_tokens['input_ids'].size(1):]

        # Calculate logits and apply softmax
        logits = torch.stack(outputs.logits, dim=1)  # Shape: [batch_size, seq_len, vocab_size]
        probs = self.softmax(logits)  # Apply softmax to convert logits to probabilities
        log_probs = torch.log(probs)  # Convert probabilities to log probabilities

        # Gather log probabilities for the generated tokens
        action_log_probs = torch.gather(
            log_probs,
            2,
            generated_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out special tokens
        non_special_mask = torch.ones_like(generated_ids, dtype=torch.bool)
        for token_id in special_tokens:
            non_special_mask &= (generated_ids != token_id)

        # Apply the mask to exclude special tokens from log probabilities
        action_log_probs = action_log_probs * non_special_mask.float()

        # Decode generated actions into text
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Process the generated text and log probabilities
        actions = []
        filtered_log_probs = []
        for i, (text, generated_id_seq, log_prob_seq) in enumerate(zip(generated_texts, generated_ids, action_log_probs)):
            # Remove special tokens from the generated text
            prompt_text = self.tokenizer.decode(obs_tokens['input_ids'][i], skip_special_tokens=True).strip()
            response_text = text.replace(prompt_text, '').strip()
            print(response_text)
            if 'output_split_prefix' in self.model_spec.model_config:
                response_text = response_text.rsplit(self.model_spec['model_config']['output_split_prefix'], maxsplit=1)[1]
            eos_to_cull = self.model_spec['model_config']['eos_to_cull']
            response_text = re.sub(eos_to_cull, "", response_text)

            # Filter out log probabilities corresponding to special tokens
            valid_indices = non_special_mask[i].nonzero(as_tuple=True)[0]
            filtered_probs = log_prob_seq[valid_indices]
            actions.append([{"role": "assistant", "content": response_text}])
            filtered_log_probs.append(filtered_probs)

        # Pad filtered log probabilities to ensure consistent tensor shape
        filtered_log_probs = torch.nn.utils.rnn.pad_sequence(filtered_log_probs, batch_first=True, padding_value=0.0)

        # --- Sanity Check ---
        assert len(actions) == len(observations), "Mismatch between number of actions and observations."
        assert filtered_log_probs.size(0) == len(observations), "Log probabilities batch size mismatch."
        assert torch.isfinite(filtered_log_probs).all(), "Log probabilities contain NaN or Inf values."
        assert all(isinstance(action[0]["content"], str) and action[0]["content"] for action in actions), \
            "Generated actions are not valid strings."

        return actions, filtered_log_probs
    
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
