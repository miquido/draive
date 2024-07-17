# copied from https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

@dataclass
class ChatCompletionRequest:
    messages: (list[dict[str, str]] | list[dict[str, list[dict[str, str | dict[str, str]]]]]) | str
    model: str
    logit_bias: dict[int, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = None
    n_choices: int = 1
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop_seqs: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    top_k: int | None = None
    grammar: str | None = None
    grammar_type: str | None = None
    adapters: list[str] | None = None

@dataclass
class CompletionRequest:
    prompt: str
    model: str
    echo_prompt: bool = False
    logit_bias: dict[int, float] | None = None
    max_tokens: int | None = None
    n_choices: int = 1
    best_of: int = 1
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stop_seqs: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    suffix: str | None = None
    grammar: str | None = None
    grammar_type: str | None = None
    adapters: list[str] | None = None

class Architecture(Enum):
    Mistral = "mistral"
    Gemma = "gemma"
    Mixtral = "mixtral"
    Llama = "llama"
    Phi2 = "phi2"
    Phi3 = "phi3"

class VisionArchitecture(Enum):
    Phi3V = "phi3v"

class Which:
    @dataclass
    class Plain:
        model_id: str
        arch: Architecture
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class XLora:
        arch: Architecture
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class Lora:
        arch: Architecture
        adapters_model_id: str
        order: str
        model_id: str | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class GGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        repeat_last_n: int = 64

    @dataclass
    class XLoraGGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        repeat_last_n: int = 64

    @dataclass
    class LoraGGUF:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        repeat_last_n: int = 64

    @dataclass
    class GGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class XLoraGGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        xlora_model_id: str
        order: str
        tgt_non_granular_index: int | None = None
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class LoraGGML:
        tok_model_id: str
        quantized_model_id: str
        quantized_filename: str
        adapters_model_id: str
        order: str
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

    @dataclass
    class VisionPlain:
        model_id: str
        arch: VisionArchitecture
        tokenizer_json: str | None = None
        repeat_last_n: int = 64

class Runner:
    def __init__(  # noqa: PLR0913
        self,
        which: Which.Plain
        | Which.Lora
        | Which.XLora
        | Which.GGUF
        | Which.GGML
        | Which.LoraGGML
        | Which.LoraGGUF
        | Which.XLoraGGML
        | Which.XLoraGGUF
        | Which.VisionPlain,
        max_seqs: int = 16,
        no_kv_cache: bool = False,
        prefix_cache_n: int = 16,
        token_source: str = "cache",
        speculative_gamma: int = 32,
        which_draft: Which | None = None,
        chat_template: str | None = None,
        num_device_layers: int | None = None,
        in_situ_quant: str | None = None,
    ) -> None: ...
    def send_chat_completion_request(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse | Iterator[ChatCompletionChunkResponse]: ...
    def send_completion_request(self, request: CompletionRequest) -> CompletionResponse: ...
    def send_re_isq(self, dtype: str) -> CompletionResponse: ...
    def activate_adapters(self, adapter_names: list[str]) -> None: ...

@dataclass
class Usage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    avg_tok_per_sec: float
    avg_prompt_tok_per_sec: float
    avg_compl_tok_per_sec: float
    total_time_sec: float
    total_prompt_time_sec: float
    total_completion_time_sec: float

@dataclass
class ResponseMessage:
    content: str
    role: str

@dataclass
class TopLogprob:
    token: int
    logprob: float
    bytes: str

@dataclass
class ResponseLogprob:
    token: str
    logprob: float
    bytes: list[int]
    top_logprobs: list[TopLogprob]

@dataclass
class Logprobs:
    content: list[ResponseLogprob] | None

@dataclass
class Choice:
    finish_reason: str
    index: int
    message: ResponseMessage
    logprobs: Logprobs

@dataclass
class ChatCompletionResponse:
    id: str
    choices: list[Choice]
    created: int
    model: str
    system_fingerprint: str
    object: str
    usage: Usage

@dataclass
class Delta:
    content: str
    role: str

@dataclass
class ChunkChoice:
    finish_reason: str | None
    index: int
    delta: Delta
    logprobs: ResponseLogprob | None

@dataclass
class ChatCompletionChunkResponse:
    id: str
    choices: list[ChunkChoice]
    created: int
    model: str
    system_fingerprint: str
    object: str

@dataclass
class CompletionChoice:
    finish_reason: str
    index: int
    text: str

@dataclass
class CompletionResponse:
    id: str
    choices: list[CompletionChoice]
    created: int
    model: str
    system_fingerprint: str
    object: str
    usage: Usage
