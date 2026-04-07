"""Types and request signatures for OpenAI compatibility

NOTE: These types may change to match the OpenAI OpenAPI specification.

Based on the OpenAI OpenAPI specification:
https://app.stainless.com/api/spec/documented/openai/openapi.documented.yml

"""

from typing import Any, List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal


# NOTE: Defining this correctly using annotations seems to break pydantic validation.
#       This is a workaround until we can figure out how to do this correctly
# JsonType = Union[None, int, str, bool, List["JsonType"], Dict[str, "JsonType"]]
JsonType = Union[None, int, str, bool, List[Any], Dict[str, Any]]


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class Embedding(TypedDict):
    index: int
    object: str
    embedding: Union[List[float], List[List[float]]]


class CreateEmbeddingResponse(TypedDict):
    object: Literal["list"]
    model: str
    data: List[Embedding]
    usage: EmbeddingUsage


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[Literal["stop", "length", "content_filter"]]


class PromptTokensDetails(TypedDict):
    cached_tokens: NotRequired[int]
    audio_tokens: NotRequired[int]


class CompletionTokensDetails(TypedDict):
    reasoning_tokens: NotRequired[int]
    audio_tokens: NotRequired[int]
    accepted_prediction_tokens: NotRequired[int]
    rejected_prediction_tokens: NotRequired[int]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: NotRequired[PromptTokensDetails]
    completion_tokens_details: NotRequired[CompletionTokensDetails]


class CreateCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: NotRequired[CompletionUsage]


class ChatCompletionResponseFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionResponseMessageFunctionCall(TypedDict):
    arguments: str
    name: str


class ChatCompletionResponseMessageAudio(TypedDict):
    id: str
    expires_at: int
    data: str
    transcript: str


class ChatCompletionResponseMessageAnnotationURLCitation(TypedDict):
    end_index: int
    start_index: int
    url: str
    title: str

class ChatCompletionResponseMessageAnnotation(TypedDict):
    type: Literal["url_citation"]
    url_citation: ChatCompletionResponseMessageAnnotationURLCitation


class ChatCompletionResponseMessage(TypedDict):
    content: Optional[str]
    refusal: Optional[str]
    role: Literal["assistant"]
    tool_calls: NotRequired["ChatCompletionMessageToolCalls"]
    annotations: NotRequired[List[ChatCompletionResponseMessageAnnotation]]
    function_call: NotRequired[ChatCompletionResponseMessageFunctionCall]  # DEPRECATED
    audio: NotRequired[Optional[ChatCompletionResponseMessageAudio]]


class ChatCompletionFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionTopLogprobToken(TypedDict):
    token: str
    logprob: float
    bytes: Optional[List[int]]


class ChatCompletionLogprobToken(ChatCompletionTopLogprobToken):
    token: str
    logprob: float
    bytes: Optional[List[int]]
    top_logprobs: List[ChatCompletionTopLogprobToken]


class ChatCompletionLogprobs(TypedDict):
    content: Optional[List[ChatCompletionLogprobToken]]
    refusal: Optional[List[ChatCompletionLogprobToken]]


class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: "ChatCompletionResponseMessage"
    logprobs: Optional[ChatCompletionLogprobs]
    finish_reason: Optional[str]


class CreateChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List["ChatCompletionResponseChoice"]
    usage: CompletionUsage


class ChatCompletionMessageToolCallChunkFunction(TypedDict):
    name: Optional[str]
    arguments: str


class ChatCompletionMessageToolCallChunk(TypedDict):
    index: int
    id: NotRequired[str]
    type: Literal["function"]
    function: ChatCompletionMessageToolCallChunkFunction


class ChatCompletionStreamResponseDeltaEmpty(TypedDict):
    pass


class ChatCompletionStreamResponseDeltaFunctionCall(TypedDict):
    name: str
    arguments: str


ChatCompletionRole = Literal[
    "developer",
    "system",
    "user",
    "assistant",
    "tool",
    "function"
]


class ChatCompletionStreamResponseDelta(TypedDict):
    content: NotRequired[Optional[str]]
    function_call: NotRequired[
        Optional[ChatCompletionStreamResponseDeltaFunctionCall]
    ]  # DEPRECATED
    tool_calls: NotRequired[Optional[List[ChatCompletionMessageToolCallChunk]]]
    role: NotRequired[Optional[Literal["developer", "system", "user", "assistant", "tool"]]]


class ChatCompletionStreamResponseChoice(TypedDict):
    index: int
    delta: Union[
        ChatCompletionStreamResponseDelta, ChatCompletionStreamResponseDeltaEmpty
    ]
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]]
    logprobs: NotRequired[Optional[ChatCompletionLogprobs]]


class CreateChatCompletionStreamResponse(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[ChatCompletionStreamResponseChoice]
    usage: NotRequired[CompletionUsage]


class ChatCompletionFunctions(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionFunctionCallOption(TypedDict):
    name: str


class ChatCompletionResponseFormatJSONSchema(TypedDict):
    name: str
    description: NotRequired[str]
    schema: NotRequired[Dict[str, Any]]
    strict: NotRequired[Optional[bool]]


class ChatCompletionRequestResponseFormat(TypedDict):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: NotRequired[ChatCompletionResponseFormatJSONSchema]


class ChatCompletionRequestMessageContentPartText(TypedDict):
    type: Literal["text"]
    text: str


class ChatCompletionRequestMessageContentPartImageImageUrl(TypedDict):
    url: str
    detail: NotRequired[Literal["auto", "low", "high"]]


class ChatCompletionRequestMessageContentPartImage(TypedDict):
    type: Literal["image_url"]
    image_url: Union[str, ChatCompletionRequestMessageContentPartImageImageUrl]


class ChatCompletionRequestMessageContentPartInputAudioData(TypedDict):
    data: str
    format: Literal["wav", "mp3"]


class ChatCompletionRequestMessageContentPartAudio(TypedDict):
    type: Literal["input_audio"]
    input_audio: ChatCompletionRequestMessageContentPartInputAudioData


class ChatCompletionRequestMessageContentPartFileData(TypedDict):
    filename: NotRequired[str]
    file_data: NotRequired[str]
    file_id: NotRequired[str]


class ChatCompletionRequestMessageContentPartFile(TypedDict):
    type: Literal["file"]
    file: ChatCompletionRequestMessageContentPartFileData


class ChatCompletionRequestMessageContentPartRefusal(TypedDict):
    type: Literal["refusal"]
    refusal: str


ChatCompletionRequestMessageContentPart = Union[
    ChatCompletionRequestMessageContentPartText,
    ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartAudio,
    ChatCompletionRequestMessageContentPartFile,
]


class ChatCompletionRequestDeveloperMessage(TypedDict):
    role: Literal["developer"]
    content: Optional[str]


class ChatCompletionRequestSystemMessage(TypedDict):
    role: Literal["system"]
    content: Optional[str]


class ChatCompletionRequestUserMessage(TypedDict):
    role: Literal["user"]
    content: Optional[Union[str, List[ChatCompletionRequestMessageContentPart]]]


class ChatCompletionMessageToolCallFunction(TypedDict):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ChatCompletionMessageToolCallFunction


ChatCompletionMessageToolCalls = List[ChatCompletionMessageToolCall]


class ChatCompletionRequestAssistantMessageFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionRequestAssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: NotRequired[Optional[str]]
    refusal: NotRequired[Optional[str]]
    tool_calls: NotRequired[ChatCompletionMessageToolCalls]
    function_call: NotRequired[
        ChatCompletionRequestAssistantMessageFunctionCall
    ]  # DEPRECATED


class ChatCompletionRequestToolMessage(TypedDict):
    role: Literal["tool"]
    content: Optional[str]
    tool_call_id: str


class ChatCompletionRequestFunctionMessage(TypedDict):
    role: Literal["function"]
    content: Optional[str]
    name: str


ChatCompletionRequestMessage = Union[
    ChatCompletionRequestDeveloperMessage,
    ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage,
    ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestToolMessage,
    ChatCompletionRequestFunctionMessage,
]


class ChatCompletionRequestFunctionCallOption(TypedDict):
    name: str


ChatCompletionRequestFunctionCall = Union[
    Literal["none", "auto"], ChatCompletionRequestFunctionCallOption
]

ChatCompletionFunctionParameters = Dict[str, JsonType]  # TODO: make this more specific


class ChatCompletionToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: ChatCompletionFunctionParameters


class ChatCompletionTool(TypedDict):
    type: Literal["function"]
    function: ChatCompletionToolFunction


class ChatCompletionAllowedTools(TypedDict):
    mode: Literal["auto", "required"]
    tools: List[Dict[str, Any]]


class ChatCompletionAllowedToolsChoice(TypedDict):
    type: Literal["allowed_tools"]
    allowed_tools: ChatCompletionAllowedTools


class ChatCompletionNamedToolChoiceFunction(TypedDict):
    name: str


class ChatCompletionNamedToolChoice(TypedDict):
    type: Literal["function"]
    function: ChatCompletionNamedToolChoiceFunction


class ChatCompletionNamedToolChoiceCustomObject(TypedDict):
    name: str


class ChatCompletionNamedToolChoiceCustom(TypedDict):
    type: Literal["custom"]
    custom: ChatCompletionNamedToolChoiceCustomObject


ChatCompletionToolChoiceOption = Union[
    Literal["none", "auto", "required"],
    ChatCompletionAllowedToolsChoice,
    ChatCompletionNamedToolChoice,
    ChatCompletionNamedToolChoiceCustom
]


# NOTE: The following type names are not part of the OpenAI OpenAPI specification
# and will be removed in a future major release.

EmbeddingData = Embedding
CompletionChunk = CreateCompletionResponse
Completion = CreateCompletionResponse
CreateCompletionStreamResponse = CreateCompletionResponse
ChatCompletionMessage = ChatCompletionResponseMessage
ChatCompletionChoice = ChatCompletionResponseChoice
ChatCompletion = CreateChatCompletionResponse
ChatCompletionChunkDeltaEmpty = ChatCompletionStreamResponseDeltaEmpty
ChatCompletionChunkChoice = ChatCompletionStreamResponseChoice
ChatCompletionChunkDelta = ChatCompletionStreamResponseDelta
ChatCompletionChunk = CreateChatCompletionStreamResponse
ChatCompletionStreamResponse = CreateChatCompletionStreamResponse
ChatCompletionResponseFunction = ChatCompletionFunction
ChatCompletionFunctionCall = ChatCompletionResponseFunctionCall
