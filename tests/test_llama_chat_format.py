import jinja2
import numpy as np
import pytest

from llama_cpp.llama_chat_format import Jinja2ChatFormatter

QWEN35_EOS_TOKEN = "<|im_end|>"

# A compact Qwen3.5-style template keeps these tests independent of model files.
QWEN35_CHAT_TEMPLATE = r"""
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, is_system=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if "image" in item or "image_url" in item or item.type == "image" %}
                {%- if is_system %}
                    {{- raise_exception("System message cannot contain images.") }}
                {%- endif %}
                {%- set image_count.value = image_count.value + 1 %}
                {%- if add_vision_id %}
                    {{- "Picture " ~ image_count.value ~ ": " }}
                {%- endif %}
                {{- "<|vision_start|><|image_pad|><|vision_end|>" }}
            {%- elif "video" in item or item.type == "video" %}
                {%- if is_system %}
                    {{- raise_exception("System message cannot contain videos.") }}
                {%- endif %}
                {%- set video_count.value = video_count.value + 1 %}
                {%- if add_vision_id %}
                    {{- "Video " ~ video_count.value ~ ": " }}
                {%- endif %}
                {{- "<|vision_start|><|video_pad|><|vision_end|>" }}
            {%- elif "text" in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception("Unexpected item type in content.") }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- "" }}
    {%- else %}
        {{- raise_exception("Unexpected content type.") }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception("No messages provided.") }}
{%- endif %}
{%- if tools %}
    {{- "<|im_start|>system\n# Tools\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" ~ (tool | tojson) }}
    {%- endfor %}
    {{- "\n</tools><|im_end|>\n" }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(
        message.content, message.role == "system"
    ) | trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception("System message must be at the beginning.") }}
        {%- endif %}
        {{- "<|im_start|>system\n" ~ content ~ "<|im_end|>\n" }}
    {%- elif message.role == "user" %}
        {{- "<|im_start|>user\n" ~ content ~ "<|im_end|>\n" }}
    {%- elif message.role == "assistant" %}
        {{- "<|im_start|>assistant\n" }}
        {%- if message.reasoning_content is string %}
            {{- "<think>\n" ~ (message.reasoning_content | trim)
                ~ "\n</think>\n\n" }}
        {%- endif %}
        {{- content }}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- set call = tool_call.function %}
                {{- "\n\n<tool_call>\n<function=" ~ call.name ~ ">\n" }}
                {%- for name, value in call.arguments | items %}
                    {{- "<parameter=" ~ name ~ ">\n" ~ value
                        ~ "\n</parameter>\n" }}
                {%- endfor %}
                {{- "</function>\n</tool_call>" }}
            {%- endfor %}
        {%- endif %}
        {{- "<|im_end|>\n" }}
    {%- elif message.role == "tool" %}
        {{- "<|im_start|>user\n<tool_response>\n" ~ content
            ~ "\n</tool_response><|im_end|>\n" }}
    {%- else %}
        {{- raise_exception("Unexpected message role.") }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- "<|im_start|>assistant\n" }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- "<think>\n\n</think>\n\n" }}
    {%- else %}
        {{- "<think>\n" }}
    {%- endif %}
{%- endif %}
"""


@pytest.fixture()
def qwen35_formatter() -> Jinja2ChatFormatter:
    return Jinja2ChatFormatter(
        template=QWEN35_CHAT_TEMPLATE,
        eos_token=QWEN35_EOS_TOKEN,
        bos_token="",
        add_generation_prompt=True,
    )


def test_qwen35_basic_conversation(qwen35_formatter: Jinja2ChatFormatter):
    response = qwen35_formatter(
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
        enable_thinking=False,
    )

    assert response.prompt == (
        "<|im_start|>system\n"
        "Be concise.<|im_end|>\n"
        "<|im_start|>user\n"
        "Hello<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )
    assert response.stop == [QWEN35_EOS_TOKEN]
    assert response.added_special is True


@pytest.mark.parametrize(
    ("enable_thinking", "expected_suffix"),
    [
        (True, "<|im_start|>assistant\n<think>\n"),
        (False, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ],
)
def test_qwen35_generation_prompt_thinking_modes(
    qwen35_formatter: Jinja2ChatFormatter,
    enable_thinking: bool,
    expected_suffix: str,
):
    response = qwen35_formatter(
        messages=[{"role": "user", "content": "Solve this problem."}],
        enable_thinking=enable_thinking,
    )

    assert response.prompt.endswith(expected_suffix)


def test_qwen35_multimodal_content(qwen35_formatter: Jinja2ChatFormatter):
    # Qwen3.5 assigns separate sequence numbers to images and videos.
    response = qwen35_formatter(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "image.png"},
                    },
                    {"type": "text", "text": "Compare this with "},
                    {"type": "video", "video": "video.mp4"},
                ],
            }
        ],
        add_vision_id=True,
        enable_thinking=False,
    )

    assert "Picture 1: <|vision_start|><|image_pad|><|vision_end|>" in response.prompt
    assert (
        "Compare this with Video 1: "
        "<|vision_start|><|video_pad|><|vision_end|>" in response.prompt
    )
    assert response.prompt.count("<|vision_start|>") == 2
    assert response.prompt.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen35_tools_and_tool_history(qwen35_formatter: Jinja2ChatFormatter):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    response = qwen35_formatter(
        messages=[
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": "I will check.",
                "reasoning_content": "A weather lookup is required.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "Sunny, 28 C",
            },
        ],
        tools=tools,
        enable_thinking=False,
    )

    # Tool calls and their responses use Qwen3.5's XML-like markers.
    assert '"description": "Get the current weather for a city"' in response.prompt
    assert "<think>\nA weather lookup is required.\n</think>" in response.prompt
    assert (
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\n"
        "London\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>" in response.prompt
    )
    assert "<tool_response>\nSunny, 28 C\n</tool_response>" in response.prompt


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        ([], "No messages provided."),
        (
            [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Too late"},
            ],
            "System message must be at the beginning.",
        ),
        (
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "image.png"},
                        }
                    ],
                },
                {"role": "user", "content": "Hello"},
            ],
            "System message cannot contain images.",
        ),
    ],
)
def test_qwen35_rejects_invalid_messages(
    qwen35_formatter: Jinja2ChatFormatter,
    messages,
    error: str,
):
    with pytest.raises(jinja2.TemplateError, match=error):
        qwen35_formatter(messages=messages)


def test_qwen35_stop_token_ids():
    # Verify that model-specific stop token IDs terminate generation.
    formatter = Jinja2ChatFormatter(
        template=QWEN35_CHAT_TEMPLATE,
        eos_token=QWEN35_EOS_TOKEN,
        bos_token="",
        stop_token_ids=[248044],
    )
    response = formatter(messages=[{"role": "user", "content": "Hello"}])

    assert response.stopping_criteria is not None
    criterion = response.stopping_criteria[0]
    logits = np.empty(0, dtype=np.single)

    assert criterion(np.array([], dtype=np.intc), logits) is False
    assert criterion(np.array([1, 248044], dtype=np.intc), logits) is True
    assert criterion(np.array([1, 2], dtype=np.intc), logits) is False
