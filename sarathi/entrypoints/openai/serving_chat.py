import codecs
import time
from dataclasses import dataclass
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Iterable,
    List,
    Optional,
    TypedDict,
    Union,
    cast,
    final,
)

from fastapi import Request
from openai.types.chat import ChatCompletionContentPartTextParam

from sarathi.config import ModelConfig
from sarathi.core.datatypes.request_output import RequestOutput
from sarathi.engine.async_llm_engine import AsyncLLMEngine
from sarathi.entrypoints.openai.protocol import (
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    FunctionCall,
    ToolCall,
    UsageInfo,
)
from sarathi.entrypoints.openai.serving_engine import OpenAIServing
from sarathi.logger import init_logger
from sarathi.utils import random_uuid

logger = init_logger(__name__)


@final  # So that it should be compatible with Dict[str, str]
class ConversationMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class ChatMessageParseResult:
    messages: List[ConversationMessage]


class OpenAIServingChat(OpenAIServing):

    def __init__(
        self,
        engine: AsyncLLMEngine,
        model_config: ModelConfig,
        served_model_names: List[str],
        response_role: str,
        chat_template: Optional[str] = None,
    ):
        super().__init__(
            engine=engine,
            model_config=model_config,
            served_model_names=served_model_names,
        )

        self.response_role = response_role
        self._load_chat_template(chat_template)

    def _load_chat_template(self, chat_template: Optional[str]):
        tokenizer = self.tokenizer

        if chat_template is not None:
            try:
                with open(chat_template, "r") as f:
                    tokenizer.chat_template = f.read()
            except OSError as e:
                JINJA_CHARS = "{}\n"
                if not any(c in chat_template for c in JINJA_CHARS):
                    msg = (
                        f"The supplied chat template ({chat_template}) "
                        f"looks like a file path, but it failed to be "
                        f"opened. Reason: {e}"
                    )
                    raise ValueError(msg) from e

                # If opening a file fails, set chat template to be args to
                # ensure we decode so our escape are interpreted correctly
                tokenizer.chat_template = codecs.decode(chat_template, "unicode_escape")

            logger.info("Using supplied chat template:\n%s", tokenizer.chat_template)
        elif tokenizer.chat_template is not None:
            logger.info("Using default chat template:\n%s", tokenizer.chat_template)
        else:
            logger.warning("No chat template provided. Chat API will not work.")

    def _parse_chat_message_content_parts(
        self,
        role: str,
        parts: Iterable[ChatCompletionContentPartParam],
    ) -> ChatMessageParseResult:
        texts: List[str] = []

        for part in parts:
            part_type = part["type"]
            if part_type == "text":
                text = cast(ChatCompletionContentPartTextParam, part)["text"]

                texts.append(text)
            else:
                raise NotImplementedError(f"Unknown part type: {part_type}")

        text_prompt = "\n".join(texts)

        messages = [ConversationMessage(role=role, content=text_prompt)]

        return ChatMessageParseResult(messages=messages)

    def _parse_chat_message_content(
        self,
        message: ChatCompletionMessageParam,
    ) -> ChatMessageParseResult:
        role = message["role"]
        content = message.get("content")

        if content is None:
            return ChatMessageParseResult(messages=[])
        if isinstance(content, str):
            messages = [ConversationMessage(role=role, content=content)]
            return ChatMessageParseResult(messages=messages)

        return self._parse_chat_message_content_parts(role, content)

    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Optional[Request] = None
    ) -> Union[ErrorResponse, AsyncGenerator[str, None], ChatCompletionResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        ChatCompletion API.

        NOTE: Currently we do not support the following feature:
            - function_call (Users should implement this by themselves)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        try:
            conversation: List[ConversationMessage] = []

            for msg in request.messages:
                chat_parsed_result = self._parse_chat_message_content(msg)

                conversation.extend(chat_parsed_result.messages)

            prompt = self.tokenizer.apply_chat_template(
                conversation=conversation,
                tokenize=False,
                add_generation_prompt=request.add_generation_prompt,
            )
        except Exception as e:
            logger.error("Error in applying chat template from request: %s", e)
            return self.create_error_response(str(e))

        request_id = f"cmpl-{random_uuid()}"
        try:
            sampling_params = request.to_sampling_params()
        except ValueError as e:
            return self.create_error_response(str(e))

        result_generator = self.engine.generate(
            request_id,
            prompt,
            sampling_params,
        )
        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request, result_generator, request_id, conversation
            )
        else:
            try:
                return await self.chat_completion_full_generator(
                    request, raw_request, result_generator, request_id, conversation
                )
            except ValueError as e:
                # TODO: Use a sarathi-specific Validation Error
                return self.create_error_response(str(e))

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        else:
            return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
    ) -> AsyncGenerator[str, None]:
        model_name = self.served_model_names[0]
        created_time = int(time.time())
        chunk_object_type = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        previous_text = ""
        previous_num_tokens = 0

        try:
            async for res in result_generator:
                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)
                    choice_data = ChatCompletionResponseStreamChoice(
                        delta=DeltaMessage(role=role), finish_reason=None
                    )
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    if request.stream_options and request.stream_options.include_usage:
                        chunk.usage = None
                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content = ""
                        if (
                            conversation
                            and conversation[-1].get("content")
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"]

                        if last_msg_content:
                            choice_data = ChatCompletionResponseStreamChoice(
                                delta=DeltaMessage(content=last_msg_content),
                                finish_reason=None,
                            )
                            chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                object=chunk_object_type,
                                created=created_time,
                                choices=[choice_data],
                                model=model_name,
                            )
                            if (
                                request.stream_options
                                and request.stream_options.include_usage
                            ):
                                chunk.usage = None
                            data = chunk.model_dump_json(exclude_unset=True)
                            yield f"data: {data}\n\n"
                    first_iteration = False

                delta_text = res.text[len(previous_text) :]
                previous_text = res.text
                previous_num_tokens = len(res.token_ids)

                if (
                    request.tool_choice
                    and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
                ):
                    delta_message = DeltaMessage(
                        tool_calls=[
                            ToolCall(
                                function=FunctionCall(
                                    name=request.tool_choice.function.name,
                                    arguments=delta_text,
                                )
                            )
                        ]
                    )
                else:
                    delta_message = DeltaMessage(content=delta_text)

                # Send token-by-token response for each request.n
                num_prompt_tokens = len(res.prompt_token_ids)
                choice_data = ChatCompletionResponseStreamChoice(
                    delta=delta_message, finish_reason=res.finish_reason
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[choice_data],
                    model=model_name,
                )
                if request.stream_options and request.stream_options.include_usage:
                    chunk.usage = None
                data = chunk.model_dump_json(exclude_unset=True)
                yield f"data: {data}\n\n"

            if request.stream_options and request.stream_options.include_usage:
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=previous_num_tokens,
                    total_tokens=num_prompt_tokens + previous_num_tokens,
                )

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request],
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        conversation: List[ConversationMessage],
    ) -> Union[ErrorResponse, ChatCompletionResponse]:

        model_name = self.served_model_names[0]
        created_time = int(time.time())
        final_res: Optional[RequestOutput] = None

        async for res in result_generator:
            if raw_request is not None and await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await self.engine.abort(request_id)
                return self.create_error_response("Client disconnected")
            final_res = res
        assert final_res is not None

        choices: List[ChatCompletionResponseChoice] = []

        role = self.get_chat_request_role(request)

        if (
            request.tool_choice
            and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
        ):
            message = ChatMessage(
                role=role,
                content="",
                tool_calls=[
                    ToolCall(
                        function=FunctionCall(
                            name=request.tool_choice.function.name, arguments=res.text
                        )
                    )
                ],
            )
        elif not request.tool_choice or request.tool_choice == "none":
            message = ChatMessage(role=role, content=res.text)

        choice_data = ChatCompletionResponseChoice(
            message=message, finish_reason=res.finish_reason
        )
        choices.append(choice_data)

        if request.echo:
            last_msg_content = ""
            if (
                conversation
                and conversation[-1].get("content")
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"]

            for choice in choices:
                full_message = last_msg_content + choice.message.content
                choice.message.content = full_message

        num_prompt_tokens = len(final_res.prompt_token_ids)
        num_generated_tokens = len(res.token_ids)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

        return response
