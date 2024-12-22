import g4f
from g4f.Provider import MetaAI
from typing import Dict, List, Optional
from langchain_core.runnables.base import Runnable,RunnableMap, Callable
from typing import Optional, List, Mapping, Any, Sequence, Union, Type, Literal
from langchain.llms.base import BaseLLM
from pydantic import BaseModel
from langchain_core.tools import tool
from langchain_core.tools.base import BaseTool
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_experimental.utilities import PythonREPL
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, BaseModel
from langchain.agents import Tool
from langchain.agents import load_tools, initialize_agent
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import json
import uuid
import logging
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.tools import StructuredTool

class LlamaFreeChat(BaseChatModel):

    model_name: str = Field(alias="model")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    tools: Optional[List[Any]] = []
    tool_tracker: Optional[List[str]] = None
    max_retries: int = 2
    #type: Dict[Any,Any] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        
        messages_list=list()
        
        for message in messages:
            if isinstance(message,SystemMessage):
                messages_list.append({"role":"system","content":message.content})
            elif isinstance(message,HumanMessage):
                messages_list.append({"role":"user","content":message.content})
            else:
                messages_list.append({"role":"assistant","content":message.content})

        response = g4f.ChatCompletion.create(
            model=self.model_name,
            provider=MetaAI,
            messages=messages_list
        )

        response_content = response.strip()

        #Check if the response contains a tool call and create a construct for detection
        additional_kwargs = {}
        try:
            #if "\"tool\":" in response_content:
            #print(f"This is response content:\n{response_content}")
            response_json = JsonOutputParser().parse(response_content)
            
            print(f"JSON response: {response_json}")
            if "tool" in response_json and "tool_input" in response_json:
                tool_call = {
                    "index":0,
                    "id": f"call_{uuid.uuid4().hex}",
                    "function": {
                        "arguments": json.dumps(response_json["tool_input"]),
                        "name": response_json["tool"],
                    },
                    "type": "function",
                }
                additional_kwargs["tool_calls"] = [tool_call]
                response_content = self.execute_tool_logic(response_json["tool"],response_json["tool_input"])
        except Exception as e:
            pass
        
        ai_message = AIMessage(
            content = response_content,
            additional_kwargs = additional_kwargs
        )
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])
    
    
    def execute_tool_logic(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        
        if self.tool_tracker is not None:
            if tool_name in self.tool_tracker:
                tool = getattr(tool_name)
                result = tool.invoke(tool_args)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
        else:
            raise ValueError(f"Unknown tool: {tool_name}")


    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "any", "none"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function,
                "auto" to automatically determine which function to call
                with the option to not call any function, "any" to enforce that some
                function is called, or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "any", "none")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, dict) and (len(formatted_tools) != 1):
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, dict) and (
                formatted_tools[0]["function"]["name"]
                != tool_choice["function"]["name"]
            ):
                raise ValueError(
                    f"Tool choice {tool_choice} was specified, but the only "
                    f"provided tool was {formatted_tools[0]['function']['name']}."
                )
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        "tool_choice can only be True when there is one tool. Received "
                        f"{len(tools)} tools."
                    )
                tool_name = formatted_tools[0]["function"]["name"]
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_name},
                }

            kwargs["tool_choice"] = tool_choice
        
        self.tool_tracker = [tool['function']['name'] for tool in formatted_tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "llama-3.1-70b via g4f"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications
            "model_name": self.model_name,
        }