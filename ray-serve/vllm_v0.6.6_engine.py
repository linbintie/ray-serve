import os

from typing import Dict, Optional, List
import logging

from fastapi import FastAPI, Response
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

# For VLLM Metrics
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from ray import serve

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath, PromptAdapterPath
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

logger = logging.getLogger("ray.serve")

app = FastAPI()

@app.get("/metrics")
def metrics():
    """
    Dieser Endpoint gibt alle Metriken aus der Default-Registry
    im Prometheus-Format zurück. Wenn vLLM seine Metriken korrekt
    registriert, erscheinen sie hier unter `vllm:...`.
    
    Wichtig:
    - Wenn du mehrere Ray-Replikas (oder mehrere Prozesse) hast,
      brauchst du evtl. die Multiprozess-Sammlung. Siehe unten.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            # Determine the name of the served model for the OpenAI client.
            if self.engine_args.served_model_name is not None:
                served_model_names = self.engine_args.served_model_name
            else:
                served_model_names = [self.engine_args.model]
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                served_model_names,
                self.response_role,
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
                request_logger=self.request_logger,
                chat_template=self.chat_template
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())




def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        logger.info(f"Processing argument: --{key} with value: {value}")

        if value is True:  # Håndter boolske flagg satt til True
            arg_strings.append(f"--{key}")
        elif value not in (None, "None"):  # Ignorer None eller 'None' som streng
            arg_strings.extend([f"--{key}", str(value)])
        else:
            arg_strings.append(f"--{key}")

    parsed_args = parser.parse_args(args=arg_strings)

    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    if "accelerator" in cli_args.keys():
        accelerator = cli_args.pop("accelerator")
    else:
        accelerator = "GPU"
    parsed_args = parse_vllm_args(cli_args)
    logger.info(f"Parsed args: {parsed_args}") # ADDED LOGGING
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    logger.info(f"Engine args: {engine_args}") # ADDED LOGGING
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, accelerator: 1})  # for the vLLM actors

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    return VLLMDeployment.bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )