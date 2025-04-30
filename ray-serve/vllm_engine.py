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
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import (
    BaseModelPath,
    LoRAModulePath,
    PromptAdapterPath,
    OpenAIServingModels,
)
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

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 5,
    },
    max_ongoing_requests=10,
)
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
        self.openai_serving_completion = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    async def _initialize_serving(self):
        """Initialize serving components if not already done."""
        if not self.openai_serving_chat or not self.openai_serving_completion:
            model_config = await self.engine.get_model_config()
            models = OpenAIServingModels(
                self.engine,
                model_config,
                [
                    BaseModelPath(
                        name=self.engine_args.model, model_path=self.engine_args.model
                    )
                ],
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
            )
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                models,
                self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format="auto",
            )
            self.openai_serving_completion = OpenAIServingCompletion(
                self.engine,
                model_config,
                models,
                request_logger=self.request_logger,
            )

    @app.get("/v1")
    async def get_v1_status(self):
        """Return API status and available endpoints."""
        return JSONResponse(
            content={
                "status": "ok",
                "message": "vLLM OpenAI-compatible API",
                "endpoints": [
                    "/v1/chat/completions",
                    "/v1/completions",
                    "/v1/models",
                ],
                "model": self.engine_args.model,
            },
            status_code=200,
        )

    @app.get("/v1/models")
    async def list_models(self):
        """List available models, compatible with OpenAI API."""
        await self._initialize_serving()
        model_config = await self.engine.get_model_config()
        model_name = self.engine_args.model
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "vLLM",
                    }
                ],
            },
            status_code=200,
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible chat completion endpoint."""
        await self._initialize_serving()
        logger.info(f"Chat completion request: {request}")
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

    @app.post("/v1/completions")
    async def create_completion(self, request: CompletionRequest, raw_request: Request):
        """OpenAI-compatible text completion endpoint."""
        await self._initialize_serving()
        logger.info(f"Text completion request: {request}")
        generator = await self.openai_serving_completion.create_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, CompletionResponse)
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
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    # if "accelerator" in cli_args.keys():
    #     accelerator = cli_args.pop("accelerator")
    # else:
    #     accelerator = "GPU"
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 8})  # for the deployment replica
    # for i in range(tp):
    #     pg_resources.append({"CPU": 8, accelerator: 2})  # for the vLLM actors

    # return VLLMDeployment.options(
    #     placement_group_bundles=pg_resources, placement_group_strategy="STRICT_PACK"
    # ).bind(
    #     engine_args,
    #     parsed_args.response_role,
    #     parsed_args.lora_modules,
    #     parsed_args.prompt_adapters,
    #     cli_args.get("request_logger"),
    #     parsed_args.chat_template,
    # )

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



# Erstelle eine Serve-App (bzw. -Deployment) mit den gewünschten Parametern
env_args = {
        "model": os.environ["MODEL_ID"],
        "gpu-memory-utilization": os.environ["GPU_MEMORY_UTILIZATION"],
        "download-dir": os.environ["DOWNLOAD_DIR"],
        "max-model-len": os.environ["MAX_MODEL_LEN"],
        "tensor-parallel-size": os.environ["TENSOR_PARALLELISM"],
        "pipeline-parallel-size": os.environ["PIPELINE_PARALLELISM"],
        "trust_remote_code": "",
        "enable-reasoning": "",
        "reasoning-parser": "deepseek_r1",
        # "cpu_offload_gb": "4"
        # "max-num-seqs": os.environ["MAX_NUM_SEQS"],
        # "enforce-eager": "True",
        # Falls du METRICS deaktivieren willst (nicht empfohlen), könntest du:
        # "disable-metrics": "True"
    }

if int(os.environ["MAX_MODEL_LEN"]) > 32768:
    env_args["rope-scaling"] = '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'


if os.environ.get("ENABLE_CHUNKED_PREFILL", "False").lower() == "true":
    env_args["enable-chunked-prefill"] = "true"  # flag without value

model = build_app(env_args)