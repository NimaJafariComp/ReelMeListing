"""Diffusers adapter for reproducible InstructPix2Pix candidate generation."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from PIL import Image, ImageOps

from listing_to_reel.core.environment import assert_device_available, capabilities_from_torch
from listing_to_reel.editing.models import EditRequest, GeneratedCandidate


class DiffusionDependenciesMissingError(RuntimeError):
    """Raised when a requested runtime lacks the optional diffusion stack."""


class InstructPix2PixEditor:
    """A provider-specific adapter kept outside the orchestration layer."""

    def _dependencies(self):
        try:
            import torch
            from diffusers import (
                EulerAncestralDiscreteScheduler,
                StableDiffusionInstructPix2PixPipeline,
            )
            from huggingface_hub import HfApi
        except ImportError as error:
            raise DiffusionDependenciesMissingError(
                "Install the image-editing stack with "
                "`uv sync --extra mps --python 3.12` for an MPS preview, or install "
                "the gpu extra on the CUDA worker."
            ) from error
        return torch, EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline, HfApi

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as candidate_file:
            for chunk in iter(lambda: candidate_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _resize_for_model(image: Image.Image, maximum_dimension: int) -> Image.Image:
        width, height = image.size
        scale = min(1.0, maximum_dimension / max(width, height))
        resized_width = max(8, round(width * scale / 8) * 8)
        resized_height = max(8, round(height * scale / 8) * 8)
        return image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    @staticmethod
    def _peak_memory_bytes(torch_module, device: str) -> int | None:
        if device == "cuda":
            return int(torch_module.cuda.max_memory_allocated())
        mps = getattr(torch_module, "mps", None)
        if device == "mps" and mps is not None and hasattr(mps, "current_allocated_memory"):
            return int(mps.current_allocated_memory())
        return None

    def generate(
        self, request: EditRequest, candidate_dir: Path
    ) -> tuple[str, list[GeneratedCandidate]]:
        """Generate candidate images and return the immutable resolved model revision."""
        torch, scheduler_type, pipeline_type, hf_api_type = self._dependencies()
        device = request.runtime_profile.device.value
        assert_device_available(
            device=request.runtime_profile.device,
            capabilities=capabilities_from_torch(torch),
        )

        requested_revision = request.configuration.model_revision or "main"
        resolved_revision = hf_api_type().model_info(
            request.configuration.model_id, revision=requested_revision
        ).sha
        torch_dtype = (
            torch.float16 if request.configuration.torch_dtype == "float16" else torch.float32
        )
        pipeline = pipeline_type.from_pretrained(
            request.configuration.model_id,
            revision=resolved_revision,
            torch_dtype=torch_dtype,
            use_safetensors=True,
        )
        pipeline.scheduler = scheduler_type.from_config(pipeline.scheduler.config)
        pipeline = pipeline.to(device)
        if request.runtime_profile.attention_slicing:
            pipeline.enable_attention_slicing()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with Image.open(request.source_path) as source_file:
            source_image = ImageOps.exif_transpose(source_file).convert("RGB")
        source_image = self._resize_for_model(source_image, request.configuration.max_dimension)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[GeneratedCandidate] = []

        for index in range(request.configuration.candidate_count):
            candidate_seed = request.seed + index
            generator_device = "cuda" if device == "cuda" else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(candidate_seed)
            started = time.perf_counter()
            output = pipeline(
                prompt=request.instruction,
                image=source_image,
                num_inference_steps=request.configuration.num_inference_steps,
                guidance_scale=request.configuration.guidance_scale,
                image_guidance_scale=request.configuration.image_guidance_scale,
                generator=generator,
            )
            artifact_path = candidate_dir / f"candidate-{index + 1:02d}.jpg"
            output.images[0].convert("RGB").save(
                artifact_path,
                format="JPEG",
                quality=request.configuration.output_quality,
                optimize=False,
                progressive=False,
            )
            candidates.append(
                GeneratedCandidate(
                    candidate_index=index + 1,
                    seed=candidate_seed,
                    artifact_path=str(artifact_path),
                    sha256=self._sha256(artifact_path),
                    wall_clock_seconds=time.perf_counter() - started,
                    peak_device_memory_bytes=self._peak_memory_bytes(torch, device),
                )
            )
        return resolved_revision, candidates
