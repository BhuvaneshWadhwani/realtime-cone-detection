"""
Build a TensorRT FP16 engine from the exported ONNX model.
Run this script whenever you need to rebuild the engine e.g. after a
driver/TensorRT update or when moving to a new machine.

Usage:
    python3 build_trt_engine.py
"""
from pathlib import Path
import tensorrt as trt

ONNX_PATH   = Path("outputs/rtv4_hgnetv2_s_fsoco/best_stg2.onnx")
ENGINE_PATH = Path("outputs/rtv4_hgnetv2_s_fsoco/best_stg2_fp16.trt")
WORKSPACE_GB = 4


def build():
    if not ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX model not found: {ONNX_PATH}\n"
                                "Run tools/deployment/export_onnx.py first.")

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing ONNX model: {ONNX_PATH}")
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed — see errors above.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GB << 30)
    config.set_flag(trt.BuilderFlag.FP16)

    # Optimization profile — fixed batch size 1, fixed resolution 640x640
    profile = builder.create_optimization_profile()
    profile.set_shape("images",            min=(1,3,640,640), opt=(1,3,640,640), max=(1,3,640,640))
    profile.set_shape("orig_target_sizes", min=(1,2),         opt=(1,2),         max=(1,2))
    config.add_optimization_profile(profile)

    print("Building TensorRT FP16 engine — this takes a few minutes…")
    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        raise RuntimeError("Engine build failed — check TensorRT warnings above.")

    ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ENGINE_PATH, "wb") as f:
        f.write(engine_bytes)

    size_mb = ENGINE_PATH.stat().st_size / 1024 / 1024
    print(f"Done — engine saved to {ENGINE_PATH}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    build()
