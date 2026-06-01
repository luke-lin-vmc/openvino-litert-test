# Reference: https://ai.google.dev/edge/litert/next/intel#5-aot-compile-optional

import argparse
from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.intel_openvino import target as intel_target


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile a TFLite model with LiteRT AOT compiler."
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the input TFLite model, for example: model.tflite",
    )

    parser.add_argument(
        "--output_dir",
        default=".",
        help="Output directory for compiled artifacts. Default: current directory",
    )

    parser.add_argument(
        "--soc_model",
        choices=["PTL", "LNL"],
        default=None,
        help="Intel SoC model to compile for. If not specified, compile every registered backend/target",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.soc_model is None:
        # Compile for every registered backend/target by default.
        aot_compile.aot_compile(
            args.model,
            output_dir=args.output_dir,
            keep_going=True,
        )
    else:
        # Compile for a single Intel NPU target.
        soc_model = getattr(intel_target.SocModel, args.soc_model)

        aot_compile.aot_compile(
            args.model,
            output_dir=args.output_dir,
            target=intel_target.Target(soc_model=soc_model),
        )


if __name__ == "__main__":
    main()