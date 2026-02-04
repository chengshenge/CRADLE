#!/usr/bin/env python3
import argparse
import os
import sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vsk_root", required=True, help="Path to VisualSketchpad repo root")
    ap.add_argument("--task_input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--task_type", default="vision", choices=["vision", "math", "geo"])
    ap.add_argument("--task_name", default=None)

    ap.add_argument("--api_key", default="")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_reply", type=int, default=10)

    ap.add_argument("--som_address", default="")
    ap.add_argument("--gd_address", default="")
    ap.add_argument("--da_address", default="")

    args = ap.parse_args()

    # Autogen 不用 docker（VisualSketchpad README 也建议这样配）
    os.environ["AUTOGEN_USE_DOCKER"] = "False"
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    vsk_agent_dir = os.path.join(os.path.abspath(args.vsk_root), "agent")
    if not os.path.isdir(vsk_agent_dir):
        raise FileNotFoundError(f"VisualSketchpad agent dir not found: {vsk_agent_dir}")

    # 让 import main / tools / agent 等都指向 VisualSketchpad/agent 目录
    sys.path.insert(0, vsk_agent_dir)

    # 导入 VisualSketchpad 的 main.py（里面有 run_agent）
    import main as vsk_main

    # 关键：VisualSketchpad 的 config.py 里通常会写死 OPENAI_API_KEY + llm_config，
    # 所以这里强制 patch main.llm_config，确保用你当前传入的 key/model。
    try:
        cfg0 = vsk_main.llm_config["config_list"][0]
        cfg0["model"] = args.model
        cfg0["temperature"] = float(args.temperature)
        cfg0["api_key"] = os.environ.get("OPENAI_API_KEY", "")
    except Exception as e:
        raise RuntimeError(f"Failed to patch vsk_main.llm_config: {e}")

    # max_reply 也在 main.py 顶层 import 进来了，直接 patch 变量即可
    vsk_main.MAX_REPLY = int(args.max_reply)

    # 如果你传了视觉专家地址，就 patch tools 里的 gradio client
    if args.som_address or args.gd_address or args.da_address:
        from gradio_client import Client
        import tools as vsk_tools
        if args.som_address:
            vsk_tools.som_client = Client(args.som_address)
        if args.gd_address:
            vsk_tools.gd_client = Client(args.gd_address)
        if args.da_address:
            vsk_tools.da_client = Client(args.da_address)

    vsk_main.run_agent(
        args.task_input,
        args.output_dir,
        task_type=args.task_type,
        task_name=args.task_name
    )

if __name__ == "__main__":
    main()
