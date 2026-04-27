#!/usr/bin/env python3
"""Run SWE-bench Verified evaluation with the LangGraph agent."""

import os
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from _utils import agent_cfg_from_setup, build_generate_fn, load_setup_config
from agentic_code_rag.agent.graph import build_agent_graph
from agentic_code_rag.agent.llm_provider import create_llm
from agentic_code_rag.agent.state import AgentState
from agentic_code_rag.benchmarks.swebench import run_swebench
from agentic_code_rag.draco_wrapper import DraCoIndex, DraCoToolAPI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--output", default="results/swebench")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--run_harness", action="store_true",
                        help="Invoke official Docker harness after generating patches")
    parser.add_argument("--repo_cache", default="/tmp/swebench_repos",
                        help="Directory to cache cloned repos")
    parser.add_argument("--trajectory_dir", default="results/trajectories")
    parser.add_argument("--setup", default=None, help="Path to a setup YAML (configs/setup_*.yaml)")
    parser.add_argument("--max_tokens", type=int, default=None,
                        help="Max output tokens (default: llm_provider default of 16384)")
    parser.add_argument("--log_level", default=os.environ.get("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )

    setup = load_setup_config(args.setup)
    agent_cfg = agent_cfg_from_setup(setup)
    mode = setup.get("mode", "agent")

    llm_kwargs = {}
    if args.max_tokens is not None:
        llm_kwargs["max_tokens"] = args.max_tokens
    llm = create_llm(args.provider, args.model, base_url=args.base_url, **llm_kwargs)
    repo_cache = Path(args.repo_cache)
    repo_cache.mkdir(parents=True, exist_ok=True)
    trajectory_dir = Path(args.trajectory_dir)

    def _prepare_repo(instance: dict) -> Path:
        instance_id = instance["instance_id"]
        repo_name = instance.get("repo", "")
        base_commit = instance.get("base_commit", "HEAD")

        repo_dir = repo_cache / instance_id
        if not repo_dir.exists():
            clone_url = f"https://github.com/{repo_name}.git"
            subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, str(repo_dir)],
                check=True,
                timeout=600,
            )
        try:
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=str(repo_dir),
                check=True,
                timeout=60,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            # Shallow clone doesn't have this commit — fetch full history
            subprocess.run(
                ["git", "fetch", "--unshallow", "origin"],
                cwd=str(repo_dir),
                check=True,
                timeout=600,
            )
            subprocess.run(
                ["git", "checkout", base_commit],
                cwd=str(repo_dir),
                check=True,
                timeout=60,
            )
        return repo_dir

    if mode in ("zero_shot", "static_rag"):
        def agent_fn(instance: dict) -> str:
            repo_dir = _prepare_repo(instance)
            tool_api = None
            if mode == "static_rag":
                index = DraCoIndex(repo_dir)
                try:
                    index.build()
                except Exception:
                    pass
                tool_api = DraCoToolAPI(index, repo_dir)

            problem = instance.get("problem_statement", "")
            generate_fn = build_generate_fn(llm, setup, tool_api=tool_api, task_hint=problem[:300])
            prompt = (
                f"You are a software engineer fixing a bug in a Python repository.\n\n"
                f"Problem statement:\n{problem}\n\n"
                f"Produce a unified diff patch that fixes the issue. "
                f"Output ONLY the patch, nothing else."
            )
            return generate_fn(prompt)

    else:
        def agent_fn(instance: dict) -> str:
            repo_dir = _prepare_repo(instance)
            instance_id = instance["instance_id"]

            index = DraCoIndex(repo_dir)
            try:
                index.build()
            except Exception:
                pass

            tool_api = DraCoToolAPI(index, repo_dir)
            agent = build_agent_graph(
                tool_api, llm,
                config=agent_cfg if agent_cfg else None,
                trajectory_dir=trajectory_dir,
            )

            state = AgentState(
                task_text=instance.get("problem_statement", ""),
                repo_root=str(repo_dir),
                snapshot_id=instance_id,
            )
            result = agent.invoke(state)
            return result.get("proposed_patch", "") if isinstance(result, dict) else ""

    results = run_swebench(
        agent_fn=agent_fn,
        output_dir=Path(args.output),
        max_instances=args.max_instances,
        run_harness=args.run_harness,
    )
    if setup:
        print(f"Setup: {setup.get('name', args.setup)}")
    print("Results:", results)


if __name__ == "__main__":
    main()
