import argparse
import sys

from agent import AgentConfig, run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local coding agent on a single prompt.")
    parser.add_argument("prompt", help="The task/prompt to give the agent.")
    parser.add_argument("--model", default=None, help="Override the Groq model name for this run.")
    parser.add_argument("--thread-id", default=None, help="Reuse a thread id to continue a previous session.")
    args = parser.parse_args()

    try:
        config = AgentConfig.from_env()
    except Exception as e:  # noqa: BLE001
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.model:
        config = config.model_copy(update={"model_name": args.model})

    result = run_agent(args.prompt, config=config, thread_id=args.thread_id)

    print("\n=== RESULT ===")
    print(f"status:     {result.status}")
    print(f"iterations: {result.iterations}")
    print(f"thread_id:  {result.thread_id}")
    if result.error:
        print(f"error:      {result.error}")

    print("\n--- output ---")
    print(result.output)

    if result.tool_calls:
        print("\n--- tool calls ---")
        for log in result.tool_calls:
            flag = "OK" if log.success else "FAIL"
            print(f"[{flag}] {log.tool_name}({log.args}) -> {log.result[:200]}")

    if result.status == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
