import argparse
import logging
import os
import re
import time

from rich.logging import RichHandler
from tqdm import tqdm

from evaluation.prompts.ext_ans import demo_prompt
from utilities import read_json, save_json


def verify_extraction(extraction):
    if extraction is None:
        return False
    if isinstance(extraction, str):
        extraction = extraction.strip()
    if extraction == "":
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer_local_rules(response: str, problem: dict) -> str:
    """
    A safe local-only extractor used when --quick_extract is enabled.
    This never calls any LLM.
    """
    question_type = problem.get("question_type")
    answer_type = problem.get("answer_type")
    choices = problem.get("choices") or []
    pid = problem.get("pid", "")

    if response is None:
        return ""
    if isinstance(response, str):
        response = response.strip()
    if response == "":
        return ""

    # If it's multi-choice and the response is exactly one of the choices
    if question_type == "multi_choice" and response in choices:
        return response

    # Try direct integer/float casting if answer_type indicates
    if answer_type == "integer":
        try:
            return str(int(response))
        except Exception:
            pass

    if answer_type == "float":
        try:
            return str(float(response))
        except Exception:
            pass

    # Common patterns:
    # The answer is "text". -> text
    try:
        m = re.search(r'The answer is\s*"(.*)"\.', response)
        if m:
            return m.group(1).strip()
    except Exception:
        pass

    # The answer is: XXX
    try:
        m = re.search(r"(?i)the answer is\s*[:：]\s*([^\n\r]+)", response)
        if m:
            return m.group(1).strip().strip(".")
    except Exception:
        pass

    # Answer: XXX
    try:
        m = re.search(r"(?i)^answer\s*[:：]\s*([^\n\r]+)", response)
        if m:
            return m.group(1).strip().strip(".")
    except Exception:
        pass

    # If multi-choice: try to find a single letter choice (A/B/C/D/...)
    if question_type == "multi_choice" and choices:
        # e.g., "Answer: C" or "Option C"
        try:
            m = re.search(r"(?i)\b([A-Z])\b", response)
            if m and m.group(1) in choices:
                return m.group(1)
        except Exception:
            pass

    # Fallback: return empty to mark extraction failure (score script may treat as incorrect)
    logging.debug(f"[{pid}] quick_extract failed to parse response.")
    return ""


def build_llm_extractor(args):
    """
    Build an LLM client+wrapper only when needed (i.e., not quick_extract).
    Priority:
      1) OpenAI official API via OPENAI_API_KEY
      2) Azure OpenAI via AZURE_* env vars / args
    """
    # Try OpenAI official first
    openai_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            from models import gpt

            client = OpenAI(api_key=openai_key)
            model_name = args.openai_model
            logging.info(f"Using OpenAI API for extraction, model={model_name}")
            return gpt.GPT_Model(client=client, model=model_name)
        except Exception as e:
            logging.warning("Failed to initialize OpenAI client for extraction, will try Azure if available.")
            logging.warning(str(e))

    # Fallback to Azure if configured
    if args.azure_openai_api_endpoint and args.azure_openai_api_key and args.azure_openai_api_version and args.azure_openai_model:
        try:
            from openai import AzureOpenAI
            from models import gpt

            client = AzureOpenAI(
                azure_endpoint=args.azure_openai_api_endpoint,
                api_key=args.azure_openai_api_key,
                api_version=args.azure_openai_api_version,
            )
            logging.info(f"Using Azure OpenAI for extraction, model={args.azure_openai_model}")
            return gpt.GPT_Model(client=client, model=args.azure_openai_model)
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI client init failed: {e}") from e

    raise RuntimeError(
        "No LLM credentials found for non-quick extraction. "
        "Set OPENAI_API_KEY (recommended) or AZURE_OPENAI_API_* env vars."
    )


def extract_answer_with_llm(model, response, problem) -> str:
    """
    General extraction using an LLM. This matches the original behavior:
    create a prompt and ask the model to output Extracted answer.
    """
    query = problem.get("query", "")
    pid = problem.get("pid", "")

    if response is None:
        return ""
    if isinstance(response, str):
        response = response.strip()
    if response == "":
        return ""

    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = model.get_response(user_prompt=full_prompt)
        if isinstance(extraction, str):
            return extraction.strip()
        return str(extraction).strip()
    except Exception as e:
        logging.info(f"Error in extracting answer for problem: {pid} with response: {response}")
        logging.info(e)
        return ""


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--results_file_path', type=str, default='answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The max number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')

    # OpenAI (recommended for non-quick extraction)
    parser.add_argument('--openai_api_key', type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument('--openai_model', type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        help="Model for extraction when using OpenAI API. Default: gpt-4o-mini (cheap).")

    # Azure OpenAI (optional fallback)
    parser.add_argument('--azure_openai_api_endpoint', type=str, default=os.getenv("AZURE_OPENAI_API_ENDPOINT"))
    parser.add_argument('--azure_openai_api_key', type=str, default=os.getenv("AZURE_OPENAI_API_KEY"))
    parser.add_argument('--azure_openai_api_version', type=str, default=os.getenv("AZURE_OPENAI_API_VERSION"))
    parser.add_argument('--azure_openai_model', type=str, default=os.getenv("AZURE_OPENAI_MODEL"))

    # retry for LLM extraction
    parser.add_argument('--max_retries', type=int, default=6)
    parser.add_argument('--retry_sleep', type=int, default=10)

    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Extract Answers - Start")
    args = parse_args()

    label = args.response_label

    logging.info(f"Reading {args.results_file_path}...")
    results = read_json(args.results_file_path)

    full_pids = list(results.keys())

    # Determine which problems to run
    skip_pids = []
    if not args.rerun:
        for pid, problem in results.items():
            extraction = problem.get('extraction')
            if extraction is not None and verify_extraction(extraction):
                skip_pids.append(problem.get('pid', pid))

    if args.rerun:
        test_pids = full_pids
    else:
        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid extractions. Skipping these problems..."
            )
        test_pids = [pid for pid in full_pids if results[pid].get('pid', pid) not in skip_pids]

    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.info(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")

    # Build LLM extractor only if needed
    llm_model = None
    if not args.quick_extract:
        llm_model = build_llm_extractor(args)

    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        if label not in problem:
            raise KeyError(f"response_label '{label}' not found in problem {pid} keys={list(problem.keys())}")

        response = problem[label]

        # Quick extract: local only
        if args.quick_extract:
            extraction = extract_answer_local_rules(response, problem)
        else:
            # LLM extraction with retry on rate-limit
            attempt = 0
            while True:
                try:
                    extraction = extract_answer_with_llm(llm_model, response, problem)
                    break
                except Exception as e:
                    attempt += 1
                    msg = str(e).lower()
                    if ("rate limit" in msg or "429" in msg) and attempt <= args.max_retries:
                        logging.warning(
                            f"[{problem.get('pid', pid)}] Rate limit during extraction. "
                            f"Retry {attempt}/{args.max_retries} after {args.retry_sleep}s"
                        )
                        time.sleep(args.retry_sleep)
                        continue
                    logging.error(f"[{problem.get('pid', pid)}] Extraction failed: {e}")
                    extraction = ""
                    break

        results[pid]['extraction'] = extraction

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, args.results_file_path)
            logging.info(f"Saved results to {args.results_file_path}")

    logging.info("MathVista: Extract Answers - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
