from mlx_lm.evaluate import MLXLM, chat_template_fn
from models.results import EvalResult
import lm_eval
from loguru import logger
import mlx.core as mx

def run_mlx_eval(
    model_name: str,
    max_tokens: int = 1024,
    temp: float = 0.6,
    seed: int = 42
) -> list[EvalResult]:

    mx.random.seed(seed)

    # lm = MLXLM(
    #     model_name,
    #     max_tokens=max_tokens,
    # )

    # MLXLM.apply_chat_template = chat_template_fn()

    raw = lm_eval.simple_evaluate(
        model="mlxlm",
        model_args=f"path_or_hf_repo={model_name}",
        # tasks=["arc_easy", "hellaswag", "mmlu_pro"],
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=1,
    )

    if raw["results"] is None:
        logger.error("bruvvv")

    results = raw["results"]

    eval_results: list[EvalResult] = []
    for model_name, metrics in results:
        eval_result = EvalResult(
            task = metrics.get("alias"),
            accuracy=metrics.get("acc_norm,none"),
            stderr=metrics.get("acc_norm_stderr,none")
        )
        eval_results.append(eval_result)


    return eval_results


if __name__ == "__main__":
    MODEL_NAME = "mlx-community/Qwen3.5-9B-MLX-4bit"
    PROMPT = "Explain what a dbt model is in one paragraph."
    run_mlx_eval(MODEL_NAME)
