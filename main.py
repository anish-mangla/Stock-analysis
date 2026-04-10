"""
main.py

Thin orchestrator for the stock-news analysis workflow.

Pipeline:
1. Select tickers
2. Compose prompt
3. Execute model(s)
4. Save outputs

This file intentionally stays thin and only connects the components.
"""

from ticker_selection import get_selected_tickers
from prompt_composition import compose_prompt
from model_execution import run_model
from response_storage import save_run_outputs


MODELS_TO_RUN = ["claude"]


def main() -> None:
    ticker_data = get_selected_tickers()

    if not ticker_data["tickers"]:
        print("No tickers met the selection rule. Nothing to do.")
        return

    prompt_bundle = compose_prompt(ticker_data)

    for model_name in MODELS_TO_RUN:
        print(f"Running model: {model_name}")
        model_result = run_model(model_name, prompt_bundle, ticker_data)
        save_run_outputs(model_name, ticker_data, prompt_bundle, model_result)
        print(
            f"Finished {model_name} | "
            f"error={model_result['error']} | "
            f"estimated_cost={model_result['metadata'].get('estimated_cost_usd')}"
        )


if __name__ == "__main__":
    main()