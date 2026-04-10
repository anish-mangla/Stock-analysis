Project Description

This project is a daily research pipeline for identifying whether recent stock price drops are logically explained by fresh negative fundamentals or are more likely temporary, sentiment-driven moves that may mean-revert.

The system begins with a quantitative filter that scans a universe of stocks and identifies names whose recent price action has weakened meaningfully over a short window. For the selected tickers, the system then generates a research prompt and sends it to one or more AI models that can search the web and analyze recent news. The purpose of the AI stage is not to predict the future directly, but to assess whether the recent drop appears justified by new, stock-specific negative information.

The output from each AI model is stored exactly as returned so that it can be evaluated later. After enough time has passed, the stored outputs can be enriched with future market data to test whether the model’s assessment would have been useful in practice. This allows comparison across models such as Claude, OpenAI, and others, both in terms of reasoning quality and real-world usefulness.

The goal is to create a repeatable, model-agnostic workflow for researching short-term price weakness through a combination of price-action screening, web-based news analysis, and later empirical evaluation.

High-Level Objective

The system is designed to answer the following practical question:

“When a stock drops materially over the last few trading days, is that drop logically explained by new, meaningful negative information, or does the move appear weakly explained and therefore more likely to be sentiment-driven or temporarily oversold?”

The system does not attempt to guarantee correct forecasts. Instead, it produces structured research records that can later be compared against actual future returns.

Functional Overview

The workflow has four main stages:

Identify candidate tickers from market data.
Convert those tickers into a prompt for an AI model.
Send the prompt to one or more AI models and collect the response.
Store the full response and metadata for later analysis.
System Design

The system should be organized around a central orchestrator. The orchestrator is responsible for running each component in order and passing outputs from one stage into the next.

+------------------------------------------------------+
|                   ORCHESTRATOR                       |
|------------------------------------------------------|
| Runs the daily workflow end-to-end                   |
| Coordinates each component                           |
| Handles model loop, saving, and logging              |
+------------------------+-----------------------------+
                         |
                         v
+------------------------------------------------------+
|          COMPONENT 1: TICKER SELECTION               |
|------------------------------------------------------|
| Runs Python market-data script                       |
| Computes recent price weakness metrics               |
| Filters tickers based on threshold rule              |
| Returns selected tickers and supporting stats        |
+------------------------+-----------------------------+
                         |
                         v
+------------------------------------------------------+
|          COMPONENT 2: PROMPT COMPOSITION             |
|------------------------------------------------------|
| Takes selected tickers and metadata                  |
| Builds the research prompt                           |
| Applies common prompt template                       |
| Produces final model-ready prompt text               |
+------------------------+-----------------------------+
                         |
                         v
+------------------------------------------------------+
|          COMPONENT 3: MODEL EXECUTION                |
|------------------------------------------------------|
| Sends prompt to target AI model                      |
| Uses model-specific API/client logic                 |
| Collects raw response                                |
| Optionally parses structured output                  |
+------------------------+-----------------------------+
                         |
                         v
+------------------------------------------------------+
|          COMPONENT 4: RESPONSE STORAGE               |
|------------------------------------------------------|
| Saves raw response exactly as returned               |
| Saves parsed/normalized representation               |
| Saves metadata such as model, prompt version, cost   |
| Organizes data for future evaluation                 |
+------------------------------------------------------+
Component Details
1. Orchestrator

The orchestrator is the top-level controller for the pipeline.

Its responsibilities are:

starting the workflow
invoking each component in the correct order
looping through multiple AI models if needed
ensuring all inputs and outputs are saved consistently
handling failures cleanly
preserving reproducibility

The orchestrator should be model-agnostic. It should not contain logic specific to Claude, OpenAI, or any other provider. Instead, it should call model-specific execution modules through a common interface.

Conceptually, the orchestrator does the following:

Start run
  -> run ticker selection
  -> compose prompt
  -> for each model:
       -> execute model call
       -> save response
End run
2. Component 1: Ticker Selection

This component runs the market-data logic that identifies candidate stocks.

In the current design, this is your Python script that:

downloads recent daily price data
computes typical price
measures the 5-trading-day change
sorts and filters names
returns only those that meet the required weakness threshold

The output of this component should not just be a plain ticker list. It should return structured information for each ticker, such as:

ticker symbol
absolute change in typical price
percentage change in typical price
date of last observation
rank among candidates

Example output:

{
  "run_date": "2026-04-09",
  "selection_rule": "5-day TP change <= -2.5%",
  "tickers": [
    {
      "ticker": "AAPL",
      "tp_change_abs": -5.12,
      "tp_change_pct": -0.031,
      "last_date": "2026-04-09",
      "rank": 1
    },
    {
      "ticker": "NVDA",
      "tp_change_abs": -8.44,
      "tp_change_pct": -0.028,
      "last_date": "2026-04-09",
      "rank": 2
    }
  ]
}

This structured output will later make the pipeline easier to audit and evaluate.

3. Component 2: Prompt Composition

This component takes the selected tickers and turns them into a model-ready prompt.

The purpose of this stage is to keep prompt creation separate from model execution. That way:

prompt versions can be tracked
prompt wording can be improved over time
all models can receive the same task definition
comparisons between models remain fair

The prompt composer should include:

the list of tickers
the recent price-drop context
the research objective
the classification categories
the requirement for citations
the expected output structure

The prompt should instruct the model to determine whether the recent weakness is:

clearly explained by fresh negative fundamentals
partially explained / mixed
weakly explained or not clearly explained

It should also ask the model to identify names that appear more likely to be:

sentiment-driven
temporarily oversold
possible short-term mean-reversion candidates

The prompt composer should produce:

final prompt text
prompt version identifier
any model-independent schema definition if structured output is required
4. Component 3: Model Execution

This component is responsible for actually calling the AI model.

Each model should have its own execution adapter. For example:

Claude adapter
OpenAI adapter
other-model adapter

Each adapter should:

accept the prompt as input
call the correct API or tool
return the raw response
optionally return a parsed structured output
collect metadata such as response time, estimated cost, and model name

The important design decision here is that all adapters should expose the same interface to the orchestrator.

Example conceptual interface:

run_model(prompt, model_config) -> model_result

Where model_result includes:

raw response text
parsed JSON if available
citations if available
model metadata
error status if any

This makes it easy to compare models later.

5. Component 4: Response Storage

This component saves everything needed for future evaluation.

The system should always preserve the exact raw response of the model. This is critical because later analysis should be based on what the model actually said at the time, not on a cleaned or rewritten version.

In addition to raw text, the system should also save:

parsed structured output
prompt text
model name
prompt version
run date
ticker input
cost estimate
latency
any error or retry information

This component should be designed for long-term auditability and comparison.

End-to-End Flow

Here is the full flow in plain text:

[Daily Run Starts]
        |
        v
[Orchestrator]
        |
        v
[Run Ticker Selection Script]
        |
        v
[Get Filtered Tickers + Price Stats]
        |
        v
[Compose Prompt from Tickers + Research Instructions]
        |
        v
[For Each AI Model]
        |
        +--> [Send Prompt to Model]
        |          |
        |          v
        |   [Receive Raw Response]
        |          |
        |          v
        |   [Parse / Normalize if possible]
        |          |
        |          v
        |   [Store Raw + Parsed Output + Metadata]
        |
        v
[Daily Run Ends]
Model-Agnostic Design Principle

Because the system will compare multiple AI providers, it should be built around a shared contract.

That shared contract should include:

Shared Inputs
run date
ticker list
price-drop metrics
prompt text
prompt version
Shared Outputs
grouped classification
per-ticker reasoning
citations
confidence
raw response
metadata

This ensures each model is being asked the same underlying question, even if the implementation details differ slightly.

Suggested Output Categories

Each model should ideally classify tickers into these groups:

1. Clear negative recent news
   - strong stock-specific, materially negative developments

2. Mixed / medium negatives
   - some negative signals, but not enough to fully explain the move

3. Mostly neutral / weak-news names
   - no strong new fundamental driver found
   - likely sentiment-driven, temporary weakness, or oversold

For each ticker, the model should also return something like:

Ticker: XYZ
Recent price weakness: -3.4%
Most relevant recent news: ...
Does the news logically explain the move?: Yes / Partially / No
Mean-reversion candidate?: Yes / Maybe / No
Confidence: Low / Medium / High
Citations: ...
Storage Design

A simple file-based structure is enough at first.

Example:

daily_runs/
  2026-04-09/
    input_tickers.json
    prompt.txt
    claude/
      raw.txt
      parsed.json
      metadata.json
    openai/
      raw.txt
      parsed.json
      metadata.json
    other_model/
      raw.txt
      parsed.json
      metadata.json

This is simple, auditable, and easy to enrich later.

Future Evaluation Design

Although this is not part of the initial execution pipeline, the stored results are meant to support a later evaluation stage.

At a later date, each saved run can be enriched with:

next-day return
5-day forward return
10-day forward return
1-month return
realized bounce from local low
realized continuation downward
volatility after classification

This will let you answer questions such as:

Did “weakly explained” names bounce more often?
Did “clear negative recent news” names continue falling?
Which AI model produced the most useful classifications?
Suggested Non-Functional Requirements

The system should aim for:

reproducibility
auditability
model-agnostic design
low operational complexity
easy future evaluation
exact preservation of raw model outputs
Summary

This project is a daily AI-assisted stock research pipeline designed to evaluate whether short-term stock price drops are justified by fresh negative fundamentals or are more likely temporary, sentiment-driven dislocations.

The architecture consists of four main components coordinated by an orchestrator:

Orchestrator
  -> Ticker Selection
  -> Prompt Composition
  -> Model Execution
  -> Response Storage

The orchestrator runs the end-to-end workflow, the first component identifies relevant tickers from price action, the second builds a research prompt, the third queries one or more AI models, and the fourth stores the exact responses and metadata for later comparison and backtesting.

The result is a repeatable framework that can be used to compare AI models on a real financial-research task over time.
