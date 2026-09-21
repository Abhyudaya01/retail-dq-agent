.PHONY: setup inject run-agent eval all

UV ?= uv
RAW_DATA ?= data/raw/retail.csv
DB ?= data/retail.duckdb
RAW_DIR ?= data/raw
INJECTED_DATA ?= data/injected/retail_corrupted.parquet
GROUND_TRUTH ?= data/ground_truth.json
AGENT_OUTPUT ?= outputs/proposals.parquet
EVAL_REPORT ?= eval/report.md

setup:
	$(UV) sync

inject:
	$(UV) run python -m src.injection --load-raw --raw-dir "$(RAW_DIR)" --db "$(DB)" --ground-truth "$(GROUND_TRUTH)"

run-agent:
	$(UV) run python -m src.agent.graph run --table-name sales_dirty --db "$(DB)" --output-dir outputs --export-powerbi --ground-truth "$(GROUND_TRUTH)"

eval:
	$(UV) run python -m eval.eval --predictions "$(AGENT_OUTPUT)" --ground-truth "$(GROUND_TRUTH)" --report "$(EVAL_REPORT)"

all: setup inject run-agent eval
