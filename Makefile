# Kairos — the AURA dual-process trading brain. Developer entry points.
# Proprietary umbrella; bundles Apache-2.0 components (see NOTICE).

VENV := .venv/bin
PY   := $(VENV)/python
KAIROS := $(PY) -m kairos.cli

.PHONY: help install install-all gate soul test test-core test-all loop perceive \
        reason build-cpp redteam reproduce lint clean

help:
	@echo "Kairos targets:"
	@echo "  make install     - create .venv and install core + dev (portable, no keys)"
	@echo "  make install-all - install core + [reasoning] + [native] + [mlx] + dev"
	@echo "  make gate        - FAST portable gate: soul_check + core tests + loop smoke"
	@echo "  make soul        - run the (scoped) Constitution enforcer"
	@echo "  make test-core   - pytest bridge + loop + perception (no LLM/MLX needed)"
	@echo "  make test-all    - full pytest incl. reasoning (needs [reasoning] extra)"
	@echo "  make loop        - run the perceive->reason->act->reflect demo (deterministic)"
	@echo "  make perceive    - System-1 offline PoC: gen->train->cluster (needs [mlx])"
	@echo "  make reason T=NVDA D=2024-05-10 - System-2 decision (needs [reasoning] + keys)"
	@echo "  make build-cpp   - build the C++ zero-copy ring (needs [native], Apple/clang)"
	@echo "  make reproduce   - end-to-end reproducibility gate (honest, portable)"
	@echo "  make lint        - ruff check"
	@echo "  make clean       - remove regenerable data/ artifacts/ build/"

install:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e ".[dev,viz]"
	@echo "\n✓ core installed. 'make gate' to verify."

install-all:
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e ".[dev,viz,reasoning,native]"
	-$(PY) -m pip install -e ".[mlx]"   # Apple Silicon only; ignored elsewhere
	@echo "\n✓ full install complete."

# --- The fast, portable regression gate (no keys, no MLX, runs anywhere) ---
gate: soul test-core loop
	@echo "\n✓ regression gate passed."

soul:
	$(PY) scripts/soul_check.py

test-core:
	$(PY) -m pytest tests/bridge tests/loop tests/perception -q -m "not mlx and not native and not reasoning"

test: test-core

test-all:
	$(PY) -m pytest tests -q

loop:
	$(KAIROS) loop --scenario toxic

perceive:
	$(KAIROS) perceive --mode synthetic

reason:
	$(KAIROS) reason $(or $(T),NVDA) $(or $(D),2024-05-10)

redteam:
	$(PY) -m pytest tests/perception/red_team -q -s

build-cpp:
	bash scripts/build_cpp.sh

reproduce:
	$(PY) scripts/reproduce.py

# Lint Kairos-original code only. The vendored System-1/System-2 subtrees
# (kairos.perception, kairos.reasoning) follow their upstream style.
KAIROS_SRC := src/kairos/bridge src/kairos/loop src/kairos/cli.py src/kairos/__init__.py \
              scripts/soul_check.py scripts/reproduce.py tests/bridge tests/loop
lint:
	$(VENV)/ruff check $(KAIROS_SRC)

clean:
	rm -rf build data artifacts results logs
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned regenerable outputs."
