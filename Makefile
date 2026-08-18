.PHONY: run-v1 run-v2 run-v3 run-fabric test visualize

run-v1:
	python baby_universe_v1.py

run-v2:
	python baby_universe_v2.py

run-v3:
	python baby_universe_v3.py

run-fabric:
	python fabric_integration.py

test:
	pytest tests/ -v

visualize:
	python visualization.py

converge:
	python convergence.py
