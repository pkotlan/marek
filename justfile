run:
	uv run python src/main.py

docs-serve:
	uv run zensical serve

# deploy:
# 	uv run pyside6-deploy --config-file pysidedeploy.spec

lint:
	uv run ruff check && uv run ty check
