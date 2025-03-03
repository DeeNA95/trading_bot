.PHONY: data prepare test_proj train train-quick train-full train-parallel train-multi-parallel inference inference-paper docker-build docker-train docker-train-parallel docker-train-multi-parallel docker-inference docker-inference-test

data:
	pipenv run python data_retrieval.py

prepare:
	pipenv install

test_proj:
	pipenv run python run_all_tests.py

train:
	pipenv run python train.py --model_name default_model

train-quick:
	pipenv run python train.py --model_name quick_model --n_episodes 100 --batch_size 32

train-full:
	nohup pipenv run python train.py --model_name full_model --n_episodes 2000 --batch_size 128 --checkpoint_interval 200 > training_output.log 2>&1 &

train-parallel:
	nohup pipenv run python train.py --model_name parallel_model --n_episodes 2000 --batch_size 128 --checkpoint_interval 200 --num_workers 6 > training_output_parallel.log 2>&1 &

train-multi-parallel:
	nohup pipenv run python train.py --parallel --config_file configs/parallel_config.json --max_parallel_jobs 4 --n_episodes 1000 --model_name parallel_experiment > training_output_multi_parallel.log 2>&1 &

inference:
	pipenv run python inference.py --test_data data/BTC_USD_complete.csv

inference-paper:
	pipenv run python inference.py --paper_trading --trade_interval 3600

# Docker commands
docker-build:
	docker build -t trading-bot:latest .

docker-train:
	docker compose up training -- --model_name docker_model

docker-train-parallel:
	docker compose run --name trading-bot-parallel training train.py --model_name docker_parallel_model --n_episodes 2000 --batch_size 128 --checkpoint_interval 200 --num_workers 0

docker-train-multi-parallel:
	docker compose run --name trading-bot-multi-parallel training train.py --parallel --config_file configs/parallel_config.json --max_parallel_jobs 4 --n_episodes 1000 --model_name docker_multi_parallel_model

docker-inference:
	docker compose up inference

docker-inference-test:
	docker compose up inference-test

# Add more targets as needed
