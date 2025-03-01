.PHONY: data prepare test_proj train train-quick train-full train-parallel inference inference-paper

data:
	pipenv run python data_retrieval.py

prepare:
	pipenv install

test_proj:
	pipenv run python run_all_tests.py

train:
	pipenv run python train.py

train-quick:
	pipenv run python train.py --n_episodes 100 --batch_size 32

train-full:
	nohup pipenv run python train.py --n_episodes 2000 --batch_size 128 --checkpoint_interval 200 > training_output.log 2>&1 &

train-parallel:
	nohup pipenv run python train.py --n_episodes 2000 --batch_size 128 --checkpoint_interval 200 --num_workers 6 > training_output_parallel.log 2>&1 &

inference:
	pipenv run python inference.py --test_data data/BTC_USD_complete.csv

inference-paper:
	pipenv run python inference.py --paper_trading --trade_interval 3600

# Add more targets as needed
