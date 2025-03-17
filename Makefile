gce-instance:
	gcloud compute instances create trading-instance \
	--zone=asia-southeast1-c \
	--machine-type=n1-standard-2 \
	--accelerator=type=nvidia-tesla-t4,count=1 \
	--maintenance-policy=TERMINATE \
	--image-family=pytorch-latest-gpu \
	--image-project=deeplearning-platform-release \
	--boot-disk-size=200GB \
	--scopes=https://www.googleapis.com/auth/cloud-platform \
	--metadata="install-nvidia-driver=True"

load_env :
	pipenv shell

train-btc: #load_env
	python train.py --data gs://crypto_trading_models/data/BTC/BTCUSDT_1m_with_metrics.csv --symbol BTCUSDT --batch_size 512 --save_path gs://crypto_trading_models/LSTM/POSITION_GAINERS/BTC/

train-eth: #load_env
	nohup python train.py --train_data gs://crypto_trading_models/data/ETH/ETHUSDT_1m_with_metrics_train.csv --test_data gs://crypto_trading_models/data/ETH/ETHUSDT_1m_with_metrics_test.csv --symbol ETHUSDT --model transformer --save_path crypto_trading_models/TRANSFORMERS/ETH/ --batch_size 2048

train-bnb: load_env
	python train.py --data gs://crypto_trading_models/data/BNB/BNBUSDT_1h_with_metrics.csv --symbol BNBUSDT --batch_size 512 --save_path gs://crypto_trading_models/LSTM/POSITION_GAINERS/BNB/

train-xrp: load_env
	python train.py --data gs://crypto_trading_models/data/XRP/XRPUSDT_1h_with_metrics.csv --symbol XRPUSDT --batch_size 512 --save_path gs://crypto_trading_models/LSTM/POSITION_GAINERS/XRP/

trade:
	./run_live_eth.sh
	#./run_live_btc.sh
	# ./run_live_xrp.sh
	# ./run_bnb_live.sh
