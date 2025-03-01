get_btc:
	pipenv shell | python3 -c "from data import DataHandler; dh = DataHandler(); dh.get_all_time_btc_data()"
