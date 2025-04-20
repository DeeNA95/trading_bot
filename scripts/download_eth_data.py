#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from data import DataHandler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Initialize data handler
    data_handler = DataHandler()

    # Calculate dates (ETH started trading on futures around Sept 2019)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 6)

    # Process data with all metrics and features
    logger.info(f"Fetching ETHUSDT data from {start_date} to {end_date}")

    processed_data = data_handler.process_market_data(
        symbol="ETHUSDT",
        interval="1m",
        start_time=start_date,
        end_time=end_date,
        save_path="gs://ctrading/data/eth/ETHUSDT_1m_with_metrics_6y.parquet",
    )

    if processed_data.empty:
        logger.error("Failed to fetch and process data")
        sys.exit(1)

    logger.info(f"Successfully processed {len(processed_data)} data points")
    logger.info("Data saved to GCS bucket")

if __name__ == "__main__":
    main()
