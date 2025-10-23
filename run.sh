timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
mv outputs backups/outputs_$timestamp
python download_data.py
python impute_data.py
python simple_forecast.py
python train_buy_strategy.py