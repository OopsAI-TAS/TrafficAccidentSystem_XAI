train:
	PYTHONPATH=. python3 train/train.py

ig:
	PYTHONPATH=. python3 xai/batch_ig.py

analysis:
	PYTHONPATH=. python3 xai/analysis_value.py