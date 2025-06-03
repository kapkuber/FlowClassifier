.PHONY: label merge train predict clean all

label:
	python scripts/label_flows.py

merge:
	python scripts/merge_flows.py

train:
	python model/train_model.py

predict:
	python model/apply_model.py

all: label merge train predict

clean:
	rm -f output/*.csv
	rm -f model/params/*.txt
