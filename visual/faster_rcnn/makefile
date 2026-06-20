.PHONY: help train infer clean
PYTHON := poetry run python
DATA_PATH ?= ./data
CHECKPOINT_PATH ?= ./checkpoints
ITER_WARM ?= 60000
ITER_COOL ?= 20000
ITER_FT_WARM ?= 20000
ITER_FT_COOL ?= 10000
IMAGE_PATH ?= 
MODEL_PATH ?= "voc_faster_rcnn_final.pth"
SCORE_THRESH ?= 0.5

help:
	@echo "Available targets:"
	@echo "  make train              - Train Faster R-CNN model"
	@echo "  make infer              - Run inference on an image"
	@echo "  make clean              - Remove checkpoint directory"
	@echo ""
	@echo "Training options:"
	@echo "  make train DATA_PATH=<path>"
	@echo "  make train CHECKPOINT_PATH=<path>"
	@echo "  make train ITER_WARM=<int> ITER_COOL=<int>"
	@echo "  make train ITER_FT_WARM=<int> ITER_FT_COOL=<int>"
	@echo ""
	@echo "Inference options:"
	@echo "  make infer IMAGE_PATH=<path> MODEL_PATH=<path>"
	@echo "  make infer IMAGE_PATH=<path> MODEL_PATH=<path> SCORE_THRESH=<float>"

train:
	$(PYTHON) -m train.pipeline \
		--data-path $(DATA_PATH) \
		--checkpoint-path $(CHECKPOINT_PATH) \
		--iter-warm $(ITER_WARM) \
		--iter-cool $(ITER_COOL) \
		--iter-ft-warm $(ITER_FT_WARM) \
		--iter-ft-cool $(ITER_FT_COOL)

infer:
	$(PYTHON) inference.py \
		--image-path $(IMAGE_PATH) \
		--model-path $(MODEL_PATH) \
		--score-thresh $(SCORE_THRESH)

clean:
	rm -rf $(CHECKPOINT_PATH)
