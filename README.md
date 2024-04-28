# loss-curve

## Dev environment 

Conda env named `curve` in `trn`

## Download dataset

```sh
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

## Convert to tfrecords

```sh
python tinystories/make_tfrecord.py \
    data/TinyStoriesV2-GPT4-train.txt \
    data/TinyStoriesV2-GPT4-valid.txt \
    data/tinystories-train.tfrecord \
    data/tinystories-valid.tfrecord
```

There are ~2717000 training samples and ~27000 validation samples.

## Get sentencepiece

```sh
cd sentencepiece
curl https://huggingface.co/t5-base/resolve/main/spiece.model -o t5-base
```

