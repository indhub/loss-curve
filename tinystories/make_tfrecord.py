import argparse
import tensorflow as tf

def create_tf_record(in_data_path, out_tfrecord_path):
    def generator():
        with open(in_data_path, 'r') as file:
            sample = ""
            for line in file:
                if "<|endoftext|>" in line:
                    yield {'text': sample}
                    sample = ""
                else:
                    sample += line
            if sample:
                yield {'text': sample}

    def serialize_example(example):
        features = {
            'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[example['text'].encode('utf-8')]))
        }
        return tf.train.Example(features=tf.train.Features(feature=features))

    with tf.io.TFRecordWriter(out_tfrecord_path) as writer:
        idx = 0
        for example in generator():
            writer.write(serialize_example(example).SerializeToString())
            if idx % 1000 == 0:
                print(f"Processed {idx} examples")
            idx += 1

def main(train_data_path: str, validation_data_path: str, train_tfrecord_path: str, validation_tfrecord_path: str):
    create_tf_record(train_data_path, train_tfrecord_path)
    create_tf_record(validation_data_path, validation_tfrecord_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and validate model.')
    parser.add_argument('train_data_path', type=str, help='Path to training data')
    parser.add_argument('validation_data_path', type=str, help='Path to validation data')
    parser.add_argument('train_tfrecord_path', type=str, help='Path to output training TFRecord')
    parser.add_argument('validation_tfrecord_path', type=str, help='Path to output validation TFRecord')
    args = parser.parse_args()
    main(args.train_data_path, args.validation_data_path, args.train_tfrecord_path, args.validation_tfrecord_path)
