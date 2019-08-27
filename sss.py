import numpy as np
import pandas as pd
import os
import collections
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    # 在BERT的输出上，添加新的层用来做分类。
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value
    # 隐藏层元素个数 e.g 输出 (1,10)  w: (6, 10)  out: (1, 6)

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())
    # 经过NN后输出个节点个数等于num_labels的个数。

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        # tranpose_b : b在计算之前先转置   (1,10) * (10,6) = (1,6)
        logits = tf.nn.bias_add(logits, output_bias)

        # probabilities = tf.nn.softmax(logits, axis=-1) ### multiclass case
        probabilities = tf.nn.sigmoid(logits)  #### multi-label case
        # 对输出的num_labels个节点，每个节点都做sigmoid，对应不同的label

        labels = tf.cast(labels, tf.float32)
        tf.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)

        # probabilities = tf.nn.softmax(logits, axis=-1)
        # log_probs = tf.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        #
        # per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        # tf.logging.info("*** Features ***")
        # for name in sorted(features.keys()):
        #    tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        # total_loss：整体损失        per_example_loss：单个样本损失      logits：输出      probabilities：sigmoid输出

        tvars = tf.trainable_variables()
        # tvars中是所有可训练参数的信息     name,shape
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            # 这个加上下面init_from_checkpoint是加载模型的方法
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            # get_assignment_map_from_checkpoint 会检查并加载init_checkpoint中的 参数 和 变量 形成map
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
            # 运行时，model的变量已经通过之前的步骤模块化一次了，接下来再训练，就是接着之前的过程继续往下。
            # 至此  上面是加载模型代码
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        output_spec = None
        print("mode:", mode, "probabilities:", probabilities)
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"probabilities": probabilities},
            scaffold=scaffold_fn)
        # mode:预测
        return output_spec

    return model_fn

class InputFeatures(object):
    """单个样本的数据"""
    # 这是经过预处理之后，转为index的数据类型

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):
        # input_ids:分词的ids      input_mask:在sentence位置标注1，padding位置标注0      segment_ids:第一句话标注0，第二句话标注1。
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids,
        self.is_real_example=is_real_example

# 添加BERT用于识别句子开始和结束的“CLS”和“SEP”标记。还为每个输入添加“index”和“segment”标记。因此，根据BERT格式化输入的所有工作都由此函数完成。
def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        print(example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        labels_ids = []
        for label in example.labels:
            labels_ids.append(int(label))

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=labels_ids))
    return features

class InputExample(object):
    """单个样本输入数据."""
    # 这是初始输入的数据类型

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels

def create_examples(df, labels_available=True):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, row) in enumerate(df.values):
        guid = row[0]
        text_a = row[1]
        if labels_available:
            labels = row[2:]
        else:
            labels = [0,0,0,0]
        examples.append(
            InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples


def input_fn_builder(LABEL_COLUMNS, features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "input_mask":
                tf.constant(
                    all_input_mask,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "segment_ids":
                tf.constant(
                    all_segment_ids,
                    shape=[num_examples, seq_length],
                    dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids, shape=[num_examples, len(LABEL_COLUMNS)], dtype=tf.int32),
        })
        # tf.data.Dataset.from_tensor_slices: 将字典分割，'a':[1,2,3,4,5], 'b':np.random((5,2))
        # 按照第一个维度切割 -->  {'a':1, 'b':[0.5,0.6]}  {'a':2, 'b':[0.1,0.8]} {'a':3, 'b':[...]} ...

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        return d

    return input_fn

def create_output(predictions, LABEL_COLUMNS):
    probabilities = []
    for (i, prediction) in enumerate(predictions):
        preds = prediction["probabilities"]
        probabilities.append(preds)
    dff = pd.DataFrame(probabilities)
    dff.columns = LABEL_COLUMNS

    return dff

def test_main():
    ID = 'id'
    DATA_COLUMN = 'content'
    LABEL_COLUMNS = ['environment','price_level','traffic','food']
    num_labels = len(LABEL_COLUMNS)
    use_one_hot_embeddings = False
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 4

    os.chdir(r'E:\Toxic_BERT_multi_task')

    # 加载分词和模型
    BERT_VOCAB = 'chinese_L-12_H-768_A-12/vocab.txt'
    # 模型词汇表
    BERT_INIT_CHKPNT = 'output/model.ckpt'
    # 模型预训练权重
    BERT_CONFIG = 'chinese_L-12_H-768_A-12/bert_config.json'
    # BERT模型架构

    tokenization.validate_case_matches_checkpoint(True,BERT_INIT_CHKPNT)
    # 检查checkpoint的合法性
    tokenizer = tokenization.FullTokenizer(
          vocab_file=BERT_VOCAB, do_lower_case=True)
    tokenizer.tokenize('查看中文分词效果。')

    # test = pd.read_csv('reforcement_test.csv')
    # x_test = test[:100][['id', 'content']] #testing a small sample
    # x_test = x_test.reset_index(drop=True)

    bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
    model_fn = model_fn_builder(bert_config=bert_config, num_labels= len(LABEL_COLUMNS),
                                init_checkpoint=BERT_INIT_CHKPNT,use_one_hot_embeddings=False)
    estimator = tf.estimator.Estimator(model_fn, params={"batch_size": BATCH_SIZE})

    return estimator, tokenizer

# print(submission.head(5))

def create_df_data(corpus, tokenizer,LABEL_COLUMNS, MAX_SEQ_LENGTH=128):
    data_output = pd.DataFrame(columns=['id', 'content','environment','price_level','traffic','food'])
    for index,x in enumerate(corpus):
        add_data = pd.Series({'id':index, 'content':x, 'environment':0,'price_level':0,'traffic':0,'food':0})
        data_output = data_output.append(add_data, ignore_index=True)
    predict_examples = create_examples(data_output,False)
    test_features = convert_examples_to_features(predict_examples, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = input_fn_builder(LABEL_COLUMNS, features=test_features, seq_length=MAX_SEQ_LENGTH,
                                        is_training=False, drop_remainder=False)
    return predict_input_fn


if __name__ == '__main__':
    strs = ['这家店是最近才开业的', '位于河西区彩悦城一层', '菜品，口味什么有时又觉得跑的快，还没夹到菜火车就走了！']

    ID = 'id'
    DATA_COLUMN = 'content'
    LABEL_COLUMNS = ['environment','price_level','traffic','food']
    num_labels = len(LABEL_COLUMNS)
    use_one_hot_embeddings = False
    MAX_SEQ_LENGTH = 128
    BATCH_SIZE = 4

    estimator, tokenizer = test_main()
    # 加载estimator
    predict_input_fn = create_df_data(strs, tokenizer, LABEL_COLUMNS)
    # 数据格式加载

    predictions = estimator.predict(predict_input_fn)
    output_df = create_output(predictions, LABEL_COLUMNS)
    # merged_df =  pd.concat([x_test, output_df], axis=1)
    # submission = merged_df.drop(['id', 'content'], axis=1)
    print(output_df)