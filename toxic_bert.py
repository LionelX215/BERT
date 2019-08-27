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


# 将数据转为BERT理解的格式
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


def create_examples(df, labels_available=True):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, row) in enumerate(df.values):
        guid = row[0]
        text_a = row[1]
        if labels_available:
            labels = row[2:]
        else:
            labels = [0,0,0,0,0,0]
        examples.append(
            InputExample(guid=guid, text_a=text_a, labels=labels))
    return examples



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



class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


def convert_single_example(ex_index, example, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        # isinstance(obj, classinfo) 判断obj是不是某个类型       examples应该是InputExample，不是PaddingInputExample
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=0,
            is_real_example=False)

    tokens_a = tokenizer.tokenize(example.text_a)
    # 分词后
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
            # 截断处理，如果文本长度大与max_length-2 就截断。   -2是因为有开头和结束符。 CLS、SEP

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    # 按照上面英文注释的格式添加，中文是字。

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # tokens是分好的词的list，转为词的ids值。

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to. 只有有真实值的位置上为1，其他是padding的位置都为0.
    input_mask = [1] * len(input_ids)

    # 如果sentence没有达到目标长度，用padding使sentence达到目标长度
    while len(input_ids) < max_seq_length:
        # input_ids:分词的ids      input_mask:在sentence位置标注1，padding位置标注0      segment_ids:第一句话标注0，第二句话标注1。
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    labels_ids = []
    for label in example.labels:
        labels_ids.append(int(label))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=labels_ids,
        is_real_example=True)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """将一组input_examples存入TFRecord文件"""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        # examples是List,则ex_index是index，example是单个样本所有信息的obj。
        # if ex_index % 10000 == 0:
        # tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example,
                                         max_seq_length, tokenizer)

        # 返回的feature是InputFeatures类型的单个样本的对象。

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            # 这里生成的是能够统一形式存储的格式，存入TFRecord文件。  整数型List
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        if isinstance(feature.label_ids, list):
            # 输出tuple，tuple的第一个元素是list
            label_ids = feature.label_ids
        else:
            label_ids = feature.label_ids[0]
        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        # 整理一条样本数据。
        writer.write(tf_example.SerializeToString())
        # 数据写入TFRecord文件。 SerializeToString()序列化字符串，节省存储空间。
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    # 创建一个input_fn必报传递给TPUEstimator。    和BERT源码中一致。

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([6], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
        # FixedLenFeature 用来处理定长的tensor
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        # 解析tfrecord文件的每条记录，即序列化后的tf.train.Example
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            # 增加数据量

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                # tfrecord数据解析   转化数据格式  int32  int64
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # 做截断，在其他位置调用中都用于了text_b，后面继续关注。

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


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


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
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
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
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
        if mode == tf.estimator.ModeKeys.TRAIN:
            # 表示  如果是训练模式

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            # create_optimizer中：
            # tf.train.polynomial_decay(learning_rate, global_step, num_train_steps, end_learning_rate=0.0, power=1.0, cycle=False)
            # 多项式学习率迭代：(初始学习率，步数，衰减训练步数，最终lr， power=1为线性衰减(0.5为平方指数衰减)， cycle为T则降到最小再上升)
            # cycle=True，学习率会再最小时上升到一定程度再衰减，有跳出局部最优的功能。https://blog.csdn.net/weixin_39875161/article/details/93423883
            # 多项式是每步都衰减的，每num_train_steps步，衰减power程度。所以可能非整数倍步衰减到最小lr值。
            # 优化器：AdamWeightDecayOptimizer     Adam + L2
            # Adamw是在Adam的更新策略中采用了计算整体损失函数的梯度来进行更新而不是只计算不带正则项部分的梯度进行更新之后再进行权重衰减。
            # tf.clip_by_global_norm(grads, clip_norm=1.0) grads是梯度，通过限制梯度L2-norm范围的方式防止梯度爆炸的问题，是常用的梯度规约的方式。

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold=scaffold_fn)
            # https://www.w3cschool.cn/tensorflow_python/tensorflow_python-q5hc2ozv.html
            # 是一个class(类)，定义在model_fn中，并且model_fn返回的是它的一个实例，是用来初始化Estimator类的，后面可以看到载入Estimator中了。
        elif mode == tf.estimator.ModeKeys.EVAL:
            # 计算模式     EstimatorSpec中的mode有: training evaluation prediction

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                logits_split = tf.split(probabilities, num_labels, axis=-1)
                label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                # 每个节点的sigmoid值和样本的label值分割为list。 https://blog.csdn.net/SangrealLilith/article/details/80272346
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                    # 返回两个值，一个是到上一个批次的auc，另一个是经过本批次更新后的auc
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                return eval_dict
                # 最终返回的是损失值，eval_dict中包含了每个label的损失和整体的平均损失。

                ## original eval metrics
                # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                # accuracy = tf.metrics.accuracy(
                #     labels=label_ids, predictions=predictions, weights=is_real_example)
                # loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                # return {
                #     "eval_accuracy": accuracy,
                #     "eval_loss": loss,
                # }

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                scaffold=scaffold_fn)
        else:
            print("mode:", mode, "probabilities:", probabilities)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities},
                scaffold=scaffold_fn)
            # mode:预测
        return output_spec

    return model_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

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
                tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
        })
        # tf.data.Dataset.from_tensor_slices: 将字典分割，'a':[1,2,3,4,5], 'b':np.random((5,2))
        # 按照第一个维度切割 -->  {'a':1, 'b':[0.5,0.6]}  {'a':2, 'b':[0.1,0.8]} {'a':3, 'b':[...]} ...

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def create_output(predictions):
    probabilities = []
    for (i, prediction) in enumerate(predictions):
        preds = prediction["probabilities"]
        probabilities.append(preds)
    dff = pd.DataFrame(probabilities)
    dff.columns = LABEL_COLUMNS

    return dff


os.chdir(r'E:\Toxic_BERT_multi_task')

# 加载分词和模型
BERT_VOCAB = 'chinese_L-12_H-768_A-12/vocab.txt'
# 模型词汇表
BERT_INIT_CHKPNT = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
# 模型预训练权重
BERT_CONFIG = 'chinese_L-12_H-768_A-12/bert_config.json'
# BERT模型架构

tokenization.validate_case_matches_checkpoint(True,BERT_INIT_CHKPNT)
# 检查checkpoint的合法性
tokenizer = tokenization.FullTokenizer(
      vocab_file=BERT_VOCAB, do_lower_case=True)
# 分词，tokenization中有两种分词模式，Basic和Full模型分词。Full表示段对端(end-to-end)的分词
# Basic主要进行的是unicode转换、标点符号分割、小写转换、中文字符分割、去除重音符号等，返回的是关于词的数组(中文是字的数组)。
# wordpiece tokenizer是将合成词分解为词根。类似unwanted  un  ##want  ##ed    英文中比较多，因为词典很难收录所有单词。
# FullTokenizer 对一个文本段进行以上两种解析，最后返回词（字）的数组，同时还提供token到id的索引以及id到token的索引。token可以理解为文本段处理过后的最小单元。
tokenizer.tokenize('查看中文分词效果。')

train_data_path='data_set/train.csv'
train = pd.read_csv(train_data_path)
test = pd.read_csv('data_set/test.csv')
train.head(5)

ID = 'id'
DATA_COLUMN = 'comment_text'
LABEL_COLUMNS = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']


TRAIN_VAL_RATIO = 0.9
LEN = train.shape[0]
SIZE_TRAIN = int(TRAIN_VAL_RATIO*LEN)
# 训练集和验证集比例

x_train = train[:SIZE_TRAIN]
x_val = train[SIZE_TRAIN:]

train_examples = create_examples(x_train)
# creat_examples是将数据中的id,values,labels都提出来，然后通过InputExample做成对象，append方法添加到返回中。
# train_examples输出是[obj1, obj2, obj3, ... ]  每个obj都包含了一个样本的所有特征(id, values, labels)。
MAX_SEQ_LENGTH = 128
# 设置句子最长tokens个数

# 参数  warm up
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 2.0

# Warmup is a period of time where hte learning rate
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500
# 存储参数的步数


num_train_steps = int(len(train_examples) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
# 计算总计训练次数
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)
# warm up 次数


os.path.abspath('.')
# 查看当前路径
train_file = os.path.join('./', "train.tf_record")
if not os.path.exists(train_file):
    # 如果不存在train.tf_record文件就创建
    open(train_file, 'w').close()

# train_examples: 由样本的obj组成的list。   tokenizer:有关字典的obj。   train_file:是train.tf_record的路径
file_based_convert_examples_to_features(
            train_examples, MAX_SEQ_LENGTH, tokenizer, train_file)
# 执行结果将样本的内容按照固定格式存入tfrecord文件中。
tf.logging.info("***** Running training *****")
tf.logging.info("  Num examples = %d", len(train_examples))
tf.logging.info("  Batch size = %d", BATCH_SIZE)
tf.logging.info("  Num steps = %d", num_train_steps)
# 日志

# 传递内容到TPU训练？ train_file是之前操作的已经将样本数据按规则存入的TFRecord文件  和BERT源码中一致。
train_input_fn = file_based_input_fn_builder(
    input_file=train_file,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=True)


OUTPUT_DIR = "./output"
# 输出路径 和 存储checkpoint的step设定
# 设置GPU使用率
session_config = tf.ConfigProto(log_device_placement=True)
session_config.gpu_options.per_process_gpu_memory_fraction = 0.6
run_config = tf.estimator.RunConfig(
    model_dir=OUTPUT_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    session_config=session_config,
    keep_checkpoint_max=1,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)


bert_config = modeling.BertConfig.from_json_file(BERT_CONFIG)
# 从存储的BERT模型结构文件中读取BERT模型框架
model_fn = model_fn_builder(
  bert_config=bert_config,
  num_labels= len(LABEL_COLUMNS),
  init_checkpoint=BERT_INIT_CHKPNT,
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps,
  use_tpu=False,
  use_one_hot_embeddings=False)
# 参数说明： bert_congfig:BERT的模型框架，从文件中导出。     num_labels: 任务数，label_columns是数据中的标签数据。
# init_checkpoint: 预训练好的权重文件
# num_warmip_steps:启用warm up，当step小于warm up setp时，学习率等于基础学习率×(当前step/warmup_step)，由于后者是一个小于1的数值，因此在整个warm up的过程中，学习率是一个递增的过程！当warm up结束后，学习率开始递减。
# 上面相当于是对Estimator类的初始化工作，包括loss, train_op等

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})
# 构建Estimator类封装了包含model_fn制定的模型函数，给定输入和其他一些参数,返回需要进行训练、计算,或预测的操作.


print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
# 训练     train_input_fn 将tfrcorde中的数据解析出来，map and batch作为真正的输入。
print("Training took time ", datetime.now() - current_time)


eval_file = os.path.join('./working', "eval.tf_record")
# 路径拼接
if not os.path.exists(eval_file):
    open(eval_file, 'w').close()

eval_examples = create_examples(x_val)
file_based_convert_examples_to_features(
    eval_examples, MAX_SEQ_LENGTH, tokenizer, eval_file)
# 其实还是一样的数据处理流程，先创建tfrecord文件，然后将数据写入文件


# This tells the estimator to run through the entire set.
eval_steps = None

eval_drop_remainder = False
eval_input_fn = file_based_input_fn_builder(
    input_file=eval_file,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)

result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
# 预测 x_val


output_eval_file = os.path.join("./working", "eval_results.txt")
with tf.gfile.GFile(output_eval_file, "w") as writer:
    tf.logging.info("***** Eval results *****")
    for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
# eval_result.txt  记录val文件中的预测结果


x_test = test[:10000] #testing a small sample
x_test = x_test.reset_index(drop=True)
predict_examples = create_examples(x_test,False)
# 存入list  [[id, text_a, lables], [id, text_a, lables], ...]


test_features = convert_examples_to_features(predict_examples, MAX_SEQ_LENGTH, tokenizer)


print('Beginning Predictions!')
current_time = datetime.now()

predict_input_fn = input_fn_builder(features=test_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
# 返回的是处理好的数据维度，可以用于输入
predictions = estimator.predict(predict_input_fn)
print("Prediction took time ", datetime.now() - current_time)


output_df = create_output(predictions)
merged_df =  pd.concat([x_test, output_df], axis=1)
submission = merged_df.drop(['comment_text'], axis=1)
submission.to_csv("sample_submission0.csv", index=False)
# 去掉语料内容  存储的label预测结果

submission.head()