# -*- coding: utf-8 -*- 
import os
import shutil
import numpy as np
import tensorflow as tf
from datetime import datetime
print(tf.__version__)

def parameter_display():
    print('>>>>> BUCKET                : {}'.format(BUCKET))
    print('>>>>> DATA_DIR              : {}'.format(DATA_DIR))
    print('>>>>> OUTPUT_DIR            : {}'.format(OUTPUT_DIR))
    print('>>>>> PATTERN               : {}'.format(PATTERN))
    print('>>>>> TRAIN_STEPS           : {}'.format(TRAIN_STEPS))
    print('>>>>> BATCH_SIZE            : {}'.format(BATCH_SIZE))
    print('>>>>> NNSIZE                : {}'.format(NNSIZE))
    print('>>>>> NEMBEDS               : {}'.format(NEMBEDS))
    print('>>>>> KEEP_CHECKPOINT_MAX   : {}'.format(KEEP_CHECKPOINT_MAX))
    

######################################################################
# Columns define
CSV_COLUMNS = "weight_pounds,is_male,mother_age,plurality,gestation_weeks".split(',')
LABEL_COLUMN = "weight_pounds"
# Set default values for each CSV column
DEFAULTS = [[0.0], ["null"], [0.0], ["null"], [0.0]]

######################################################################
def add_engineered_features(features):
    features["dummy"] = features["mother_age"]
    return features

######################################################################
def get_categorical_indicator(name, values):
    return tf.feature_column.indicator_column(
        categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key = name, vocabulary_list = values))

######################################################################
def read_dataset(data_dir, filename_pattern, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(records = value_column, record_defaults = DEFAULTS)
            # Default feature
            features = dict(zip(CSV_COLUMNS, columns))
            # NEW: Add engineered features
            #features = add_engineered_features(features)
            # Default label
            label = features.pop(LABEL_COLUMN)
            return features, label
    
        if PATTERN == "":
            file_path = "{}/{}*".format(data_dir, filename_pattern)
        else:
            file_path = "{}/{}*{}*".format(data_dir, filename_pattern, PATTERN)
        print('>>>>> data filename : {}'.format(file_path))
        
        # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename = file_path)

        # Create dataset from file list
        dataset = (tf.data.TextLineDataset(filenames = file_list)  # Read text file
                     .map(map_func = decode_csv))  # Transform each elem by applying decode_csv fn

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(count = num_epochs).batch(batch_size = batch_size)
        return dataset
    return _input_fn

######################################################################
def get_feature_cols():
    # Vocabulary List
    voca_list_is_male = ["True","False","Unknown"]
    voca_list_plurality = ["Single(1)","Twins(2)","Triplets(3)","Quadruplets(4)","Quintuplets(5)","Multiple(2+)"]

    # Default Feature column
    fc_is_male = tf.feature_column.categorical_column_with_vocabulary_list(key="is_male", vocabulary_list=voca_list_is_male)
    fc_plurality = tf.feature_column.categorical_column_with_vocabulary_list(key="plurality", vocabulary_list=voca_list_plurality)
    fc_mother_age = tf.feature_column.numeric_column(key = "mother_age")
    fc_gestation_weeks = tf.feature_column.numeric_column(key = "gestation_weeks")

    # if DNNRegressor model, use below line
    # fc_is_male   = get_categorical_indicator("is_male", voca_list_is_male)
    # fc_plurality = get_categorical_indicator("plurality", voca_list_plurality)

    
    # ADD Feature column
    fc_dummy = tf.feature_column.numeric_column(key = "dummy")

    # Bucketized columns
    fc_buckets_mother_age = tf.feature_column.bucketized_column(source_column = fc_mother_age, boundaries = np.arange(start = 15, stop = 45, step = 1).tolist())
    fc_buckets_gestation_weeks = tf.feature_column.bucketized_column(source_column = fc_gestation_weeks, boundaries = np.arange(start = 17, stop = 47, step = 1).tolist())
   
    # Embeded Feature columns
    crossed = tf.feature_column.crossed_column(keys=[fc_is_male,fc_plurality,fc_buckets_mother_age,fc_buckets_gestation_weeks], 
                                            hash_bucket_size = 20000)
    fc_embed = tf.feature_column.embedding_column(categorical_column = crossed, dimension = NEMBEDS)

    # Feature columns
    feature_columns = [fc_is_male,
                       fc_plurality,
                       fc_mother_age,
                       fc_gestation_weeks,
                       fc_dummy
                      ]
    
    # Sparse wide columns
    wide = [fc_is_male,fc_plurality,fc_buckets_mother_age,fc_buckets_gestation_weeks]
    
    #Deep colomns
    deep = [fc_mother_age,
            fc_gestation_weeks,
            fc_embed]
    
    return feature_columns, wide, deep

######################################################################
def serving_input_fn():
    feature_placeholders = {
        "is_male"        : tf.placeholder(dtype = tf.string,  shape = [None]),
        "mother_age"     : tf.placeholder(dtype = tf.float32, shape = [None]),
        "plurality"      : tf.placeholder(dtype = tf.string,  shape = [None]),
        "gestation_weeks": tf.placeholder(dtype = tf.float32, shape = [None])
    }
    
    #features = add_engineered_features(feature_placeholders)
    
    # if feature의 shape=(?,), use below line
    features = {
                key: tf.expand_dims(input = tensor, axis = -1)
                for key, tensor in feature_placeholders.items()
               }
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = feature_placeholders)


######################################################################
def _accuracy_bigger(best_eval_result, current_eval_result):
    metric = 'accuracy'
    return best_eval_result[metric] < current_eval_result[metric]


######################################################################
def my_rmse(labels, predictions):
    pred_values = predictions["predictions"]
    return {
        "rmse": tf.metrics.root_mean_squared_error(
            labels=labels,
            predictions=pred_values
        )
    }

######################################################################
def train_and_evaluate(output_dir):
    EVAL_INTERVAL = 300  # seconds
    
    parameter_display()
    feature_columns, wide, deep = get_feature_cols()
        
    run_config = tf.estimator.RunConfig(
        save_checkpoints_secs = EVAL_INTERVAL,
        keep_checkpoint_max = KEEP_CHECKPOINT_MAX)

    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = output_dir,
        linear_feature_columns = wide,
        dnn_feature_columns = deep,
        dnn_hidden_units = NNSIZE,
        config = run_config)
 
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)

    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(DATA_DIR, "train", mode = tf.estimator.ModeKeys.TRAIN, batch_size=BATCH_SIZE),
        max_steps = TRAIN_STEPS)
    
    #Final_exporter = tf.estimator.FinalExporter('./exporter', serving_input_receiver_fn=serving_input_fn)
    #exporter = Final_exporter

    exporter = tf.estimator.LatestExporter(
        name="exporter",
        serving_input_receiver_fn=serving_input_fn,
        exports_to_keep=None)
    
    
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(DATA_DIR, "eval", mode = tf.estimator.ModeKeys.EVAL, batch_size=BATCH_SIZE),
        steps = None,
        start_delay_secs = 60, # start evaluating after N seconds
        throttle_secs = EVAL_INTERVAL,  # evaluate every N seconds
        exporters = exporter)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    
