import tensorflow as tf
import numpy as np
import shutil
print(tf.__version__)

#1. Train and Evaluate Input Functions
CSV_COLUMN_NAMES = ["fare_amount","dayofweek","hourofday","pickuplon","pickuplat","dropofflon","dropofflat"]
CSV_DEFAULTS = [[0.0],[1],[0],[-74.0],[40.0],[-74.0],[40.7]]

def read_dataset(csv_path):
    def _parse_row(row):
        # Decode the CSV row into list of TF tensors
        fields = tf.decode_csv(records = row, record_defaults = CSV_DEFAULTS)

        # Pack the result into a dictionary
        features = dict(zip(CSV_COLUMN_NAMES, fields))
        
        # NEW: Add engineered features
        features = add_engineered_features(features)
        
        # Separate the label from the features
        label = features.pop("fare_amount") # remove label from features and store

        return features, label
    
    # Create a dataset containing the text lines.
    dataset = tf.data.Dataset.list_files(file_pattern = csv_path) # (i.e. data_file_*.csv)
    dataset = dataset.flat_map(map_func = lambda filename:tf.data.TextLineDataset(filenames = filename).skip(count = 1))

    # Parse each CSV row into correct (features,label) format for Estimator API
    dataset = dataset.map(map_func = _parse_row)
    
    return dataset

def train_input_fn(csv_path, batch_size = 128):
    #1. Convert CSV into tf.data.Dataset with (features,label) format
    dataset = read_dataset(csv_path)
      
    #2. Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(buffer_size = 1000).repeat(count = None).batch(batch_size = batch_size)
   
    return dataset

def eval_input_fn(csv_path, batch_size = 128):
    #1. Convert CSV into tf.data.Dataset with (features,label) format
    dataset = read_dataset(csv_path)

    #2.Batch the examples.
    dataset = dataset.batch(batch_size = batch_size)
   
    return dataset
  
#2. Feature Engineering
# One hot encode dayofweek and hourofday
fc_dayofweek = tf.feature_column.categorical_column_with_identity(key = "dayofweek", num_buckets = 7)
fc_hourofday = tf.feature_column.categorical_column_with_identity(key = "hourofday", num_buckets = 24)

# Cross features to get combination of day and hour
fc_day_hr = tf.feature_column.crossed_column(keys = [fc_dayofweek, fc_hourofday], hash_bucket_size = 24 * 7)

# Bucketize latitudes and longitudes
NBUCKETS = 16
latbuckets = np.linspace(start = 38.0, stop = 42.0, num = NBUCKETS).tolist()
lonbuckets = np.linspace(start = -76.0, stop = -72.0, num = NBUCKETS).tolist()
fc_bucketized_plat = tf.feature_column.bucketized_column(source_column = tf.feature_column.numeric_column(key = "pickuplon"), boundaries = lonbuckets)
fc_bucketized_plon = tf.feature_column.bucketized_column(source_column = tf.feature_column.numeric_column(key = "pickuplat"), boundaries = latbuckets)
fc_bucketized_dlat = tf.feature_column.bucketized_column(source_column = tf.feature_column.numeric_column(key = "dropofflon"), boundaries = lonbuckets)
fc_bucketized_dlon = tf.feature_column.bucketized_column(source_column = tf.feature_column.numeric_column(key = "dropofflat"), boundaries = latbuckets)

def add_engineered_features(features):
    features["dayofweek"] = features["dayofweek"] - 1 # subtract one since our days of week are 1-7 instead of 0-6
    
    features["latdiff"] = features["pickuplat"] - features["dropofflat"] # East/West
    features["londiff"] = features["pickuplon"] - features["dropofflon"] # North/South
    features["euclidean_dist"] = tf.sqrt(features["latdiff"]**2 + features["londiff"]**2)

    return features

feature_cols = [
  #1. Engineered using tf.feature_column module
  tf.feature_column.indicator_column(categorical_column = fc_day_hr),
  fc_bucketized_plat,
  fc_bucketized_plon,
  fc_bucketized_dlat,
  fc_bucketized_dlon,
  #2. Engineered in input functions
  tf.feature_column.numeric_column(key = "latdiff"),
  tf.feature_column.numeric_column(key = "londiff"),
  tf.feature_column.numeric_column(key = "euclidean_dist") 
]

#3. Serving Input Receiver Function
def serving_input_receiver_fn():
    receiver_tensors = {
        'dayofweek' : tf.placeholder(dtype = tf.int32, shape = [None]), # shape is vector to allow batch of requests
        'hourofday' : tf.placeholder(dtype = tf.int32, shape = [None]),
        'pickuplon' : tf.placeholder(dtype = tf.float32, shape = [None]), 
        'pickuplat' : tf.placeholder(dtype = tf.float32, shape = [None]),
        'dropofflat' : tf.placeholder(dtype = tf.float32, shape = [None]),
        'dropofflon' : tf.placeholder(dtype = tf.float32, shape = [None]),
    }
    
    features = add_engineered_features(receiver_tensors) # 'features' is what is passed on to the model
    
    return tf.estimator.export.ServingInputReceiver(features = features, receiver_tensors = receiver_tensors)
  
#4. Train and Evaluate
def train_and_evaluate(params):
    OUTDIR = params["output_dir"]

    model = tf.estimator.DNNRegressor(
        hidden_units = params["hidden_units"].split(","), # NEW: paramaterize architecture
        feature_columns = feature_cols, 
        model_dir = OUTDIR,
        config = tf.estimator.RunConfig(
            tf_random_seed = 1, # for reproducibility
            save_checkpoints_steps = max(100, params["train_steps"] // 10) # checkpoint every N steps
        ) 
    )

    # Add custom evaluation metric
    def my_rmse(labels, predictions):
        pred_values = tf.squeeze(input = predictions["predictions"], axis = -1)
        return {"rmse": tf.metrics.root_mean_squared_error(labels = labels, predictions = pred_values)}
    
    model = tf.contrib.estimator.add_metrics(model, my_rmse)  

    train_spec = tf.estimator.TrainSpec(
        input_fn = lambda: train_input_fn(params["train_data_path"]),
        max_steps = params["train_steps"])

    exporter = tf.estimator.FinalExporter(name = "exporter", serving_input_receiver_fn = serving_input_receiver_fn) # export SavedModel once at the end of training
    # Note: alternatively use tf.estimator.BestExporter to export at every checkpoint that has lower loss than the previous checkpoint


    eval_spec = tf.estimator.EvalSpec(
        input_fn = lambda: eval_input_fn(params["eval_data_path"]),
        steps = None,
        start_delay_secs = 1, # wait at least N seconds before first evaluation (default 120)
        throttle_secs = 1, # wait at least N seconds before each subsequent evaluation (default 600)
        exporters = exporter) # export SavedModel once at the end of training

    tf.logging.set_verbosity(v = tf.logging.INFO) # so loss is printed during training
    shutil.rmtree(path = OUTDIR, ignore_errors = True) # start fresh each time

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
