import argparse
import json
import os

from . import model

import tensorflow as tf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bucket",
        help="GCS path to data. We assume that data is in \
        gs://BUCKET/babyweight/preproc/",
        required=True
    )
    parser.add_argument(
        "--data_dir",
        help="train and eval data directory",
        required=True
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",
        required=True
    )
    parser.add_argument(
        "--pattern",
        help="data file pattern",
        required=True
    )
    parser.add_argument(
        "--train_steps",
        help="Number of Train Step.",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--batch_size",
        help="Number of examples to compute gradient over.",
        type=int,
        default=512
    )
    parser.add_argument(
        "--job-dir",
        help="this model ignores this field, but it is required by gcloud",
        default="junk"
    )
    parser.add_argument(
        "--nnsize",
        help="Hidden layer sizes to use for DNN feature columns -- provide \
        space-separated layers",
        nargs="+",
        type=int,
        default=[128, 32, 4]
    )
    parser.add_argument(
        "--nembeds",
        help="Embedding size of a cross of n key real-valued parameters",
        type=int,
        default=6
    )
    parser.add_argument(
        "--save_checkpoints_sec",
        help="",
        type=int,
        default=30
    )
    parser.add_argument(
        "--keep_checkpoints_max",
        help="",
        type=int,
        default=10
    )
    parser.add_argument(
        "--eval_secs",
        help="",
        type=int,
        default=30
    )

    # Parse arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # Pop unnecessary args needed for gcloud
    arguments.pop("job-dir", None)

    # Assign the arguments to the model variables
    output_dir                  = arguments.pop("output_dir")
    model.OUTPUT_DIR            = output_dir
    model.BUCKET                = arguments.pop("bucket")
    model.DATA_DIR              = arguments.pop("data_dir")
    model.PATTERN               = arguments.pop("pattern")
    model.TRAIN_STEPS           = arguments.pop("train_steps")
    model.BATCH_SIZE            = arguments.pop("batch_size")
    model.NNSIZE                = arguments.pop("nnsize")
    model.NEMBEDS               = arguments.pop("nembeds")
    model.SAVE_CHECKPOINTS_SECS = arguments.pop("save_checkpoints_sec")
    model.KEEP_CHECKPOINT_MAX   = arguments.pop("keep_checkpoints_max")
    model.EVAL_SECS             = arguments.pop("eval_secs")
    
    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    )

    # Run the training job
    model.train_and_evaluate(output_dir)
