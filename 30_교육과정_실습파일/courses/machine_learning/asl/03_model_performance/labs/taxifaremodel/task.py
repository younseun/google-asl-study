import argparse
import json
import os

from . import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hidden_units",
        help = "Hidden layer sizes to use for DNN feature columns -- provide space-separated layers",
        type = str,
        default = "10,10"
    )
    
    parser.add_argument(
        "--train_data_path",
        help = "GCS or local path to training data",
        required = True
    )
    parser.add_argument(
        "--train_steps",
        help = "Steps to run the training job for (default: 1000)",
        type = int,
        default = 1000
    )
    parser.add_argument(
        "--eval_data_path",
        help = "GCS or local path to evaluation data",
        required = True
    )
    parser.add_argument(
        "--output_dir",
        help = "GCS location to write checkpoints and export models",
        required = True
    )
    parser.add_argument(
        "--job-dir",
        help="This is not used by our model, but it is required by gcloud",
    )
    args = parser.parse_args().__dict__
    
    args["output_dir"] = os.path.join(
        args["output_dir"],
        json.loads(
            os.environ.get("TF_CONFIG", "{}")
        ).get("task", {}).get("trial", "")
    ) 
  
    # Run the training job
    model.train_and_evaluate(args)
