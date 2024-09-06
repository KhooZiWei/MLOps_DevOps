from azureml.core.run import Run
from azureml.core import Dataset, Datastore, Workspace
import os
import argparse
import joblib
import json
from train import split_data, train_model, get_model_metrics, feature_engineering

# Register dataset function
def register_dataset(
        aml_workspace: Workspace, 
        dataset_name: str, 
        datastore_name: str, 
        file_path: str
    ) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace, name=dataset_name, create_new_version=True)
    
    return dataset

def main():
    print("Running train_aml.py")

    # Initializes an argument parser for the script.
    parser = argparse.ArgumentParser("train")
    parser.add_argument(
        "--model_name", 
        type=str, 
        help="Name of the Model", 
        default="superstore_model.pkl"
    )

    parser.add_argument(
        "--data_file_path", 
        type=str, 
        help="data file path, if specified, a new version of the dataset will be registered", 
        default="superstore" # Get it from Azure CLI pipeine run command
    )

    parser.add_argument(
        "--dataset_name", 
        type=str, 
        help="Dataset name", 
        default="cleaned_superstore"
    )

    # Parses the arguments provided when the script is executed
    args = parser.parse_args()
    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    model_name = args.model_name
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context() # Gets the current Azure ML run context

    # Load training parameters from parameters.json
    with open("parameters.json") as f:
        pars = json.load(f)
    try:
        train_args = pars["training"]
    except KeyError:
        print("Could not load training values from file, using default parameters")
        train_args = {}

    # Log the training parameters
    print(f"Parameters: {train_args}")
    for k, v in train_args.items():
        run.log(k, v)

    # Get the dataset
    if dataset_name:
        if data_file_path == 'none':
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name)
        else:
            dataset = register_dataset(run.experiment.workspace, 
                                       dataset_name, 
                                       "workspaceblobstore", 
                                       data_file_path)
    else:
        e = ("No dataset provided")
        print(e)
        raise Exception(e)

    # Link dataset to the run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset

    # Convert dataset to pandas DataFrame and perform feature engineering
    df = dataset.to_pandas_dataframe()
    data = feature_engineering(df)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = split_data(data)

    # Train the model
    model = train_model(X_train, y_train, train_args)

    # Evaluate and log the metrics returned from the train function
    mape = get_model_metrics(model, X_valid, y_valid)
    run.log("Mean Absolute Percentage Error (MAPE)", mape)

    # Save the trained model
    os.makedirs('outputs', exist_ok=True)
    output_path = os.path.join('outputs', model_name)
    joblib.dump(value=model, filename=output_path)

    run.tag("run_type", value="train")
    print(f"Tags now present for run: {run.tags}")

    # Upload the model to run artifacts
    print("Uploading the model into run artifacts...")
    run.upload_file(name=f"./outputs/models/{model_name}", path_or_stream=output_path)
    print(f"Uploaded the model {model_name} to experiment {run.experiment.name}")
    dirpath = os.getcwd()
    print(dirpath)
    print("Following files are uploaded:")
    print(run.get_file_names())

    run.complete()


if __name__ == '__main__':
    main()
