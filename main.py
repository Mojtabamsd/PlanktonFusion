import argparse
import os
import sys
from tools import console
from data_preparation.sampling import sampling
from data_preparation.sampling_syn import sampling_syn
from train.train import train_nn
from inference.prediction import prediction
from inference.prediction_auto import prediction_auto
from feature_extraction.train_autoencoder import train_autoencoder
from feature_extraction.ssl import train_ssl
from feature_extraction.classifier import classifier
from train.train_memory_attention import train_memory
from train.train_contrastive import train_contrastive


def add_arguments(obj):
    obj.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default="./config/config.yaml",
        help="Path to the configuration file 'config.yaml'",
    )
    obj.add_argument(
        "-i",
        "--input_folder",
        type=str,
        help="Input path to load data.",
    )
    obj.add_argument(
        "-o",
        "--output_folder",
        default="results",
        type=str,
        help="Output path to write the results.",
    )


def main(args=None):

    parser = argparse.ArgumentParser(description="Plankton Fusion Classifier")
    subparsers = parser.add_subparsers()

    # sampling
    parser_sampling = subparsers.add_parser(
        "sampling",
        help="Prepare dataset ready for training.",
    )
    add_arguments(parser_sampling)
    parser_sampling.set_defaults(func=call_sampling)

    # sampling synthetic
    parser_sampling_syn = subparsers.add_parser(
        "sampling_syn",
        help="Prepare real and synthetic dataset ready for training.",
    )
    add_arguments(parser_sampling_syn)
    parser_sampling_syn.set_defaults(func=call_sampling_syn)

    # training
    parser_training = subparsers.add_parser(
        "training", help="Train a classifier"
    )
    add_arguments(parser_training)
    parser_training.set_defaults(func=call_training)

    # training contrastive
    parser_training_contrastive = subparsers.add_parser(
        "training_contrastive", help="Train a contrastive architecture"
    )
    add_arguments(parser_training_contrastive)
    parser_training_contrastive.set_defaults(func=call_training_contrastive)

    # prediction
    parser_prediction = subparsers.add_parser(
        "prediction", help="Prediction from trained model"
    )
    add_arguments(parser_prediction)
    parser_prediction.set_defaults(func=call_prediction)

    # prediction autoencoder
    parser_prediction_auto = subparsers.add_parser(
        "prediction_auto", help="Prediction from autoencoder reconstruction model"
    )
    add_arguments(parser_prediction_auto)
    parser_prediction_auto.set_defaults(func=call_prediction_auto)

    # autoencoder
    parser_autoencoder = subparsers.add_parser(
        "autoencoder", help="Train an Autoencoder"
    )
    add_arguments(parser_autoencoder)
    parser_autoencoder.set_defaults(func=call_autoencoder)

    # ssl
    parser_ssl = subparsers.add_parser(
        "ssl", help="Train an ssl"
    )
    add_arguments(parser_ssl)
    parser_ssl.set_defaults(func=call_ssl)

    # classifier
    parser_classifier = subparsers.add_parser(
        "classifier", help="Train a classifier from features"
    )
    add_arguments(parser_classifier)
    parser_classifier.set_defaults(func=call_classifier)

    # memory attention
    parser_memory = subparsers.add_parser(
        "memory", help="Train a memory attention"
    )
    add_arguments(parser_memory)
    parser_memory.set_defaults(func=call_memory)

    if len(sys.argv) == 1 and args is None:
        # Show help if no args provided
        parser.print_help(sys.stderr)
        sys.exit(2)
    args = parser.parse_args(args)
    args.func(args)


def call_sampling(args):
    sampling(args.configuration_file)


def call_sampling_syn(args):
    sampling_syn(args.configuration_file)


def call_training(args):
    train_nn(args.configuration_file, args.input_folder, args.output_folder)


def call_training_contrastive(args):
    train_contrastive(args.configuration_file, args.input_folder, args.output_folder)


def call_prediction(args):
    prediction(args.configuration_file, args.input_folder, args.output_folder)


def call_prediction_auto(args):
    prediction_auto(args.configuration_file, args.input_folder, args.output_folder)


def call_autoencoder(args):
    train_autoencoder(args.configuration_file, args.input_folder, args.output_folder)


def call_ssl(args):
    train_ssl(args.configuration_file, args.input_folder, args.output_folder)


def call_classifier(args):
    classifier(args.configuration_file, args.input_folder, args.output_folder)


def call_memory(args):
    train_memory(args.configuration_file, args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
