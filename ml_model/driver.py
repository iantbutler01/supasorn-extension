import tensorflow as tf
from model import Seq2Shape
from tensorflow import estimator
from sys import argv
from pprint import pprint
from tensorflow.python.ops import lookup_ops
from inputs import ModelInputs
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.learn import learn_runner, RunConfig, Experiment
from tensorflow.contrib.training import HParams
import numpy as np
import json

with open('./hyperparameters.json', 'r') as f:
    HPARAMS = json.load(f)

def model_fn(features, labels, mode, params, config):

    model = Seq2Shape(params.batch_size, 20, mode)
    if mode == estimator.ModeKeys.TRAIN or mode == estimator.ModeKeys.EVAL:
        t_out = model.translate(params.num_units, features)
        spec = model.prepare_train_eval(t_out, params.num_units, 100, labels, params.learning_rate)
    if mode == estimator.ModeKeys.PREDICT:
        p_out = model.translate(params.num_units, features)
        spec = model.prepare_predict(p_out)
    return spec

def experiment_fn(run_config, hparams):
    input_fn_factory = ModelInputs(hparams.batch_size, hparams.train_dataset_path)
    train_input_fn, train_input_hook = input_fn_factory.get_inputs()
    eval_input_fn, eval_input_hook = input_fn_factory.get_inputs(mode=estimator.ModeKeys.EVAL)
    exp_estimator = get_estimator(run_config, hparams)
    run_config.replace(save_checkpoints_steps=hparams.min_eval_frequency)

    return Experiment(
        estimator=exp_estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=hparams.num_steps,
        min_eval_frequency=hparams.min_eval_frequency,
        train_monitors=[train_input_hook],
        eval_hooks=[eval_input_hook],
        eval_steps=20000
    )

def get_estimator(run_config, hparams):
    return estimator.Estimator(
        model_fn=model_fn,
        params=hparams,
        config=run_config,
    )

def print_predictions(predictions, hparams):
    for pred in predictions:
        print(pred)

def main():
    hparams = HParams(**HPARAMS)
    run_config = RunConfig(model_dir='./save')

    if len(argv) < 2 or argv[1] == 'train':
        learn_runner.run(
            experiment_fn=experiment_fn,
            run_config=run_config,
            schedule="train_and_evaluate",
            hparams=hparams,
        )
    elif argv[1] == 'predict':
        input_fn_factory = ModelInputs(hparams.batch_size, hparams.train_dataset_path)
        predict_input_fn, predict_input_hook = input_fn_factory.get_inputs( mode=tf.estimator.ModeKeys.PREDICT)
        classifier = get_estimator(run_config, hparams)
        predictions = classifier.predict(input_fn=predict_input_fn, hooks=[predict_input_hook])
        print_predictions(predictions, hparams)
    else:
        print('Unknown Operation.')

if __name__ == '__main__':
    main()
