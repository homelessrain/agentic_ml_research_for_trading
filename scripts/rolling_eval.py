import argparse

from utils.script_utils import build_param_overrides, resolve_model_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--param', action='append', default=[], metavar='KEY=VALUE')
    # each --eval should follow the format of --eval eval_window_1=train_start_datestr,train_end_datestr,test_start_datestr,test_end_datestr
    parser.add_argument('--eval', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()

    init_params = build_param_overrides(args.param)
    eval_windows = build_param_overrides(args.eval)

    for eval_window_key in eval_windows:
        train_start_datestr, train_end_datestr, test_start_datestr, test_end_datestr = eval_windows[eval_window_key].split(',')
        print(f"Evaluating {eval_window_key}: train from {train_start_datestr} to {train_end_datestr}, test from {test_start_datestr} to {test_end_datestr}...")
        model_class = resolve_model_class(args.model)
        model = model_class(**init_params)

        training_metrics, validation_metrics = model.train(train_start_datestr, train_end_datestr)
        testing_metrics = model.test(test_start_datestr, test_end_datestr)

        print(f"Training metrics: {training_metrics}")
        print(f"Validation metrics: {validation_metrics}")
        print(f"Testing metrics: {testing_metrics}")
