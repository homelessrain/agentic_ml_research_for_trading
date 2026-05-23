import argparse

from utils.script_utils import build_param_overrides, resolve_model_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--train-start', type=str, required=False, default='2026-01-01')
    parser.add_argument('--train-end', type=str, required=False, default='2026-03-31')
    parser.add_argument('--test-start', type=str, required=False, default='2026-04-01')
    parser.add_argument('--test-end', type=str, required=False, default='2026-04-30')
    parser.add_argument('--param', action='append', default=[], metavar='KEY=VALUE')
    args = parser.parse_args()

    # Get custom parameters to initialize model
    init_params = build_param_overrides(args.param)

    model_class = resolve_model_class(args.model)
    model = model_class(**init_params)

    training_metrics, validation_metrics = model.train(args.train_start, args.train_end)
    testing_metrics = model.test(args.test_start, args.test_end)

    print(f"Training metrics: {training_metrics}")
    print(f"Validation metrics: {validation_metrics}")
    print(f"Testing metrics: {testing_metrics}")
