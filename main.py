# main.py
# Entry point for NLP-Challenge training demos.

import importlib.util
import sys
from pathlib import Path


def import_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    print("Starting NLP-Challenge Demos")

    base = Path(__file__).parent

    # Load fragments using path-safe module names.
    setup = import_module_from_path("setup", base / "setup.py")
    batching = import_module_from_path("batching", base / "batching.py")
    core_modules = import_module_from_path("core_modules", base / "core_modules.py")
    models = import_module_from_path("models_bert_bart", base / "models_bert_bart.py")
    gpt = import_module_from_path("model_gpt_skeleton", base / "model_gpt_skeleton.py")
    training = import_module_from_path("training_utils_and_demos", base / "training_utils_and_demos.py")

    # Run the demo orchestration function
    training.run_all_demos(steps=20)


if __name__ == "__main__":
    main()
