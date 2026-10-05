import builtins
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parent
MAIN_PATH = PROJECT_DIR / "main.py"


class FakeDataset:
    class_names = ["backpack", "laptop"]

    def prefetch(self, buffer_size):
        return self


class FakeModel:
    def __init__(self, evaluate_result=(0.25, 0.85)):
        self.evaluate_result = evaluate_result
        self.compiled = False
        self.fit_called = False

    def compile(self, **kwargs):
        self.compiled = True
        self.compile_kwargs = kwargs

    def summary(self):
        return None

    def fit(self, *args, **kwargs):
        self.fit_called = True
        return object()

    def evaluate(self, *args, **kwargs):
        return self.evaluate_result


class FakeLayer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeSequential:
    def __init__(self, layers):
        self.layers = layers
        self.compiled = False

    def compile(self, **kwargs):
        self.compiled = True
        self.compile_kwargs = kwargs

    def summary(self):
        return None


class FakeCallback:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def install_fake_dependencies():
    tf = types.ModuleType("tensorflow")

    class FakeConfig:
        @staticmethod
        def list_physical_devices(kind):
            return []

    class FakeUtils:
        @staticmethod
        def image_dataset_from_directory(*args, **kwargs):
            return FakeDataset()

    class FakeData:
        AUTOTUNE = object()

    class FakeLayers:
        RandomFlip = FakeLayer
        RandomRotation = FakeLayer
        RandomZoom = FakeLayer
        Rescaling = FakeLayer
        Conv2D = FakeLayer
        MaxPooling2D = FakeLayer
        Flatten = FakeLayer
        Dense = FakeLayer
        Dropout = FakeLayer

    class FakeKeras:
        Sequential = FakeSequential
        layers = FakeLayers
        utils = FakeUtils
        callbacks = types.SimpleNamespace(
            ModelCheckpoint=FakeCallback,
            EarlyStopping=FakeCallback,
        )

    tf.config = FakeConfig
    tf.keras = FakeKeras
    tf.data = FakeData

    coco_builder = types.ModuleType("coco_builder")
    coco_builder.build_coco_dataset = lambda: None

    pick_module = types.ModuleType("pick")
    pick_module.pick = lambda options, *args, **kwargs: (options[0], 0)

    tabulate_module = types.ModuleType("tabulate")
    tabulate_module.tabulate = lambda rows, headers=None, tablefmt=None: "TABLE"

    sys.modules["tensorflow"] = tf
    sys.modules["coco_builder"] = coco_builder
    sys.modules["pick"] = pick_module
    sys.modules["tabulate"] = tabulate_module


def load_main_module():
    install_fake_dependencies()
    sys.modules.pop("main_under_test", None)
    spec = importlib.util.spec_from_file_location("main_under_test", MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["main_under_test"] = module
    spec.loader.exec_module(module)
    return module


class TestProjectOneAndTwo(unittest.TestCase):
    def setUp(self):
        self.main = load_main_module()

    def test_all_settings_menu_options_are_callable(self):
        for name, function in self.main.settings_menu.items():
            self.assertTrue(callable(function), f"{name} is not callable")

    def test_all_learn_menu_options_are_callable(self):
        for name, function in self.main.learn_menu.items():
            self.assertTrue(callable(function), f"{name} is not callable")

    def test_change_numeric_settings(self):
        with patch.object(self.main, "give_time"):
            with patch("builtins.input", side_effect=["12"]):
                self.main.change_epochs()
            with patch("builtins.input", side_effect=["8"]):
                self.main.change_batch_size()
            with patch("builtins.input", side_effect=["4"]):
                self.main.change_patience()

        self.assertEqual(self.main.EPOCHS, 12)
        self.assertEqual(self.main.BATCH_SIZE, 8)
        self.assertEqual(self.main.PATIENCE, 4)

    def test_change_image_size(self):
        with patch.object(self.main, "give_time"), patch.object(
            self.main, "choose_image_size", return_value=(224, 224)
        ):
            self.main.change_image_size()

        self.assertEqual(self.main.IMAGE_SIZE, (224, 224))

    def test_dropout_setting_is_togglable(self):
        self.main.dropout_status = "INACTIVE"

        with patch.object(self.main, "give_time"):
            self.main.change_dropout()
            self.assertEqual(self.main.dropout_status, "ACTIVE")
            self.main.change_dropout()
            self.assertEqual(self.main.dropout_status, "INACTIVE")

    def test_save_and_load_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "settings.json"
            self.main.SETTINGS_FILE = settings_file
            self.main.saved_settings = {}

            self.main.EPOCHS = 7
            self.main.IMAGE_SIZE = (192, 192)
            self.main.BATCH_SIZE = 16
            self.main.PATIENCE = 2
            self.main.dropout_status = "ACTIVE"

            with patch.object(self.main, "give_time"), patch(
                "builtins.input", return_value="My Profile"
            ):
                self.main.save_current_settings()

            self.assertTrue(settings_file.exists())
            saved = json.loads(settings_file.read_text())
            self.assertEqual(saved["My Profile"]["image_size"], [192, 192])

            loaded = self.main.load_saved_settings()
            self.assertEqual(loaded["My Profile"]["batch_size"], 16)

    def test_create_new_profile_saves_expected_types(self):
        self.main.saved_settings = {}

        pick_results = [
            ("160x160", 3),
            ("ACTIVE", 0),
        ]

        def fake_pick(options, *args, **kwargs):
            return pick_results.pop(0)

        with patch.object(self.main, "clear_terminal"), patch.object(
            self.main, "give_time"
        ), patch.object(self.main, "save_saved_settings"), patch.object(
            self.main, "pick", side_effect=fake_pick
        ), patch(
            "builtins.input",
            side_effect=["Profile A", "5", "16", "2"],
        ):
            self.main.create_new_profile()

        profile = self.main.saved_settings["Profile A"]
        self.assertEqual(profile["epochs"], 5)
        self.assertEqual(profile["image_size"], (160, 160))
        self.assertEqual(profile["batch_size"], 16)
        self.assertEqual(profile["patience"], 2)
        self.assertEqual(profile["dropout"], "ACTIVE")

    def test_build_model_uses_dropout_when_active(self):
        self.main.dropout_status = "ACTIVE"
        model = self.main.build_model(3)
        self.assertTrue(
            any(isinstance(layer, FakeLayer) and layer.kwargs.get("rate") == 0.5
                for layer in model.layers),
            "Dropout setting is exposed but does not affect the model",
        )

    def test_test_model_extracts_accuracy_from_keras_result(self):
        model = FakeModel(evaluate_result=(0.20, 0.90))
        with patch.object(
            self.main.tf.keras.utils,
            "image_dataset_from_directory",
            return_value=FakeDataset(),
        ):
            accuracy = self.main.test_model(model)

        self.assertAlmostEqual(accuracy, 0.90)

    def test_default_training_path_records_result(self):
        calls = []

        def record(name):
            def wrapper(*args, **kwargs):
                calls.append(name)
                if name == "test_model":
                    return 0.91
                if name == "create_run_record":
                    return {"accuracy": args[0]}
                return args[0] if args else None
            return wrapper

        class StopMain(Exception):
            pass

        choices = ["Defaults"]

        def fake_pick(options, *args, **kwargs):
            if choices:
                return choices.pop(0), 0
            raise StopMain

        with patch.object(self.main, "clear_terminal"), patch.object(
            self.main, "pick", side_effect=fake_pick
        ), patch.object(self.main, "load_training_data", side_effect=record("load_training_data")), patch.object(
            self.main, "load_validation_data", side_effect=record("load_validation_data")
        ), patch.object(self.main, "prepare_data", side_effect=record("prepare_data")), patch.object(
            self.main, "build_model", side_effect=record("build_model")
        ), patch.object(self.main, "compile", side_effect=record("compile")), patch.object(
            self.main, "show_model", side_effect=record("show_model")
        ), patch.object(self.main, "training_callbacks", side_effect=record("training_callbacks")), patch.object(
            self.main, "early_stopping", side_effect=record("early_stopping")
        ), patch.object(self.main, "train", side_effect=record("train")), patch.object(
            self.main, "test_model", side_effect=record("test_model")
        ), patch.object(self.main, "create_run_record", side_effect=record("create_run_record")), patch.object(
            self.main, "save_training_run", side_effect=record("save_training_run")
        ):
            with self.assertRaises(StopMain):
                self.main.main()

        self.assertEqual(calls[0], "load_training_data")
        self.assertIn("test_model", calls)
        self.assertIn("save_training_run", calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
