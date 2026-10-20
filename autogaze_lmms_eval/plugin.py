"""Entry-point target for lmms-eval's model registry.

Kept separate from client.py so registering the model imports nothing heavy:
lmms-eval loads every entry point while building its registry and silently
drops any that raise, which would make `--model autogaze` vanish. The class
itself is imported only when lmms-eval actually instantiates it.
"""
from lmms_eval.models.registry_v2 import ModelManifest

manifest = ModelManifest(model_id="autogaze", simple_class_path="autogaze_lmms_eval.client.AutoGaze")
