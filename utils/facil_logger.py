import datetime
import json
import sys
import time
import warnings
from typing import List, TYPE_CHECKING, Tuple, Type, Optional, TextIO

import torch

from avalanche.core import SupervisedPlugin, BasePlugin
from avalanche.evaluation.metric_results import MetricValue, TensorImage
from avalanche.logging import BaseLogger
from avalanche.evaluation.metric_utils import stream_type, phase_and_task
from avalanche.training.plugins import EvaluationPlugin
from models.resnet_18 import ResNet

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

UNSUPPORTED_TYPES: Tuple[Type] = (TensorImage,)


class FileOutputDuplicator(object):
    def __init__(self, duplicate, fname, mode):
        self.file = open(fname, mode)
        self.duplicate = duplicate

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.duplicate.write(data)

    def flush(self):
        self.file.flush()
        self.duplicate.flush()


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Add other handling of logging
        if issubclass(type(obj), BasePlugin):
            return obj.__class__.__name__

        if type(obj) == EvaluationPlugin:
            return EvaluationPlugin.__name__

        if type(obj) == ResNet:
            return "ResNet"

        return str(obj)


class FACIL_Logger(BaseLogger, SupervisedPlugin):
    """
        This Logger should log results similar to the FACIL framework compact stdout
        https://github.com/mmasana/FACIL/tree/master/src/loggers
    """

    def __init__(self, file=sys.stdout):
        """
        Creates an instance of `TextLogger` class.

        :param file: destination file to which print metrics
            (default=sys.stdout).
        """
        super().__init__()

        self.file = file
        self.metric_vals = {}
        self.started_training = False
        self.start_time = None
        self.train_time_now = 0

    def log_single_metric(self, name, value, x_plot) -> None:
        # We only keep track of the last value for each metric
        # We filter the class accuracies
        if "ClassAcc" not in name and "U_Exp" not in name and "V_Exp" not in name and "W_Exp" not in name and "V_After_Exp" not in name:
            self.metric_vals[name] = (name, x_plot, value)

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return "\n" + str(m_val)
        elif isinstance(m_val, float):
            return f"{m_val:.4f}"
        else:
            return str(m_val)

    def print_current_metrics(self):
        sorted_vals = sorted(self.metric_vals.values(), key=lambda x: x[0])
        for name, x, val in sorted_vals:
            if isinstance(val, UNSUPPORTED_TYPES):
                continue
            val = self._val_to_str(val)
            print(f" {name.strip().split('/')[0]} = {val}", end=" |", file=self.file, flush=True)
        print(file=self.file, flush=True)

    def before_training_exp(self, strategy: "SupervisedTemplate", metric_values: List["MetricValue"], **kwargs):
        super().before_training_exp(strategy, metric_values, **kwargs)
        self._on_exp_start(strategy)
        self.started_training = True

    def before_eval_exp(self, strategy: "SupervisedTemplate", metric_values: List["MetricValue"], **kwargs, ):
        super().before_eval_exp(strategy, metric_values, **kwargs)
        self._on_exp_start(strategy)

    def after_training_epoch(self, strategy: "SupervisedTemplate", metric_values: List["MetricValue"], **kwargs, ):
        super().after_training_epoch(strategy, metric_values, **kwargs)
        cur_time = time.time()
        self.train_time_now += cur_time - self.start_time
        self.start_time = cur_time
        print(f"|Train | Time {self.train_time_now / 60:.2f} min | Experience {strategy.clock.train_exp_counter + 1} "
              f"| Epoch {strategy.clock.train_exp_epochs + 1} |", end="", file=self.file, flush=True, )
        self.print_current_metrics()
        self.metric_vals = {}

    def after_eval_exp(self, strategy: "SupervisedTemplate", metric_values: List["MetricValue"], **kwargs, ):
        super().after_eval_exp(strategy, metric_values, **kwargs)

    def before_training(self, strategy: "SupervisedTemplate", metric_values: List["MetricValue"], **kwargs, ):
        super().before_training(strategy, metric_values, **kwargs)
        print("-- >> Start of training phase << --", file=self.file, flush=True)
        self.start_time = time.time()

    def before_eval(self, strategy: "SupervisedTemplate", metric_values: List["MetricValue"], **kwargs, ):
        super().before_eval(strategy, metric_values, **kwargs)

    def after_training(self, strategy: "SupervisedTemplate", metric_values: List["MetricValue"], **kwargs, ):
        super().after_training(strategy, metric_values, **kwargs)
        print("-- >> End of training phase << --", file=self.file, flush=True)
        self.started_training = False
        self.train_time_now += time.time() - self.start_time

    def after_eval(self, strategy: "SupervisedTemplate", metric_values: List["MetricValue"], **kwargs, ):
        super().after_eval(strategy, metric_values, **kwargs)
        e_id = strategy.clock.train_exp_epochs
        print(f"|Eval  | Time {self.train_time_now / 60:.2f} min | Experience {strategy.clock.train_exp_counter} "
              f"| Epoch {e_id} | Stream {stream_type(strategy.experience)} |", end="", file=self.file, flush=True, )
        self.print_current_metrics()
        self.metric_vals = {}

    def _on_exp_start(self, strategy: "SupervisedTemplate"):
        action_name = "training" if strategy.is_training else "eval"
        exp_id = strategy.experience.current_experience
        task_id = phase_and_task(strategy)[1]
        stream = stream_type(strategy.experience)
        if action_name == "training":
            if task_id is None:
                print("-- Starting {} on experience {} from {} stream --".format(action_name, exp_id, stream), file=self.file, flush=True, )
            else:
                print("-- Starting {} on experience {} (Task {}) from {} stream --".format(action_name, exp_id, task_id, stream), file=self.file, flush=True, )

    def __getstate__(self):
        # Implementation of pickle serialization
        out = self.__dict__.copy()
        fobject_serialized_def = FACIL_Logger._fobj_serialize(out['file'])
        if fobject_serialized_def is not None:
            out['file'] = fobject_serialized_def
        else:
            warnings.warn(f'Cannot properly serialize the file object used for text '
                          f'logging: {out["file"]}.')
        return out

    def __setstate__(self, state):
        # Implementation of pickle deserialization
        fobj = FACIL_Logger._fobj_deserialize(state['file'])

        if fobj is not None:
            state['file'] = fobj
        else:
            raise RuntimeError(f'Cannot deserialize file object {state["file"]}')
        self.__dict__ = state
        self.on_checkpoint_resume()

    def on_checkpoint_resume(self):
        # https://stackoverflow.com/a/25887393
        utc_dt = datetime.datetime.now(datetime.timezone.utc)  # UTC time
        now_w_timezone = utc_dt.astimezone()  # local time
        print(f'[{self.__class__.__name__}] Resuming from checkpoint.', f'Current time is',
            now_w_timezone.strftime("%Y-%m-%d %H:%M:%S %z"), file=self.file, flush=True)

    @staticmethod
    def _fobj_serialize(file_object) -> Optional[str]:
        is_notebook = False
        try:
            is_notebook = file_object.__class__.__name__ == 'OutStream' and 'ipykernel' in file_object.__class__.__module__
        except Exception:
            pass

        if is_notebook:
            # Running in a notebook
            out_file_path = None
            stream_name = 'stdout'
        else:
            # Standard file object
            out_file_path = FACIL_Logger._file_get_real_path(file_object)
            stream_name = FACIL_Logger._file_get_stream(file_object)

        if out_file_path is not None:
            return 'path:' + str(out_file_path)
        elif stream_name is not None:
            return 'stream:' + stream_name
        else:
            return None

    @staticmethod
    def _fobj_deserialize(file_def: str) -> Optional[TextIO]:
        if not isinstance(file_def, str):
            # Custom object (managed by pickle or dill library)
            return file_def

        if file_def.startswith('path:'):
            file_def = _remove_prefix(file_def, 'path:')
            return open(file_def, 'a')
        elif file_def.startswith('stream:'):
            file_def = _remove_prefix(file_def, 'stream:')
            if file_def == 'stdout':
                return sys.stdout
            elif file_def == 'stderr':
                return sys.stderr

        return None

    @staticmethod
    def _file_get_real_path(file_object) -> Optional[str]:
        try:
            if hasattr(file_object, 'file'):
                # Manage files created by tempfile
                file_object = file_object.file
            fobject_path = file_object.name
            if fobject_path in ['<stdout>', '<stderr>']:
                return None
            return fobject_path
        except AttributeError:
            return None

    @staticmethod
    def _file_get_stream(file_object) -> Optional[str]:
        if file_object == sys.stdout or file_object == sys.__stdout__:
            return 'stdout'
        if file_object == sys.stderr or file_object == sys.__stderr__:
            return 'stderr'
        return None


def _remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text  # or whatever


__all__ = ['FACIL_Logger']
