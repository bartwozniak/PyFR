# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import re
import uuid

import numpy as np


class BaseReader(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _to_raw_pyfrm(self):
        pass

    def to_pyfrm(self):
        mesh = self._to_raw_pyfrm()

        # Add metadata
        mesh['mesh_uuid'] = str(uuid.uuid4())

        return mesh
