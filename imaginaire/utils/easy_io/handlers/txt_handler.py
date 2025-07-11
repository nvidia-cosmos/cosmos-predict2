# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from imaginaire.utils.easy_io.handlers.base import BaseFileHandler


class TxtHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        del kwargs
        return file.read()

    def dump_to_fileobj(self, obj, file, **kwargs):
        del kwargs
        if not isinstance(obj, str):
            obj = str(obj)
        file.write(obj)

    def dump_to_str(self, obj, **kwargs):
        del kwargs
        if not isinstance(obj, str):
            obj = str(obj)
        return obj
