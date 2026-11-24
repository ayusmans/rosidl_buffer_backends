#!/usr/bin/env python3
# Copyright 2026 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run demo.launch.py and check that both backends deliver intact buffers."""

import json
import os
import sys
import time
import unittest

from ament_index_python.packages import get_package_share_directory
from cuda_buffer import CudaBuffer
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_testing
import launch_testing.actions
import launch_testing.asserts
import launch_testing.markers
import pytest
import rclpy
from std_msgs.msg import String


@pytest.mark.launch_test
@launch_testing.markers.keep_alive
def generate_test_description():
    demo = os.path.join(
        get_package_share_directory('cuda_buffer_examples'), 'launch', 'demo.launch.py')
    # The CMake target passes layout:=... on the command line.
    layout = next(
        (arg.split(':=', 1)[1] for arg in sys.argv if arg.startswith('layout:=')),
        'cross_process')
    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(demo),
            launch_arguments={
                'layout': layout, 'size_bytes': str(256 * 1024), 'rate_hz': '20.0',
            }.items(),
        ),
        launch_testing.actions.ReadyToTest(),
    ]), {'layout': layout}


class TestBufferExamples(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = rclpy.create_node('test_buffer_examples')
        self.stats = {}
        self.node.create_subscription(String, 'buffer_stats', self._on_stats, 10)

    def tearDown(self):
        self.node.destroy_node()

    def _on_stats(self, msg):
        stats = json.loads(msg.data)
        self.stats[stats['topic']] = stats

    def _wait_for(self, received, timeout_sec):
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if all(self.stats.get(t, {}).get('received', 0) >= received
                   for t in ('/buffer_vmm', '/buffer_nvsci')):
                return True
        return False

    def test_both_backends_deliver_intact_buffers(self, layout):
        self.assertTrue(self._wait_for(received=20, timeout_sec=40.0), f'stats: {self.stats}')
        for topic, stats in self.stats.items():
            self.assertEqual(0, stats['corrupt'], f'{topic}: {stats}')
        vmm = self.stats['/buffer_vmm']
        nvsci = self.stats['/buffer_nvsci']
        # VMM is shared zero-copy wherever the device supports VMM IPC.
        self.assertGreater(vmm['vmm'], 0, self.stats)
        self.assertEqual(0, vmm['nvsci'], self.stats)
        if not CudaBuffer.is_storage_available('nvsci'):
            return
        # NvSci buffers never arrive as VMM. Within a process they arrive
        # zero-copy; across processes that also needs NvSciIpc endpoint pairs,
        # and the CPU path is used without them.
        self.assertEqual(0, nvsci['vmm'], self.stats)
        if layout == 'same_process':
            self.assertGreater(nvsci['nvsci'], 0, self.stats)


@launch_testing.post_shutdown_test()
class TestShutdown(unittest.TestCase):

    def test_exit_codes(self, proc_info):
        launch_testing.asserts.assertExitCodes(proc_info, allowable_exit_codes=[0, -2, -15])
