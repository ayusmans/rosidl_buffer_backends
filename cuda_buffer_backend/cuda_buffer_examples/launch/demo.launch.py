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

"""
Send CUDA buffers with the VMM and NvSci storage backends side by side.

Two talkers publish on /buffer_vmm and /buffer_nvsci, each allocating from its
own backend, and one listener subscribes to both and reports how every message
arrived. The talkers always share one process, so the two backends are used
in the same process.

  layout:=cross_process  (default) the listener runs in a second process
  layout:=same_process   everything runs in one process
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

LAYOUTS = ('cross_process', 'same_process')


def talkers(size_bytes, rate_hz):
    return [
        ComposableNode(
            package='cuda_buffer_examples',
            plugin='cuda_buffer_examples::BufferTalker',
            name=f'talker_{storage}',
            parameters=[{
                'storage': storage,
                'topic': f'buffer_{storage}',
                'size_bytes': size_bytes,
                'rate_hz': rate_hz,
            }],
        )
        for storage in ('vmm', 'nvsci')
    ]


def listener():
    return ComposableNode(
        package='cuda_buffer_examples',
        plugin='cuda_buffer_examples::BufferListener',
        name='listener',
        parameters=[{'topics': ['buffer_vmm', 'buffer_nvsci']}],
    )


def container(name, nodes):
    return ComposableNodeContainer(
        name=name,
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=nodes,
        output='screen',
    )


def launch_setup(context):
    layout = LaunchConfiguration('layout').perform(context)
    if layout not in LAYOUTS:
        raise ValueError(f'layout must be one of {LAYOUTS}, not {layout!r}')
    size_bytes = int(LaunchConfiguration('size_bytes').perform(context))
    rate_hz = float(LaunchConfiguration('rate_hz').perform(context))

    if layout == 'same_process':
        return [container('buffer_demo', talkers(size_bytes, rate_hz) + [listener()])]
    return [
        container('buffer_talkers', talkers(size_bytes, rate_hz)),
        container('buffer_listener', [listener()]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'layout', default_value='cross_process', description=' or '.join(LAYOUTS)),
        DeclareLaunchArgument(
            'size_bytes', default_value=str(1 << 20), description='payload size per message'),
        DeclareLaunchArgument(
            'rate_hz', default_value='10.0', description='publish rate of each talker'),
        # The buffer backends are implemented for Fast DDS.
        SetEnvironmentVariable('RMW_IMPLEMENTATION', 'rmw_fastrtps_cpp'),
        OpaqueFunction(function=launch_setup),
    ])
