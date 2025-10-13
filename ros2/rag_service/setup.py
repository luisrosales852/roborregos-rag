from setuptools import setup
import os
from glob import glob

package_name = 'rag_service'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RoboRregos Team',
    maintainer_email='A01255674@tec.mx',
    description='ROS2 service wrapper for RAG (Retrieval-Augmented Generation) system',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rag_service_node = rag_service.rag_service_node:main',
            'rag_client_example = rag_service.rag_client_example:main',
        ],
    },
)
