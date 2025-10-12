#!/usr/bin/env python3
"""
Launch file for RAG Service Node

This launch file starts the RAG service node with configurable parameters.
You can customize the PDF paths, vector database paths, and other settings.

Usage:
    ros2 launch rag_service rag_service_launch.py

    Or with custom parameters:
    ros2 launch rag_service rag_service_launch.py knowledge1_pdf:=custom_knowledge1.pdf
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for RAG service."""

    # Declare launch arguments with default values (paths relative to package root)
    knowledge1_pdf_arg = DeclareLaunchArgument(
        'knowledge1_pdf',
        default_value='data/pdfs/knowledge1.pdf',
        description='Path to the first knowledge base PDF file (relative to package root)'
    )

    knowledge2_pdf_arg = DeclareLaunchArgument(
        'knowledge2_pdf',
        default_value='data/pdfs/knowledge2.pdf',
        description='Path to the second knowledge base PDF file (relative to package root)'
    )

    vector_db1_path_arg = DeclareLaunchArgument(
        'vector_db1_path',
        default_value='data/vector_dbs/chroma_knowledge1_db',
        description='Path to the first vector database (relative to package root)'
    )

    vector_db2_path_arg = DeclareLaunchArgument(
        'vector_db2_path',
        default_value='data/vector_dbs/chroma_knowledge2_db',
        description='Path to the second vector database (relative to package root)'
    )

    chunk_size_arg = DeclareLaunchArgument(
        'chunk_size',
        default_value='400',
        description='Chunk size for text splitting'
    )

    chunk_overlap_arg = DeclareLaunchArgument(
        'chunk_overlap',
        default_value='50',
        description='Chunk overlap for text splitting'
    )

    retrieval_k_arg = DeclareLaunchArgument(
        'retrieval_k',
        default_value='3',
        description='Number of documents to retrieve (k parameter)'
    )

    # Create the RAG service node
    rag_service_node = Node(
        package='rag_service',
        executable='rag_service_node',
        name='rag_service_node',
        output='screen',
        parameters=[{
            'knowledge1_pdf': LaunchConfiguration('knowledge1_pdf'),
            'knowledge2_pdf': LaunchConfiguration('knowledge2_pdf'),
            'vector_db1_path': LaunchConfiguration('vector_db1_path'),
            'vector_db2_path': LaunchConfiguration('vector_db2_path'),
            'chunk_size': LaunchConfiguration('chunk_size'),
            'chunk_overlap': LaunchConfiguration('chunk_overlap'),
            'retrieval_k': LaunchConfiguration('retrieval_k'),
        }],
        emulate_tty=True
    )

    return LaunchDescription([
        knowledge1_pdf_arg,
        knowledge2_pdf_arg,
        vector_db1_path_arg,
        vector_db2_path_arg,
        chunk_size_arg,
        chunk_overlap_arg,
        retrieval_k_arg,
        rag_service_node
    ])
