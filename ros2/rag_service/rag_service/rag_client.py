

import rclpy
from rclpy.node import Node
from rag_interfaces.srv import RAGQuery
import sys


class RAGClientExample(Node):

    def __init__(self):
        """Initialize the RAG client node."""
        super().__init__('rag_client')

        # Create a client for the RAG service
        self.client = self.create_client(RAGQuery, 'rag_query')

        # Wait for the service to be available
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for RAG service...')

        self.get_logger().info('RAG service available')

    def send_query(self, question):
        # Create the request
        request = RAGQuery.Request()
        request.question = question

        self.get_logger().info(f'Sending query: {question}')

        # Call the service asynchronously
        future = self.client.call_async(request)

        # Wait for the response
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            return future.result()
        else:
            self.get_logger().error('Service call failed')
            return None

    def print_response(self, response):
        print("RAG SERVICE RESPONSE")

        if response.success:
            print(f"\nAnswer:\n{response.answer}")
            print(f"\nResponse Time: {response.response_time:.2f} seconds")
            print(f"From Cache: {'Yes' if response.from_cache else 'No'}")
        else:
            print(f"\nError: {response.error_message}")
            print(f"Response Time: {response.response_time:.2f} seconds")

        print("=" * 80 + "\n")


def main(args=None):
    """Main entry point for the RAG client example."""
    rclpy.init(args=args)

    # Create the client node
    client_node = RAGClientExample()

    try:
        # Check if a question was provided as a command-line argument
        if len(sys.argv) > 1:
            # Use the provided question
            question = ' '.join(sys.argv[1:])
        else:
            # Interactive mode - ask for questions
            print("\n" )
            print("RAG SERVICE CLIENT")
            print("\n")
            print("Enter your questions (or 'quit' to exit)")
            print("\n")

            while True:
                try:
                    question = input("Question: ").strip()

                    if question.lower() in ['quit', 'exit', 'q']:
                        print("\nExiting...")
                        break

                    if not question:
                        print("Please enter a valid question.\n")
                        continue

                    # Send the query
                    response = client_node.send_query(question)

                    if response:
                        client_node.print_response(response)

                except KeyboardInterrupt:
                    print("\n\nExiting...")
                    break

            client_node.destroy_node()
            rclpy.shutdown()
            return

        # Single question mode
        response = client_node.send_query(question)

        if response:
            client_node.print_response(response)

    except Exception as e:
        client_node.get_logger().error(f'Error: {str(e)}')
    finally:
        client_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
