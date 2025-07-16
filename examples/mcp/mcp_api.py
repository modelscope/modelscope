#!/usr/bin/env python3
"""
Example usage of MCPManager with OpenAI API for multi-turn conversations.

This example demonstrates how to use MCPManager with OpenAI-compatible API
for multi-turn conversations with tool calling capabilities. It shows
the integration between ModelScope MCP tools and OpenAI API endpoints.

Features:
- Multi-turn conversation with tool calling
- Streaming response handling
- Tool execution with timeout protection
- Service registry query capabilities
- Comprehensive error handling
"""

import concurrent.futures
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
from openai import OpenAI

from modelscope.hub.modelscope_mcp.manager import MCPManager  # type: ignore

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'modelscope'))

# Configure logging
logging.basicConfig(
    # level=logging.WARNING,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Set ModelScope token
os.environ['MODELSCOPE_SDK_TOKEN'] = 'Your Modelscope Token'


class MCPAPIConversationExample:
    """Example class demonstrating MCPManager with OpenAI API integration."""

    def __init__(self):
        """Initialize the example with configuration."""
        self.api_key = 'Your Modelscope Token'
        self.base_url = 'https://api-inference.modelscope.cn/v1/'
        self.mcp_config = 'mcp_config.json'
        self.manager: Optional[MCPManager] = None
        self.client: Optional[OpenAI] = None

    def setup_mcp_manager(self) -> None:
        """Set up MCPManager instance."""
        if MCPManager is None:
            raise RuntimeError(
                'MCPManager not available. Please ensure modelscope is properly installed.'
            )

        logger.info('Creating MCPManager instance...')
        self.manager = MCPManager(
            mcp_config=self.mcp_config,
            warmup_connect=True,  # Enable warmup connection mode
            modelscope_token=self.api_key,
            use_intl_site=False)
        logger.info('MCPManager created successfully')

    def setup_openai_client(self) -> None:
        """Set up OpenAI client."""
        logger.info('Creating OpenAI client...')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info('OpenAI client created successfully')

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools."""
        if not self.manager:
            raise RuntimeError('MCPManager not initialized')

        logger.info('Getting available tools...')
        tools = self.manager.get_tools()
        logger.info(f'Found {len(tools)} tools')

        return tools

    def build_system_message(self) -> str:
        """Build system message with tool information."""
        if not self.manager:
            raise RuntimeError('MCPManager not initialized')

        # Get service brief information (includes usage rules)
        service_brief = mcp_manager.get_service_brief_for_prompt(
        )  # type: ignore

        system_prompt = f"""You can use the following tools to help users.

    {service_brief}

    Please handle user requests efficiently and concisely."""

        return system_prompt

    def execute_tool_with_timeout(self, tool_name: str,
                                  arguments: Dict[str, Any]) -> str:
        """Execute tool with timeout protection."""
        if not self.manager:
            raise RuntimeError('MCPManager not initialized')

        try:
            # Use thread pool to execute tool call, avoiding blocking
            with concurrent.futures.ThreadPoolExecutor() as executor:

                def call_tool():
                    # Try to get tool by name first
                    tool = self.manager.get_tool_by_name(
                        tool_name)  # type: ignore

                    if not tool:
                        # If not found, try querying service registry
                        logger.info(
                            f'Tool {tool_name} not found, trying service registry query...'
                        )
                        services = self.manager.query_service_registry(
                            [tool_name])  # type: ignore
                        if services:
                            logger.info(
                                f"Found related services: {[s['name'] for s in services]}"
                            )
                            # Use first matching service
                            matched_tool_name = services[0]['name']
                            tool = self.manager.get_tool_by_name(
                                matched_tool_name)  # type: ignore

                    if not tool:
                        return f'Error: Tool {tool_name} not found'
                    return tool.call(arguments)

                future = executor.submit(call_tool)
                try:
                    result = future.result(timeout=180)  # 3 minutes timeout
                except concurrent.futures.TimeoutError:
                    result = f'Tool call timeout: {tool_name}'
                    logger.warning(f'Tool call timeout: {tool_name}')
                except Exception as e:
                    result = f'Tool call failed: {str(e)}'
                    logger.error(f'Tool call exception: {str(e)}')

                return result

        except Exception as e:
            logger.error(f'Failed to execute tool {tool_name}: {e}')
            return f'Tool execution error: {str(e)}'

    def process_streaming_response(
            self, response) -> tuple[str, List[Dict[str, Any]]]:
        """Process streaming response and extract tool calls."""
        assistant_message: Dict[str, Any] = {
            'role': 'assistant',
            'content': ''
        }
        tool_calls = []
        done_thinking = False

        for chunk in response:
            # Handle reasoning process
            reasoning_content = getattr(chunk.choices[0].delta,
                                        'reasoning_content', None)
            if reasoning_content:
                if not done_thinking:
                    print('\n=== Reasoning Process ===')
                    done_thinking = True
                print(reasoning_content, end='', flush=True)

            # Handle response content
            if chunk.choices[0].delta.content:
                if not done_thinking:
                    print('\n\n=== Response Content ===\n')
                    done_thinking = True
                assistant_message['content'] += chunk.choices[0].delta.content
                print(chunk.choices[0].delta.content, end='', flush=True)

            # Handle tool calls
            if chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if tool_call.index is not None:
                        while len(tool_calls) <= tool_call.index:
                            tool_calls.append({
                                'id': '',
                                'type': 'function',
                                'function': {
                                    'name': '',
                                    'arguments': ''
                                }
                            })

                        if tool_call.id:
                            tool_calls[tool_call.index]['id'] = tool_call.id
                        if tool_call.function.name:
                            tool_calls[tool_call.index]['function'][
                                'name'] = tool_call.function.name
                        if tool_call.function.arguments:
                            tool_calls[tool_call.index]['function'][
                                'arguments'] += tool_call.function.arguments

        if tool_calls:
            assistant_message['tool_calls'] = tool_calls

        return assistant_message['content'], tool_calls

    def run_multi_turn_conversation(self, user_message: str) -> str:
        """Run multi-turn conversation with tool calling."""
        if not self.manager or not self.client:
            raise RuntimeError('MCPManager or OpenAI client not initialized')

        logger.info(f'Starting multi-turn conversation with: {user_message}')

        # Build system message
        system_message = self.build_system_message()
        openai_tools = self.manager.get_openai_tools()  # type: ignore

        # Initialize conversation history
        messages = [{
            'role': 'system',
            'content': system_message
        }, {
            'role': 'user',
            'content': user_message
        }]

        # Multi-turn conversation loop
        round_count = 0
        max_rounds = 10
        final_response = ''

        while round_count < max_rounds:
            logger.info(f'=== Round {round_count + 1} ===')

            # Call model
            response = self.client.chat.completions.create(  # type: ignore
                model='Qwen/Qwen3-32B',
                messages=messages,  # type: ignore
                tools=openai_tools if openai_tools else None,  # type: ignore
                stream=True,
                extra_body={
                    'enable_thinking': True,
                    'thinking_budget': 4096
                })

            # Process streaming response
            content, tool_calls = self.process_streaming_response(response)

            # Build assistant message
            assistant_message: Dict[str, Any] = {
                'role': 'assistant',
                'content': content
            }
            if tool_calls:
                assistant_message['tool_calls'] = tool_calls

            # Add assistant message to conversation history
            messages.append(assistant_message)
            final_response = content

            # Check if there are tool calls
            if not tool_calls:
                logger.info(
                    f'Conversation completed naturally in {round_count + 1} rounds'
                )
                break

            # Check if max rounds reached, force final response
            if round_count >= max_rounds - 1:
                logger.info(
                    f'Reached max rounds {max_rounds}, forcing final response')
                print('Based on tool results, providing complete answer...')

                # Build final message with all tool call history
                final_messages = [{
                    'role':
                    'system',
                    'content':
                    ('You are a helpful assistant. Please provide complete, detailed answers and suggestions '
                     'based on all previous tool call results. Ensure the answer covers all user questions.'
                     )
                }, {
                    'role': 'user',
                    'content': user_message
                }] + messages[
                    2:]  # Include all assistant responses and tool call results

                try:
                    logger.info('Generating final response...')
                    final_response_obj = self.client.chat.completions.create(  # type: ignore
                        model='Qwen/Qwen3-32B',
                        messages=final_messages,  # type: ignore
                        stream=False,
                        extra_body={'enable_thinking': False})
                    final_response = final_response_obj.choices[
                        0].message.content
                    print('\n=== Final Response ===')
                    print(final_response)
                    logger.info(
                        f'Conversation ended after {round_count + 1} rounds')
                    break
                except Exception as e:
                    logger.error(f'Failed to generate final response: {e}')
                    print('Using current round response as final answer')
                    break

            # Process tool calls
            logger.info(f'=== Tool Calls (Round {round_count + 1}) ===')
            tool_results = []

            for tool_call in tool_calls:
                try:
                    # Parse arguments
                    arguments = json.loads(tool_call['function']['arguments'])

                    # Call MCP tool
                    tool_name = tool_call['function']['name']
                    logger.info(f'Calling tool: {tool_name}')
                    logger.info(
                        f"Arguments: {tool_call['function']['arguments']}")

                    result = self.execute_tool_with_timeout(
                        tool_name, arguments)

                    tool_results.append({
                        'role': 'tool',
                        'tool_call_id': tool_call['id'],
                        'content': result
                    })

                    logger.info(f'Tool call result: {result[:200]}...')

                except json.JSONDecodeError as e:
                    error_result = f'Parameter parsing failed: {str(e)}'
                    tool_results.append({
                        'role': 'tool',
                        'tool_call_id': tool_call['id'],
                        'content': error_result
                    })
                    logger.error('Parameter parsing error: {error_result}')
                except Exception as e:
                    error_result = f'Tool call failed: {str(e)}'
                    tool_results.append({
                        'role': 'tool',
                        'tool_call_id': tool_call['id'],
                        'content': error_result
                    })
                    logger.error(f'Tool call error: {error_result}')

            # Add tool call results to conversation history
            messages.extend(tool_results)

            # Increment round count
            round_count += 1

        return final_response

    def show_statistics(self) -> None:
        """Show tool usage statistics."""
        if not self.manager:
            raise RuntimeError('MCPManager not initialized')

        logger.info('=== Tool Usage Statistics ===')
        stats = self.manager.get_tool_statistics()  # type: ignore
        logger.info(f"Total tools: {stats['total_tools']}")
        logger.info(f"Server status: {stats['servers']}")

        # Show tool summary
        logger.info('=== Tool Summary ===')
        tools_summary = self.manager.get_tools_summary()  # type: ignore
        logger.info(f"Connection mode: {tools_summary['connection_mode']}")
        logger.info('By server:')
        for server_name, info in tools_summary['servers'].items():
            logger.info(f"  {server_name}: {info['tool_count']} tools")

        # Show server status
        logger.info('=== Server Status ===')
        servers = self.manager.list_available_servers()  # type: ignore
        for server_name in servers:
            status = self.manager.get_server_status(
                server_name)  # type: ignore
            if status:
                logger.info(f"  {server_name}: {status['tool_count']} tools")

    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info('Cleaning up resources...')
        if self.manager:
            self.manager.shutdown()
        logger.info('MCPManager closed')

    def run_example(self) -> None:
        """Run the complete example."""
        try:
            # Setup
            self.setup_mcp_manager()
            self.setup_openai_client()

            # Get available tools
            self.get_available_tools()

            # User query
            user_message = ('查看杭州下周一的天气，以及那天从北京去杭州的高铁票')
            logger.info(f'User query: {user_message}')

            # Run conversation
            final_response = self.run_multi_turn_conversation(user_message)

            # Show statistics
            self.show_statistics()

            # Display final result
            print('\n=== Final Answer ===')
            print(final_response)

        except Exception as e:
            logger.error(f'Error: {e}')
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            self.cleanup()


def main():
    """Main function to run the MCP API conversation example."""
    example = MCPAPIConversationExample()
    example.run_example()


if __name__ == '__main__':
    main()
