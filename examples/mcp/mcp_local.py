#!/usr/bin/env python3
"""
Example usage of MCPManager with local LLM for multi-turn conversations.

This example demonstrates how to use MCPManager with local language models
for multi-turn conversations with tool calling capabilities. It shows
the integration between ModelScope MCP tools and local LLM inference.

Features:
- Local LLM integration with MCP tools
- Multi-turn conversation with tool calling
- Streaming response handling
- Tool execution with service registry
- Comprehensive error handling
- Optimized for 8B models for faster inference
"""

import concurrent.futures
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import torch

from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope.hub.modelscope_mcp.manager import MCPManager  # type: ignore

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'modelscope'))

# Set ModelScope token
os.environ['MODELSCOPE_SDK_TOKEN'] = 'Your Modelscope Token'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class LocalLLMManager:
    """Local LLM manager for model loading and inference."""

    def __init__(self, model_name: str = 'Qwen/Qwen3-8B'):
        self.model_name = model_name
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.is_loaded = False

    def load_model(self):
        """Load model and tokenizer."""
        logger.info(f'Loading model: {self.model_name}')

        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f'✅ GPU available: {torch.cuda.device_count()} cards')
            device = 'auto'
        else:
            logger.warning('⚠️ GPU not available, using CPU')
            device = 'cpu'

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model
        if device == 'auto':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
                # attn_implementation="flash_attention_2"  # Temporarily commented, requires flash_attn
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map='auto',
                trust_remote_code=True)

        self.is_loaded = True
        logger.info('✅ Model loading completed!')

    def generate_response(self,
                          messages: List[Dict[str, str]],
                          max_new_tokens: int = 2048,
                          enable_thinking: bool = True,
                          stream: bool = True) -> Tuple[str, str]:
        """Generate response with separated thinking and final content."""
        if not self.is_loaded or self.tokenizer is None or self.model is None:
            raise RuntimeError(
                'Model not loaded, please call load_model() first')

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking)

        if torch.cuda.is_available():
            model_inputs = self.tokenizer([text],
                                          return_tensors='pt').to('cuda:0')
        else:
            model_inputs = self.tokenizer([text], return_tensors='pt')

        start_time = time.time()

        with torch.no_grad():
            eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)

            if stream:
                from transformers.generation.streamers import TextStreamer

                if self.tokenizer is not None:
                    streamer = TextStreamer(
                        self.tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True)

                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=eos_token_id,
                        streamer=streamer)
                else:
                    generated_ids = self.model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=eos_token_id)
            else:
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=eos_token_id)

            # Robust check: if no new content generated, return empty strings
            if generated_ids.shape[1] <= model_inputs.input_ids.shape[1]:
                return '', ''

            output_ids = generated_ids[0][len(model_inputs.input_ids[0]
                                              ):].tolist()

        end_time = time.time()
        generation_time = end_time - start_time
        logger.info(f'⏱️ Generation time: {generation_time:.2f} seconds')

        # Parse thinking content
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(
            output_ids[:index], skip_special_tokens=True).strip('\n')
        content = self.tokenizer.decode(
            output_ids[index:], skip_special_tokens=True).strip('\n')

        return thinking_content, content


def build_system_prompt_with_tools(mcp_manager: Any) -> str:
    """Build system prompt with tool information."""

    # Get service brief information (includes usage rules)
    service_brief = mcp_manager.get_service_brief_for_prompt()  # type: ignore

    system_prompt = f"""You can use the following tools to help users.

{service_brief}

Please handle user requests efficiently and concisely."""

    return system_prompt


def extract_tool_calls_from_content(content: str) -> List[Dict[str, Any]]:
    """Extract tool calls from model response content."""
    tool_calls = []

    # Find all content within <function_call> tags
    import re
    function_call_pattern = r'<function_call>\s*(\{.*?\})\s*</function_call>'
    matches = re.findall(function_call_pattern, content, re.DOTALL)

    for i, match in enumerate(matches):
        try:
            tool_call_data = json.loads(match)

            # Check required fields
            if 'name' not in tool_call_data:
                continue

            tool_call = {
                'id': f'call_{i}',
                'type': 'function',
                'function': {
                    'name':
                    tool_call_data.get('name', ''),
                    'arguments':
                    json.dumps(
                        tool_call_data.get('parameters', {}),
                        ensure_ascii=False)
                }
            }
            tool_calls.append(tool_call)

        except json.JSONDecodeError:
            continue
        except Exception:
            continue

    return tool_calls


def execute_tool_calls(
        mcp_manager: Any, tool_calls: List[Dict[str,
                                                Any]]) -> List[Dict[str, Any]]:
    """Execute tool calls and return results."""
    tool_results = []

    for tool_call in tool_calls:
        try:
            arguments = json.loads(tool_call['function']['arguments'])
            tool_name = tool_call['function']['name']
            logger.info(f'Calling tool: {tool_name}')

            # Handle service registry tools
            if tool_name == 'query_service_registry':
                keywords = arguments.get('keywords', [])
                if isinstance(keywords, str):
                    keywords = [keywords]
                result = mcp_manager.query_service_registry(
                    keywords)  # type: ignore
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                logger.info(
                    f'Service registry query result: found {len(result)} services'
                )

            elif tool_name == 'get_service_metadata':
                service_id = arguments.get('service_id', '')
                result = mcp_manager.get_service_metadata(
                    service_id)  # type: ignore
                if result:
                    result_str = json.dumps(
                        result, ensure_ascii=False, indent=2)
                    logger.info(
                        f'Service metadata query successful: {service_id}')
                else:
                    result_str = f'Error: Service {service_id} not found'
                    logger.error(
                        f'Service metadata query failed: {service_id}')

            else:
                # Handle other MCP tools
                tool = mcp_manager.get_tool_by_name(tool_name)  # type: ignore

                if not tool:
                    result_str = f'Error: Tool {tool_name} not found'
                    logger.error(result_str)
                else:
                    # Execute tool call
                    try:
                        result = tool.call(arguments)
                        result_str = result
                        logger.info(f'Tool call successful: {result[:100]}...')

                    except Exception as e:
                        result_str = f'Tool call failed: {str(e)}'
                        logger.error('Tool call exception: {result_str}')

            tool_results.append({
                'role': 'tool',
                'tool_call_id': tool_call['id'],
                'content': result_str
            })

        except json.JSONDecodeError as e:
            error_result = f'Parameter parsing failed: {str(e)}'
            tool_results.append({
                'role': 'tool',
                'tool_call_id': tool_call['id'],
                'content': error_result
            })
        except Exception as e:
            error_result = f'Tool call failed: {str(e)}'
            tool_results.append({
                'role': 'tool',
                'tool_call_id': tool_call['id'],
                'content': error_result
            })

    return tool_results


class MCPLocalConversationExample:
    """Example class demonstrating MCPManager with local LLM integration."""

    def __init__(self):
        """Initialize the example."""
        self.mcp_manager: Optional[Any] = None
        self.llm_manager: Optional[LocalLLMManager] = None

    def setup_mcp_manager(self) -> None:
        """Set up MCPManager instance."""
        if MCPManager is None:
            raise RuntimeError(
                'MCPManager not available. Please ensure modelscope is properly installed.'
            )

        logger.info('Initializing MCPManager...')
        self.mcp_manager = MCPManager()
        logger.info('MCPManager initialized successfully')

    def setup_local_llm(self, model_name: str = 'Qwen/Qwen3-8B') -> None:
        """Set up local LLM manager."""
        logger.info(f'Setting up local LLM: {model_name}')
        self.llm_manager = LocalLLMManager(model_name)
        self.llm_manager.load_model()
        logger.info('Local LLM setup completed')

    def run_multi_turn_conversation(self, user_query: str) -> str:
        """Run multi-turn conversation with local LLM and MCP tools."""
        if not self.mcp_manager or not self.llm_manager:
            raise RuntimeError('MCPManager or LocalLLMManager not initialized')

        logger.info(f'Starting multi-turn conversation with: {user_query}')

        # Build system prompt
        system_prompt = build_system_prompt_with_tools(self.mcp_manager)

        # Initialize conversation history
        messages = [{
            'role': 'system',
            'content': system_prompt
        }, {
            'role': 'user',
            'content': user_query
        }]

        # Print current prompt for analysis
        print('\n' + '=' * 80)
        print('📝 Current System Prompt:')
        print('=' * 80)
        print(system_prompt)
        print('=' * 80)

        # Multi-turn conversation loop
        round_count = 0
        max_rounds = 10
        final_response = ''

        while round_count < max_rounds:
            logger.info(f'=== Round {round_count + 1} ===')

            # Print complete prompt sent to model
            print('\n' + '-' * 60)
            print('📤 Complete prompt sent to model:')
            print('-' * 60)
            for msg in messages:
                print(f"[{msg['role']}]: {msg['content'][:200]}"
                      f"{'...' if len(msg['content']) > 200 else ''}")
            print('-' * 60)

            # Generate response (optimized version)
            thinking_content, content = self.llm_manager.generate_response(
                messages,
                max_new_tokens=128,  # Reduced to 128 for speed
                enable_thinking=False,  # Disable thinking feature
                stream=True  # Keep streaming generation
            )

            # Display thinking process
            if thinking_content:
                print('Thinking process:', thinking_content)

            print('Response:', content)

            # Extract tool calls
            tool_calls = extract_tool_calls_from_content(content)

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
                    f'Conversation completed in {round_count + 1} rounds')
                break

            # Check if max rounds reached
            if round_count >= max_rounds - 1:
                logger.info(
                    f'Reached max rounds {max_rounds}, generating final response'
                )

                # Build final message
                final_messages = [{
                    'role':
                    'system',
                    'content':
                    ('You are a helpful assistant. Please provide complete, '
                     'detailed answers and suggestions based on all previous '
                     'tool call results.')
                }, {
                    'role': 'user',
                    'content': user_query
                }] + messages[2:]

                try:
                    final_thinking, final_response = self.llm_manager.generate_response(
                        final_messages,
                        max_new_tokens=1024,
                        enable_thinking=False)
                    print('Final response:', final_response)
                    break
                except Exception as e:
                    logger.error(f'Failed to generate final response: {e}')
                    break

            # Process tool calls
            tool_results = execute_tool_calls(self.mcp_manager, tool_calls)

            # Add tool call results to conversation history
            messages.extend(tool_results)

            # Increment round count
            round_count += 1

        return final_response

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.mcp_manager:
                self.mcp_manager.shutdown()  # type: ignore
                logger.info('MCPManager shutdown completed')
        except Exception as e:
            logger.error(f'Error during MCPManager shutdown: {e}')

    def run_example(self) -> None:
        """Run the complete example."""
        try:
            # Setup
            self.setup_mcp_manager()
            self.setup_local_llm('Qwen/Qwen3-8B')

            # User query
            user_query = ('看看杭州下周一的天气，以及那天从北京去杭州的高铁票')
            logger.info(f'User query: {user_query}')

            # Run conversation
            final_response = self.run_multi_turn_conversation(user_query)

            # Display final result
            print('\n=== Final Answer ===')
            print(final_response)

        except Exception as e:
            logger.error(f'Error during example execution: {e}')
            import traceback
            traceback.print_exc()

        finally:
            # Cleanup
            self.cleanup()
            logger.info('✅ Multi-turn conversation example completed!')


def main():
    """Main function to run the MCP local conversation example."""
    example = MCPLocalConversationExample()
    example.run_example()


if __name__ == '__main__':
    main()
