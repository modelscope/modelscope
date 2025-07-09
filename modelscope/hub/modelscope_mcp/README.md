# ModelScope MCP Framework

[English](#english) | [中文](#chinese)

---

## ⚠️ Important Tips

### 🔑 Key Features
- **Integrated ModelScope MCP Services**: This framework integrates with MCP services deployed on ModelScope. Users only need to log in to use the MCP servers configured on the ModelScope platform.
- **Simplified Development**: Developers only need to use the `MCPManager` - all underlying client operations are completely hidden within the manager.
- **Performance Optimization**: It's not recommended to enable too many MCP servers simultaneously as it may slow down output. ModelScope will launch personal MCP server management features in the future.
- **Warmup & Summary Features**: Built-in warmup connection mode and server summary functionality for better performance.

### 🚀 Development Options
- **Direct Usage**: Use `MCPManager` directly for quick integration
- **Framework Development**: Build custom solutions based on the ModelScope MCP framework
- **Reference Examples**: See provided examples for both approaches

### 💡 Developer Tips
1. **ModelScope Integration**: This framework integrates with MCP services deployed on ModelScope. Simply log in to use the MCP servers configured on the ModelScope official platform.
2. **Manager-First Approach**: Developers only need to use the `MCPManager` - all underlying client operations are completely hidden within the manager for simplified development.
3. **Server Performance**: Avoid enabling too many MCP servers simultaneously as it may slow down output. ModelScope will launch personal MCP server management features in the future, stay tuned.
4. **Advanced Features**:
   - **Warmup Mode**: Manager initialization defaults to `warmup_connect=True` for pre-established connections and faster tool calls
   - **Summary Server**: Built-in server status and tool statistics functionality for better monitoring
5. **Resource Management**: Always call `manager.shutdown()` when done to properly clean up resources
6. **Error Handling**: The framework provides comprehensive error handling - check logs for detailed error information
7. **Tool Discovery**: Use `manager.get_tools()` to discover available tools and `manager.get_tools_summary()` for overview
8. **Service Registry**: Use `manager.query_service_registry(keywords)` to find tools by keywords

---

## English

### Overview

The ModelScope MCP (Model Context Protocol) Framework is a comprehensive toolkit for integrating ModelScope Hub's AI models with external tools and services through the Model Context Protocol. This framework provides seamless integration between local/remote language models and various MCP-compatible tools, enabling powerful multi-turn conversations with tool calling capabilities.

### Key Features

- **Multi-Transport Support**: Supports stdio and SSE connections
- **Local & Remote LLM Integration**: Works with both local models and API calls
- **Advanced Tool Management**: Automatic tool discovery, registration, and routing, no manual manipulation required
- **Warmup Connection**: Pre-establish connections during manager initialization to speed up tool calls
- **Server Summary**: Automatic server status and tool statistics
- **Service Registry**: Dynamic service discovery and metadata management
- **Streaming Support**: Real-time streaming responses with tool calling
- **Robust Error Handling**: Comprehensive error recovery and reconnection mechanisms
- **Resource Management**: Automatic cleanup and resource isolation
- **Health Monitoring**: Built-in health checks and connection monitoring

### Architecture

![MCP Architecture](./Modelscope%20MCP%20Architecture.png)

### Core Components

#### 1. MCPManager (Main Interface)
The primary interface for developers, providing high-level encapsulation. Developers only need to use the manager to complete all MCP deployments:
- Tool discovery and management
- Service registry operations
- OpenAI-compatible tool format conversion
- Multi-turn conversation orchestration
- All underlying client operations are completely hidden

#### 2. MCPClient (Transport Layer)
Low-level transport management, operations completely hidden, supporting:
- Multiple connection types (stdio, SSE)
- Connection lifecycle management
- Error handling and recovery
- Resource cleanup



### Development Guide

#### Option 1: Direct MCPManager Usage
For quick integration, use MCPManager directly:
```python
from modelscope.hub.modelscope_mcp.manager import MCPManager

manager = MCPManager(warmup_connect=True)
tools = manager.get_tools()
# Use tools with your LLM
```

#### Option 2: Framework Development
For custom solutions, build on the ModelScope MCP framework:
```python
from modelscope.hub.modelscope_mcp.client import MCPClient
from modelscope.hub.modelscope_mcp.manager import MCPManager

# Custom implementation using ModelScope MCP framework components
class CustomMCPHandler:
    def __init__(self):
        self.manager = MCPManager()
        # Custom logic here
```

#### Reference Examples
- `examples/mcp_api.py`: Complete OpenAI API integration
- `examples/mcp_local.py`: Complete local LLM integration

### Quick Start

#### Installation

```bash
# Install required dependencies
pip install modelscope openai torch transformers mcp

# Set ModelScope token
export MODELSCOPE_SDK_TOKEN="your_token_here"
```

#### Basic Usage

For complete usage examples, please refer to:
- `examples/mcp_api.py`: OpenAI API integration with tool calling
- `examples/mcp_local.py`: Local LLM integration with tool calling

### Examples

For complete working examples, see:
- `examples/mcp_api.py`: Complete OpenAI API integration
- `examples/mcp_local.py`: Complete local LLM integration

### Configuration

#### MCP Configuration File Example (mcp_config.json)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {
        "DEBUG": "1"
      }
    }
  }
}
```


### API Reference

#### MCPManager Methods

- `get_tools()`: Get all available tools
- `get_openai_tools()`: Get tools in OpenAI format for API usage with MCP
- `get_tool_by_name(name)`: Get specific tool
- `query_service_registry(keywords)`: Query service registry
- `get_service_metadata(service_id)`: Get service metadata
- `get_service_brief_summary()`: Generate MCP service brief information
- `get_service_brief_for_prompt()`: Get service brief for prompts
- `get_tool_statistics()`: Get usage statistics
- `shutdown()`: Cleanup resources

### Troubleshooting

#### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Connection Failures**: Check network connectivity and server status
3. **Tool Not Found**: Verify tool names and service registry
4. **Timeout Errors**: Increase timeout values for slow operations
5. **Performance Issues**: Reduce the number of active MCP servers

#### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Contributing

Welcome contributions! Please follow these steps:

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Create a Pull Request

### License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Chinese

### 概述

ModelScope MCP（Model Context Protocol）框架是一个综合工具包，用于通过模型上下文协议将ModelScope Hub的AI模型与外部工具和服务集成。该框架提供了本地/远程语言模型与各种MCP兼容工具之间的无缝集成，支持强大的多轮对话和工具调用功能。

### 核心特性

- **多传输支持**：支持stdio和SSE连接
- **本地和远程LLM集成**：支持本地模型和API调用
- **高级工具管理**：自动工具发现、注册和路由，无需手动操控
- **预热连接**：初始化manager时预建立连接以加快工具调用
- **服务器摘要**：自动服务器状态和工具统计
- **服务注册表**：动态服务发现和元数据管理
- **流式支持**：实时流式响应和工具调用
- **健壮错误处理**：全面的错误恢复和重连机制
- **资源管理**：自动清理和资源隔离
- **健康监控**：内置健康检查和连接监控

### 架构设计

![Modelscope MCP 架构图](./Modelscope%20MCP%20Architecture.png)

### 核心组件

#### 1. MCPManager（主要接口）
开发者的主要接口，提供高级封装，开发者仅需通过manager即可完成全部mcp部署：
- 工具发现和管理
- 服务注册表操作
- OpenAI兼容工具格式转换
- 多轮对话编排
- 所有底层客户端操作完全隐藏

#### 2. MCPClient（传输层）
底层传输管理，操作完全隐藏，支持：
- 多种连接类型（stdio、SSE）
- 连接生命周期管理
- 错误处理和恢复
- 资源清理



### 开发指南

#### 选项1：直接使用MCPManager
快速集成，直接使用MCPManager：
```python
from modelscope.hub.modelscope_mcp.manager import MCPManager

manager = MCPManager(warmup_connect=True)
tools = manager.get_tools()
# 与您的LLM一起使用工具
```

#### 选项2：基于框架开发
自定义解决方案，基于ModelScope MCP框架构建：
```python
from modelscope.hub.modelscope_mcp.client import MCPClient
from modelscope.hub.modelscope_mcp.manager import MCPManager

# 使用ModelScope MCP框架组件的自定义实现
class CustomMCPHandler:
    def __init__(self):
        self.manager = MCPManager()
        # 自定义逻辑
```

#### 参考示例
- `examples/mcp_api.py`：完整的OpenAI API集成
- `examples/mcp_local.py`：完整的本地LLM集成

### 💡 开发者tips
1. **ModelScope集成**：本框架集成了用户在ModelScope上部署的MCP服务，只需要登录即可使用ModelScope官网端配置的MCP服务器
2. **Manager优先**：开发者只需要使用MCPManager即可，底层的client完全隐藏在manager的使用中
3. **服务器性能**：使用中不建议同时启用太多个MCP server，会导致输出变慢。后续ModelScope会上线个人MCP server管理功能，敬请期待
4. **高级功能**：
   - **预热功能**：manager初始化默认使用warmup_connect=True进行预建立连接，加快工具调用速度
   - **摘要服务器**：内置服务器状态和工具统计功能，便于监控和管理
5. **资源管理**：完成后务必调用manager.shutdown()以正确清理资源
6. **错误处理**：框架提供全面的错误处理机制，查看日志获取详细错误信息
7. **工具发现**：使用manager.get_tools()发现可用工具，使用manager.get_tools_summary()获取概览
8. **服务注册表**：使用manager.query_service_registry(keywords)按关键词查找工具

### 快速开始

#### 安装

```bash
# 安装依赖
pip install modelscope openai torch transformers mcp

# 设置ModelScope令牌
export MODELSCOPE_SDK_TOKEN="your_token_here"
```

#### 基本用法

完整的使用示例请参考：
- `examples/mcp_api.py`：OpenAI API集成与工具调用
- `examples/mcp_local.py`：本地LLM集成与工具调用

### 配置

#### MCP配置文件示例 (mcp_config.json)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": {
        "DEBUG": "1"
      }
    }
  }
}
```


### API参考

#### MCPManager方法

- `get_tools()`: 获取所有可用工具
- `get_openai_tools()`: 获取OpenAI格式的工具，用于API使用MCP
- `get_tool_by_name(name)`: 获取特定工具
- `query_service_registry(keywords)`: 查询服务注册表
- `get_service_metadata(service_id)`: 获取服务元数据
- `get_service_brief_summary()`: 生成MCP服务简要信息
- `get_service_brief_for_prompt()`: 获取提示词用的服务简介
- `get_tool_statistics()`: 获取使用统计
- `shutdown()`: 清理资源

### 故障排除

#### 常见问题

1. **导入错误**：确保所有依赖都已安装
2. **连接失败**：检查网络连接和服务器状态
3. **工具未找到**：验证工具名称和服务注册表
4. **超时错误**：为慢速操作增加超时值
5. **性能问题**：减少活跃MCP服务器的数量

#### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

### 许可证

本项目采用MIT许可证。详见LICENSE文件。
