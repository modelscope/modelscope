# ModelScope MCP Framework

[English](#english-version) | [中文版](#chinese-version)

---

## English Version

### Overview

The ModelScope MCP (Model Context Protocol) Framework enables seamless integration of AI models from ModelScope Hub with external tools and services. It supports powerful multi-turn conversations with tool calling capabilities across both local and remote language models.

### Key Features

- **ModelScope Integration**: Connects to MCP services deployed on ModelScope platform with simple authentication
- **Developer-Friendly**: High-level `MCPManager` abstraction hides underlying complexity
- **Multi-Transport**: Supports stdio and Server-Sent Events (SSE) connections
- **Hybrid LLM Support**: Works with both local models and API-based services
- **Tool Orchestration**: Automatic discovery, registration, and routing of tools
- **Performance Optimization**: Pre-warmed connections and resource management
- **Comprehensive Monitoring**: Built-in health checks and usage statistics

### Architecture

![MCP Architecture Diagram](./Modelscope%20MCP%20Architecture.png)

### Core Components

#### 1. MCPManager
The primary interface providing:
- Tool lifecycle management
- Service registry operations
- OpenAI-compatible tool format conversion
- Multi-turn conversation orchestration

#### 2. MCPClient
Handles low-level transport with:
- Connection management (stdio/SSE)
- Error recovery and reconnection
- Resource cleanup

### Quick Start

#### Installation
```bash
pip install modelscope openai torch transformers mcp
export MODELSCOPE_SDK_TOKEN="your_token_here"
```

#### Basic Usage
```python
from modelscope.hub.modelscope_mcp.manager import MCPManager

# Initialize with pre-warmed connections
manager = MCPManager(warmup_connect=True)

# Get available tools
tools = manager.get_tools()

# Integrate with your LLM
response = llm.generate(prompt, tools=tools)
```

### Configuration

#### Example Configuration (mcp_config.json)
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": { "DEBUG": "1" }
    }
  }
}
```

### Advanced Usage

#### Custom Integration
```python
from modelscope.hub.modelscope_mcp.client import MCPClient
from modelscope.hub.modelscope_mcp.manager import MCPManager

class CustomMCPHandler:
    def __init__(self):
        self.manager = MCPManager()
        # Add custom logic here
```

### API Reference

#### MCPManager Methods
| Method                     | Description                          |
|----------------------------|--------------------------------------|
| `get_tools()`              | List all available tools             |
| `get_openai_tools()`       | Get OpenAI-compatible tool definitions |
| `get_tool_by_name(name)`   | Retrieve specific tool by name       |
| `query_service_registry(keywords)` | Search services by keywords |
| `shutdown()`               | Clean up resources                   |

### Developer Tips

1. **ModelScope Integration**: Log in to ModelScope to use pre-configured MCP servers
2. **Performance**: Limit concurrent MCP servers to avoid slowdowns
3. **Resource Management**: Always call `shutdown()` after use
4. **Debugging**: Enable debug logging with:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Troubleshooting

| Issue                | Solution                             |
|----------------------|--------------------------------------|
| Import errors        | Verify dependencies are installed    |
| Connection failures  | Check network and server status      |
| Tool not found       | Validate tool names and registry     |
| Timeout errors       | Increase timeout values              |
| Performance issues   | Reduce active MCP servers            |

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a Pull Request

### License

This project is licensed under the MIT License.


---

## 中文版

### 概述

ModelScope MCP（模型上下文协议）框架是一个用于将ModelScope Hub的AI模型与外部工具和服务集成的综合工具包。它支持本地和远程语言模型与各种MCP兼容工具的无缝集成，实现强大的多轮对话和工具调用功能。

### 核心特性

- **ModelScope集成**：通过简单认证即可连接ModelScope平台上部署的MCP服务
- **开发友好**：高级`MCPManager`抽象隐藏底层复杂性
- **多传输支持**：支持stdio和服务器发送事件(SSE)连接
- **混合LLM支持**：同时支持本地模型和基于API的服务
- **工具编排**：自动发现、注册和路由工具
- **性能优化**：预连接和资源管理机制
- **全面监控**：内置健康检查和使用统计

### 架构设计

![MCP架构图](./Modelscope%20MCP%20Architecture.png)

### 核心组件

#### 1. MCPManager
提供以下主要功能：
- 工具生命周期管理
- 服务注册操作
- OpenAI兼容工具格式转换
- 多轮对话编排

#### 2. MCPClient
处理底层传输：
- 连接管理(stdio/SSE)
- 错误恢复和重连
- 资源清理

### 快速开始

#### 安装
```bash
pip install modelscope openai torch transformers mcp
export MODELSCOPE_SDK_TOKEN="your_token_here"
```

#### 基本用法
```python
from modelscope.hub.modelscope_mcp.manager import MCPManager

# 初始化并预建立连接
manager = MCPManager(warmup_connect=True)

# 获取可用工具
tools = manager.get_tools()

# 与您的大语言模型集成
response = llm.generate(prompt, tools=tools)
```

### 配置

#### 配置示例 (mcp_config.json)
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
      "env": { "DEBUG": "1" }
    }
  }
}
```

### 高级用法

#### 自定义集成
```python
from modelscope.hub.modelscope_mcp.client import MCPClient
from modelscope.hub.modelscope_mcp.manager import MCPManager

class CustomMCPHandler:
    def __init__(self):
        self.manager = MCPManager()
        # 在此添加自定义逻辑
```

### API参考

#### MCPManager方法
| 方法                     | 描述                          |
|--------------------------|-------------------------------|
| `get_tools()`            | 列出所有可用工具              |
| `get_openai_tools()`     | 获取OpenAI兼容的工具定义      |
| `get_tool_by_name(name)` | 按名称检索特定工具            |
| `query_service_registry(keywords)` | 按关键词搜索服务 |
| `shutdown()`             | 清理资源                      |

### 开发者提示

1. **ModelScope集成**：登录ModelScope以使用预配置的MCP服务器
2. **性能优化**：限制并发MCP服务器数量以避免性能下降
3. **资源管理**：使用完毕后始终调用`shutdown()`
4. **调试方法**：通过以下代码启用调试日志
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### 故障排除

| 问题                | 解决方案                          |
|---------------------|-----------------------------------|
| 导入错误            | 验证依赖是否安装                  |
| 连接失败            | 检查网络和服务器状态              |
| 工具未找到          | 验证工具名称和注册表              |
| 超时错误            | 增加超时值                        |
| 性能问题            | 减少活跃的MCP服务器数量           |

### 贡献指南

1. Fork仓库
2. 创建功能分支
3. 提交您的更改
4. 提交Pull Request

### 许可证

本项目采用MIT许可证。


### 优化说明

1. **结构优化**：
   - 将安装和基本用法提前至"Quick Start"部分，符合开发者使用习惯
   - 合并重复的示例和参考章节，减少冗余

2. **表述优化**：
   - 使用更简洁的技术术语表述（如"流式支持"改为"Server-Sent Events"）
   - 统一中英文术语对应关系（如"服务注册表"对应"Service Registry"）

3. **可读性提升**：
   - 为API参考添加表格形式，提高信息密度
   - 优化故障排除部分的呈现方式
   - 增强内部导航链接

4. **双语一致性**：
   - 确保中英文版本结构完全对应
   - 简化双语切换机制，使用统一的标题格式

这些优化使文档更加简洁、一致，同时保持了完整的技术信息，有助于开发者更快速地理解和使用ModelScope MCP框架。
