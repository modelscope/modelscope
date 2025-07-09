# MCP API 测试说明

## 概述

`test_hub_mcp_api.py` 是 ModelScope MCP (Model Context Protocol) API 的测试文件，用于验证 MCP 相关功能的正确性。

## 运行测试前的准备

### 1. 获取访问令牌

1. 访问 [魔搭社区](https://modelscope.cn/)
2. 登录你的账号
3. 进入"访问令牌"页面
4. 创建一个新的访问令牌

### 2. 设置环境变量

在运行测试之前，需要设置环境变量：

```bash
# Linux/macOS
export TEST_ACCESS_TOKEN_CITEST="你的访问令牌"

# Windows (CMD)
set TEST_ACCESS_TOKEN_CITEST=你的访问令牌

# Windows (PowerShell)
$env:TEST_ACCESS_TOKEN_CITEST="你的访问令牌"
```

## 运行测试

### 运行所有 MCP 测试

```bash
python -m pytest modelscope/tests/hub/test_hub_mcp_api.py -v
```

### 运行特定测试

```bash
# 运行获取服务器列表的测试
python -m pytest modelscope/tests/hub/test_hub_mcp_api.py::TestMcpApi::test_get_mcp_servers -v

# 运行获取操作 URL 的测试
python -m pytest modelscope/tests/hub/test_hub_mcp_api.py::TestMcpApi::test_get_mcp_server_operational_urls -v
```

### 一行命令设置环境变量并运行

```bash
# Linux/macOS
TEST_ACCESS_TOKEN_CITEST="你的访问令牌" python -m pytest modelscope/tests/hub/test_hub_mcp_api.py -v

# Windows (PowerShell)
$env:TEST_ACCESS_TOKEN_CITEST="你的访问令牌"; python -m pytest modelscope/tests/hub/test_hub_mcp_api.py -v
```

## 测试覆盖范围

### 当前启用的测试

- ✅ `test_get_mcp_servers()` - 获取 MCP 服务器列表
- ✅ `test_get_mcp_servers_with_filter()` - 带过滤的服务器列表
- ✅ `test_get_mcp_server_operational()` - 获取用户托管服务器
- ✅ `test_get_mcp_server_operational_urls()` - 获取操作 URL
- ✅ `test_get_mcp_server_special()` - 获取特定服务器详情
- ✅ `test_get_mcp_server_special_with_operational_url()` - 带 URL 的服务器详情
- ✅ `test_get_mcp_servers_without_token()` - 无 token 获取服务器列表
- ✅ `test_get_mcp_server_operational_without_token()` - 无 token 获取托管服务器
- ✅ `test_get_mcp_server_special_without_token()` - 无 token 获取特定服务器
- ✅ `test_operational_urls_structure()` - 验证 URL 结构
- ✅ `test_server_brief_list_structure()` - 验证服务器列表结构
- ✅ `test_pagination_parameters()` - 分页参数测试
- ✅ `test_search_functionality()` - 搜索功能测试
- ✅ `test_endpoint_override()` - 端点覆盖测试

### 暂时注释的测试（部署相关）

以下测试暂时被注释，等待部署功能实现后启用：

- 🔄 `test_deploy_mcp_server()` - 测试部署 MCP 服务器
- 🔄 `test_deploy_mcp_server_with_auth_check()` - 测试带认证检查的部署
- 🔄 `test_undeploy_mcp_server()` - 测试取消部署 MCP 服务器
- 🔄 `test_deploy_mcp_server_without_token()` - 测试无 token 部署的错误处理
- 🔄 `test_undeploy_mcp_server_without_token()` - 测试无 token 取消部署的错误处理

## 测试级别控制

测试使用 `@unittest.skipUnless(test_level() >= 0, 'skip test in current test level')` 装饰器来控制执行级别。

可以通过设置 `TEST_LEVEL` 环境变量来控制测试执行：

```bash
export TEST_LEVEL=0  # 运行所有测试
export TEST_LEVEL=1  # 运行级别 >= 1 的测试
export TEST_LEVEL=2  # 运行级别 >= 2 的测试
```

## 注意事项

1. **安全性**: 不要在代码中硬编码访问令牌，始终使用环境变量
2. **权限**: 确保你的访问令牌有足够的权限来访问 MCP 相关的 API
3. **网络**: 测试会调用真实的 API，需要网络连接
4. **频率**: 避免频繁运行测试，以免触发 API 频率限制
5. **环境**: 建议在测试环境中运行，避免影响生产环境

## 故障排除

### 常见错误

1. **Token 无效**: 检查环境变量是否正确设置
2. **网络错误**: 检查网络连接和防火墙设置
3. **权限不足**: 确认访问令牌有足够的权限
4. **频率限制**: 等待一段时间后重试

### 调试模式

启用详细输出：

```bash
python -m pytest modelscope/tests/hub/test_hub_mcp_api.py -v -s
```

## 贡献指南

1. 添加新测试时，请遵循现有的命名和结构规范
2. 确保测试覆盖正常情况和异常情况
3. 使用 `@unittest.skipUnless` 控制测试级别
4. 不要硬编码敏感信息
5. 添加适当的文档字符串
