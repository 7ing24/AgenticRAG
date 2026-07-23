package com.demo.aiknowledge.service;

import java.util.List;
import java.util.Map;

/**
 * 全链路追踪服务接口。
 * 提供 ThreadLocal 级 trace 上下文管理和事件记录，请求结束时批量落库。
 *
 * <pre>
 * try (TraceScope scope = agentTraceService.openTrace(traceId, sessionId, userId)) {
 *     agentTraceService.recordEvent("REQUEST_RECEIVED", "HTTP", input, null);
 *     // ... 业务逻辑 ...
 *     agentTraceService.mergePythonTraces(pythonTraces);
 * } // scope.close() 自动 flush + 清理 ThreadLocal
 * </pre>
 */
public interface AgentTraceService {

    /** 开启新的 trace 上下文，返回 AutoCloseable 的 TraceScope */
    TraceScope openTrace(String traceId, String sessionId, Long userId);

    /** 记录一条简单事件（无 token/latency） */
    void recordEvent(String eventType, String phase, Object input, Object output);

    /** 记录一条带耗时的事件 */
    void recordEvent(String eventType, String phase, Object input, Object output, Long latencyMs);

    /** 记录一条带 token 和耗时的 Agent 调用事件 */
    void recordAgentCall(String eventType, String agentName, String modelName,
                         Object input, Object output, Long latencyMs,
                         Integer inputTokens, Integer outputTokens, Integer totalTokens);

    /** 完整的 recordEvent，所有参数 */
    void recordEvent(String eventType, String phase, String agentName, String modelName,
                     Object input, Object output, Long latencyMs,
                     Integer inputTokens, Integer outputTokens, Integer totalTokens,
                     Map<String, Object> metadata);

    /** 将 Python 返回的 trace dict 列表转为 RequestTrace 并合并到当前上下文 */
    void mergePythonTraces(List<Map<String, Object>> pythonTraces);

    /** 批量写入 request_trace 表 */
    void flush();

    /** 标记当前 trace 为失败状态 */
    void markFailed();

    /** 从当前 ThreadLocal 获取 traceId */
    String currentTraceId();

    /** AutoCloseable 的 trace 作用域，请求结束时自动 flush + 清理 */
    interface TraceScope extends AutoCloseable {
        @Override
        void close();
    }
}
