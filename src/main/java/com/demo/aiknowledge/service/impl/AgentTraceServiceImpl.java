package com.demo.aiknowledge.service.impl;

import com.demo.aiknowledge.common.AgentTraceContext;
import com.demo.aiknowledge.entity.RequestTrace;
import com.demo.aiknowledge.mapper.RequestTraceMapper;
import com.demo.aiknowledge.service.AgentTraceService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 全链路追踪服务实现。
 * 内存中用 Map 收集事件，请求结束时序列化为 trace_json 一条 INSERT 落库（一个 trace 占一行）。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class AgentTraceServiceImpl implements AgentTraceService {

    private final RequestTraceMapper requestTraceMapper;
    private final ObjectMapper objectMapper;

    private static final int MAX_SNAPSHOT_LENGTH = 100000; // 不截断
    private static final DateTimeFormatter ISO_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS");

    // ── 生命周期 ─────────────────────────────────────

    @Override
    public TraceScope openTrace(String traceId, String sessionId, Long userId) {
        AgentTraceContext ctx = new AgentTraceContext(traceId, sessionId, userId);
        AgentTraceContext.set(ctx);
        return new TraceScopeImpl(ctx);
    }

    // ── 事件记录 ─────────────────────────────────────

    @Override
    public void recordEvent(String eventType, String phase, Object input, Object output) {
        recordEvent(eventType, phase, null, null, input, output, null, null, null, null, null);
    }

    /** 带耗时的 recordEvent（ChatServiceImpl 用） */
    public void recordEvent(String eventType, String phase, Object input, Object output, Long latencyMs) {
        recordEvent(eventType, phase, null, null, input, output, latencyMs, null, null, null, null);
    }

    @Override
    public void recordAgentCall(String eventType, String agentName, String modelName,
                                Object input, Object output, Long latencyMs,
                                Integer inputTokens, Integer outputTokens, Integer totalTokens) {
        recordEvent(eventType, "AI_CALL", agentName, modelName, input, output, latencyMs,
                inputTokens, outputTokens, totalTokens, null);
    }

    @Override
    public void recordEvent(String eventType, String phase, String agentName, String modelName,
                            Object input, Object output, Long latencyMs,
                            Integer inputTokens, Integer outputTokens, Integer totalTokens,
                            Map<String, Object> metadata) {
        AgentTraceContext ctx = AgentTraceContext.current();
        if (ctx == null) {
            log.warn("No AgentTraceContext found, skipping event: {}", eventType);
            return;
        }

        Map<String, Object> event = new LinkedHashMap<>();
        event.put("eventType", eventType);
        event.put("phase", phase);
        event.put("source", "java");
        if (agentName != null) event.put("agentName", agentName);
        if (modelName != null) event.put("modelName", modelName);
        if (input != null) event.put("inputData", truncate(input));
        if (output != null) event.put("outputData", truncate(output));
        if (latencyMs != null) event.put("latencyMs", latencyMs);
        if (inputTokens != null) event.put("inputTokens", inputTokens);
        if (outputTokens != null) event.put("outputTokens", outputTokens);
        if (totalTokens != null) event.put("totalTokens", totalTokens);
        if (metadata != null && !metadata.isEmpty()) event.put("metadata", metadata);
        event.put("eventTime", LocalDateTime.now().format(ISO_FMT));

        ctx.addEvent(event);
    }

    // ── 合并 Python traces ───────────────────────────

    @Override
    public void mergePythonTraces(List<Map<String, Object>> pythonTraces) {
        AgentTraceContext ctx = AgentTraceContext.current();
        if (ctx == null || pythonTraces == null || pythonTraces.isEmpty()) {
            return;
        }
        for (Map<String, Object> pyEvent : pythonTraces) {
            // Python 使用 snake_case，统一转为 camelCase
            Map<String, Object> normalized = new LinkedHashMap<>();
            pyEvent.forEach((k, v) -> normalized.put(snakeToCamel(k), v));
            ctx.addEvent(normalized);
        }
        log.debug("Merged {} Python trace events into trace {}", pythonTraces.size(), ctx.traceId());
    }

    private static String snakeToCamel(String key) {
        StringBuilder sb = new StringBuilder();
        boolean upper = false;
        for (int i = 0; i < key.length(); i++) {
            char c = key.charAt(i);
            if (c == '_') {
                upper = true;
            } else if (upper) {
                sb.append(Character.toUpperCase(c));
                upper = false;
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    // ── 标记失败 ─────────────────────────────────────

    @Override
    public void markFailed() {
        AgentTraceContext ctx = AgentTraceContext.current();
        if (ctx != null) {
            ctx.markFailed();
        }
    }

    // ── 落库 ─────────────────────────────────────────

    @Override
    public void flush() {
        AgentTraceContext ctx = AgentTraceContext.current();
        if (ctx == null || ctx.events().isEmpty()) {
            return;
        }
        try {
            long durationMs = (System.nanoTime() - ctx.startedAt()) / 1_000_000;

            // 取第一个事件的时间作为请求开始时间（兼容 chat 的 REQUEST_RECEIVED 和 parse 的 PARSE_START）
            String startTime = ctx.events().isEmpty() ? null
                    : (String) ctx.events().get(0).get("eventTime");

            // 构建 trace_json
            Map<String, Object> traceJson = new LinkedHashMap<>();
            traceJson.put("traceId", ctx.traceId());
            traceJson.put("sessionId", ctx.sessionId());
            traceJson.put("userId", ctx.userId());
            traceJson.put("status", ctx.isFailed() ? "FAILED" : "SUCCESS");
            traceJson.put("durationMs", durationMs);
            traceJson.put("startTime", startTime);
            traceJson.put("events", ctx.events());

            RequestTrace row = new RequestTrace();
            row.setTraceId(ctx.traceId());
            row.setSessionId(ctx.sessionId());
            row.setUserId(ctx.userId());
            row.setStatus(ctx.isFailed() ? "FAILED" : "SUCCESS");
            row.setEventCount(ctx.events().size());
            row.setDurationMs(durationMs);
            row.setTraceJson(objectMapper.writeValueAsString(traceJson));
            row.setCreatedAt(LocalDateTime.now());

            requestTraceMapper.insert(row);
            log.debug("Flushed trace {} with {} events ({}ms)", ctx.traceId(), ctx.events().size(), durationMs);
        } catch (Exception e) {
            log.error("Failed to flush trace {}", ctx.traceId(), e);
        }
    }

    @Override
    public String currentTraceId() {
        AgentTraceContext ctx = AgentTraceContext.current();
        return ctx != null ? ctx.traceId() : null;
    }

    // ── TraceScope 实现 ──────────────────────────────

    private class TraceScopeImpl implements TraceScope {
        private final AgentTraceContext ctx;

        TraceScopeImpl(AgentTraceContext ctx) {
            this.ctx = ctx;
        }

        @Override
        public void close() {
            try {
                flush();
            } finally {
                AgentTraceContext.remove();
            }
        }
    }

    // ── 内部工具 ─────────────────────────────────────

    @SuppressWarnings("unchecked")
    private Object truncate(Object obj) {
        if (obj == null) return null;
        if (obj instanceof Map) {
            Map<String, Object> mutable = new LinkedHashMap<>((Map<String, Object>) obj);
            mutable.replaceAll((k, v) -> truncateValue(v));
            return mutable;
        }
        if (obj instanceof String s && s.length() > MAX_SNAPSHOT_LENGTH) {
            return s.substring(0, MAX_SNAPSHOT_LENGTH) + "...[truncated]";
        }
        return obj;
    }

    private Object truncateValue(Object value) {
        if (value instanceof String s && s.length() > MAX_SNAPSHOT_LENGTH) {
            return s.substring(0, MAX_SNAPSHOT_LENGTH) + "...[truncated]";
        }
        return value;
    }
}
