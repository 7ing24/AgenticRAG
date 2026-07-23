package com.demo.aiknowledge.common;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * 请求级别的 trace 上下文，通过 ThreadLocal 绑定当前线程。
 * 一次请求内所有事件以 Map 形式累积到 events 列表，
 * 由 AgentTraceServiceImpl.flush() 统一序列化为 trace_json 落库。
 */
public class AgentTraceContext {

    private static final ThreadLocal<AgentTraceContext> CURRENT = new ThreadLocal<>();

    /** 请求唯一标识 */
    private final String traceId;
    /** 会话 ID */
    private final String sessionId;
    /** 用户 ID */
    private final Long userId;
    /** 请求开始时间（纳秒） */
    private final long startedAt;
    /** 累积的 trace 事件（每个事件为 Map，key 对齐原 RequestTrace 字段名） */
    private final List<Map<String, Object>> events = new ArrayList<>();
    /** 线程安全的 stepOrder 生成器 */
    private final AtomicInteger stepCounter = new AtomicInteger(0);
    /** 是否标记为失败 */
    private boolean failed = false;

    public AgentTraceContext(String traceId, String sessionId, Long userId) {
        this.traceId = traceId;
        this.sessionId = sessionId;
        this.userId = userId;
        this.startedAt = System.nanoTime();
    }

    // ── ThreadLocal 存取 ──────────────────────────────

    public static AgentTraceContext current() {
        return CURRENT.get();
    }

    public static void set(AgentTraceContext ctx) {
        CURRENT.set(ctx);
    }

    public static void remove() {
        CURRENT.remove();
    }

    // ── 事件管理 ─────────────────────────────────────

    /** 添加一个事件 Map，自动分配 stepOrder */
    public void addEvent(Map<String, Object> event) {
        event.put("stepOrder", stepCounter.incrementAndGet());
        events.add(event);
    }

    /** 标记为失败 */
    public void markFailed() {
        this.failed = true;
    }

    // ── getters ──────────────────────────────────────

    public String traceId() { return traceId; }
    public String sessionId() { return sessionId; }
    public Long userId() { return userId; }
    public long startedAt() { return startedAt; }
    public List<Map<String, Object>> events() { return events; }
    public boolean isFailed() { return failed; }
}
