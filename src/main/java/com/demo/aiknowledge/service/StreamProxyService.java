package com.demo.aiknowledge.service;

import com.demo.aiknowledge.common.AgentTraceContext;
import reactor.core.publisher.Flux;

import java.util.Map;

/**
 * Python SSE 流式问答代理
 * 统一 Chat 和 AdminChat 的 webClient 透传管线，差异通过回调处理。
 */
public interface StreamProxyService {

    /**
     * 代理 Python /api/ask/stream 流式问答
     *
     * @param requestBody 请求体（已含 question/user_id/conversation_id/trace_id）
     * @param content     原始问题
     * @param traceId     追踪 ID
     * @param opener      订阅时打开 trace 并记录事件
     * @param persister   完成时落库（接收最终回答/任务类型/来源 JSON）
     */
    Flux<String> proxy(
            Map<String, Object> requestBody,
            String content,
            String traceId,
            StreamTraceOpener opener,
            StreamAnswerPersister persister);

    /** 订阅时打开 trace 的回调 */
    @FunctionalInterface
    interface StreamTraceOpener {
        void open(AgentTraceService.TraceScope[] scopeHolder, AgentTraceContext[] ctxHolder);
    }

    /** 完成时落库的回调 */
    @FunctionalInterface
    interface StreamAnswerPersister {
        void persist(String answer, String taskType, String sourcesJson,
                     java.util.List<Map<String, Object>> sources,
                     java.util.List<Map<String, Object>> steps);
    }
}
