package com.demo.aiknowledge.service.impl;

import com.demo.aiknowledge.common.AgentTraceContext;
import com.demo.aiknowledge.service.AgentTraceService;
import com.demo.aiknowledge.service.StreamProxyService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Python SSE 流式问答代理实现
 * 统一 Chat 和 AdminChat 的 webClient 透传管线。
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class StreamProxyServiceImpl implements StreamProxyService {

    private final WebClient webClient;
    private final ObjectMapper objectMapper;
    private final AgentTraceService agentTraceService;

    @Override
    public Flux<String> proxy(
            Map<String, Object> requestBody,
            String content,
            String traceId,
            StreamTraceOpener opener,
            StreamAnswerPersister persister) {

        final StringBuilder answer = new StringBuilder();
        final String[] taskType = {null};
        final List<Map<String, Object>> sources = new ArrayList<>();
        final List<Map<String, Object>> steps = new ArrayList<>();
        final List<Map<String, Object>> pythonTraces = new ArrayList<>();
        final long[] pyStart = {0};
        final boolean[] firstToken = {true};
        final AgentTraceService.TraceScope[] scopeHolder = {null};
        final AgentTraceContext[] ctxHolder = {null};

        return webClient.post()
                .uri("/api/ask/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(requestBody)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .retrieve()
                .bodyToFlux(String.class)  // 把 HTTP 响应体声明为一个 Flux<String>，每个元素是 SSE 流的一行
                .doOnSubscribe(s -> {
                    opener.open(scopeHolder, ctxHolder);
                    pyStart[0] = System.nanoTime();
                })
                .publishOn(Schedulers.boundedElastic())
                .filter(line -> !line.isEmpty())
                .map(line -> line.replaceFirst("^data: ", ""))
                .doOnNext(json -> {
                    try {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> event = objectMapper.readValue(json, Map.class);
                        String type = (String) event.get("type");
                        if ("token".equals(type) && event.get("content") != null) {
                            if (firstToken[0]) {
                                long ttfb = (System.nanoTime() - pyStart[0]) / 1_000_000;
                                log.info("First token: traceId={}, ttfbMs={}", traceId, ttfb);
                                firstToken[0] = false;
                            }
                            answer.append(event.get("content"));
                        } else if ("end".equals(type)) {
                            if (event.get("content") != null) { answer.setLength(0); answer.append((String) event.get("content")); }
                            if (event.get("task_type") != null) taskType[0] = (String) event.get("task_type");
                            log.info("SSE end event received: traceId={}, contentLen={}, taskType={}", traceId, answer.length(), taskType[0]);
                        } else if ("routed".equals(type) && event.get("task_type") != null) {
                            taskType[0] = (String) event.get("task_type");
                            log.info("Stream routed: traceId={}, taskType={}", traceId, taskType[0]);
                        } else if ("sources".equals(type) && event.get("content") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> srcs = (List<Map<String, Object>>) event.get("content");
                            sources.addAll(srcs);
                        } else if ("steps".equals(type) && event.get("content") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> stps = (List<Map<String, Object>>) event.get("content");
                            steps.addAll(stps);
                        } else if ("traces".equals(type) && event.get("traces") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> pts = (List<Map<String, Object>>) event.get("traces");
                            pythonTraces.addAll(pts);
                            log.info("Python trace events received: traceId={}, count={}", traceId, pts.size());
                        }
                    } catch (Exception ignored) {}
                })
                .doOnComplete(() -> {
                    if (ctxHolder[0] != null) { AgentTraceContext.set(ctxHolder[0]); }
                    try {
                        long pyLatency = (System.nanoTime() - pyStart[0]) / 1_000_000;
                        String finalAnswer = answer.toString();
                        String finalTaskType = taskType[0];
                        String sourcesJson = sources.isEmpty() ? null : writeJson(sources);

                        if (!pythonTraces.isEmpty()) {
                            agentTraceService.mergePythonTraces(pythonTraces);
                        }

                        agentTraceService.recordEvent("PYTHON_CALL_END", "AI_CALL",
                                Map.of("question", content),
                                Map.of("answer", finalAnswer, "taskType",
                                        finalTaskType != null ? finalTaskType : "unknown",
                                        "sourceCount", sources.size()),
                                pyLatency);

                        if (finalAnswer != null && !finalAnswer.isEmpty()) {
                            persister.persist(finalAnswer, finalTaskType, sourcesJson, sources, steps);

                            agentTraceService.recordEvent("REQUEST_FINISHED", "HTTP", null,
                                    Map.of("answerLength", finalAnswer.length(), "taskType", finalTaskType));

                            log.info("Stream response generated: traceId={}, answerLen={}, taskType={}, latencyMs={}, sourceCount={}",
                                    traceId, finalAnswer.length(), finalTaskType, pyLatency, sources.size());
                        } else {
                            agentTraceService.markFailed();
                            agentTraceService.recordEvent("REQUEST_FAILED", "HTTP", null,
                                    Map.of("error", "Empty answer from Python"));
                            log.warn("Empty answer, not persisted: traceId={}", traceId);
                        }
                    } finally {
                        if (scopeHolder[0] != null) { scopeHolder[0].close(); }
                    }
                })
                .doOnError(e -> {
                    log.error("Stream error: traceId={}", traceId, e);
                    if (ctxHolder[0] != null) { AgentTraceContext.set(ctxHolder[0]); }
                    try {
                        if (scopeHolder[0] != null) {
                            agentTraceService.recordEvent("REQUEST_FAILED", "HTTP", null,
                                    Map.of("error", e.getClass().getSimpleName()));
                            agentTraceService.markFailed();
                            scopeHolder[0].close();
                        }
                    } finally {
                        AgentTraceContext.remove();
                    }
                });
    }

    private String writeJson(Object obj) {
        try {
            return objectMapper.writeValueAsString(obj);
        } catch (Exception e) {
            return "\"\"";
        }
    }
}
