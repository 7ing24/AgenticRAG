package com.demo.aiknowledge.controller;

import com.demo.aiknowledge.common.Result;
import com.demo.aiknowledge.config.CacheConfig;
import com.demo.aiknowledge.dto.AiResponse;
import com.demo.aiknowledge.dto.FeedbackRequest;
import com.demo.aiknowledge.entity.AdminConversation;
import com.demo.aiknowledge.entity.AdminConversation;
import com.demo.aiknowledge.entity.AdminMessage;
import com.demo.aiknowledge.mapper.AdminConversationMapper;
import com.demo.aiknowledge.mapper.AdminMessageMapper;
import com.demo.aiknowledge.common.AgentTraceContext;
import com.demo.aiknowledge.service.AdminChatService;
import com.demo.aiknowledge.service.AgentTraceService;
import com.demo.aiknowledge.service.AiService;
import com.demo.aiknowledge.service.CacheService;
import com.demo.aiknowledge.utils.CacheUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import com.fasterxml.jackson.databind.ObjectMapper;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import java.util.ArrayList;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/admin-chat")
@RequiredArgsConstructor
@Slf4j
public class AdminChatController {

    private final AdminChatService adminChatService;
    private final WebClient webClient;
    private final AdminMessageMapper adminMessageMapper;
    private final AdminConversationMapper adminConversationMapper;
    private final ObjectMapper objectMapper;
    private final AgentTraceService agentTraceService;
    private final CacheService cacheService;
    private final AiService aiService;

    @PostMapping("/conversations")
    public Result<AdminConversation> createConversation(
            @RequestParam Long adminId,
            @RequestParam(required = false) String title) {
        return Result.success(adminChatService.createConversation(adminId, title));
    }

    @GetMapping("/conversations")
    public Result<List<AdminConversation>> getHistory(@RequestParam Long adminId) {
        return Result.success(adminChatService.getHistory(adminId));
    }

    @PostMapping("/messages")
    public Result<AdminMessage> sendMessage(
            @RequestParam Long adminId,
            @RequestParam Long conversationId,
            @RequestBody Map<String, String> request) {
        String content = request.get("content");
        return Result.success(adminChatService.sendMessage(adminId, conversationId, content));
    }

    @PostMapping(value = "/stream/messages", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> sendMessageStream(
            @RequestParam Long adminId,
            @RequestParam Long conversationId,
            @RequestBody Map<String, String> request) {
        String content = request.get("content");
        String traceId = UUID.randomUUID().toString();

        // ── Servlet 线程：验证会话 + 保存用户消息 ──
        AdminConversation conversation = adminConversationMapper.selectById(conversationId);
        if (conversation == null || !conversation.getAdminId().equals(adminId)) {
            return Flux.error(new RuntimeException("会话不存在或无权访问"));
        }

        AdminMessage userMsg = new AdminMessage();
        userMsg.setConversationId(conversationId);
        userMsg.setRole("user");
        userMsg.setContent(content);
        userMsg.setCreateTime(LocalDateTime.now());
        adminMessageMapper.insert(userMsg);

        // 缓存查询
        String cacheKey = "admin:" + CacheUtils.normalizeQuestion(content);
        com.demo.aiknowledge.dto.AiResponse cachedResponse =
                cacheService.get(com.demo.aiknowledge.config.CacheConfig.CacheConstants.CACHE_AI_ANSWER,
                        cacheKey, com.demo.aiknowledge.dto.AiResponse.class);
        boolean cacheHit = cachedResponse != null
                && cachedResponse.getAnswer() != null
                && !cachedResponse.getAnswer().contains("AI服务");

        if (cacheHit) {
            log.info("[Admin Stream] Cache HIT for question: {} (traceId={})",
                    content.length() > 50 ? content.substring(0, 50) + "..." : content, traceId);
            return buildCachedStreamFlux(cachedResponse, adminId, conversationId, content, traceId);
        }

        // 精确未命中 → 尝试语义缓存
        String semanticKey = aiService.semanticCacheLookup(content);
        if (semanticKey != null) {
            com.demo.aiknowledge.dto.AiResponse semanticCached = cacheService.get(
                    CacheConfig.CacheConstants.CACHE_AI_ANSWER, "admin:" + semanticKey,
                    com.demo.aiknowledge.dto.AiResponse.class);
            if (semanticCached != null && semanticCached.getAnswer() != null
                    && !semanticCached.getAnswer().contains("AI服务")) {
                log.info("[Admin Stream] Semantic cache HIT: key='{}' (traceId={})", semanticKey, traceId);
                return buildCachedStreamFlux(semanticCached, adminId, conversationId, content, traceId);
            }
        }

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("question", content);
        requestBody.put("is_admin", true);
        requestBody.put("conversation_id", conversationId.toString());
        requestBody.put("user_id", adminId.toString());
        requestBody.put("username", "admin_" + adminId);
        requestBody.put("trace_id", traceId);

        log.info("[Admin Stream] Request received: adminId={}, conversationId={}, question={}, traceId={}",
                adminId, conversationId, content.length() > 50 ? content.substring(0, 50) + "..." : content, traceId);
        log.info(">>> [Admin Stream] Calling Python /api/ask/stream: traceId={}", traceId);

        final StringBuilder answer = new StringBuilder();
        final String[] taskType = {null};
        final List<Map<String, Object>> sources = new ArrayList<>();
        final List<Map<String, Object>> pythonTraces = new ArrayList<>();
        final List<Map<String, Object>> agentSteps = new ArrayList<>();
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
                .bodyToFlux(String.class)
                .doOnSubscribe(s -> {
                    scopeHolder[0] = agentTraceService.openTrace(
                            traceId, "admin_" + conversationId, adminId);
                    ctxHolder[0] = AgentTraceContext.current();
                    agentTraceService.recordEvent("REQUEST_RECEIVED", "HTTP",
                            Map.of("content", content, "conversationId", conversationId, "adminId", adminId), null);
                    agentTraceService.recordEvent("USER_MESSAGE_SAVED", "DB",
                            Map.of("content", content), Map.of("messageId", userMsg.getId()));
                    agentTraceService.recordEvent("PYTHON_CALL_START", "AI_CALL",
                            Map.of("question", content, "conversationId", conversationId), null);
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
                                log.info("Admin first token: traceId={}, ttfbMs={}", traceId, ttfb);
                                firstToken[0] = false;
                            }
                            answer.append(event.get("content"));
                        } else if ("end".equals(type)) {
                            if (event.get("content") != null) { answer.setLength(0); answer.append((String) event.get("content")); }
                            if (event.get("task_type") != null) taskType[0] = (String) event.get("task_type");
                        } else if ("routed".equals(type) && event.get("task_type") != null) {
                            taskType[0] = (String) event.get("task_type");
                            log.info("Admin stream routed: traceId={}, taskType={}", traceId, taskType[0]);
                        } else if ("sources".equals(type) && event.get("content") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> srcs = (List<Map<String, Object>>) event.get("content");
                            sources.addAll(srcs);
                        } else if ("traces".equals(type) && event.get("traces") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> pts = (List<Map<String, Object>>) event.get("traces");
                            pythonTraces.addAll(pts);
                            log.info("<<< [Admin Stream] Python trace events received: traceId={}, count={}", traceId, pts.size());
                        } else if ("step_started".equals(type) || "step_finished".equals(type)) {
                            agentSteps.add(event);
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
                                        finalTaskType != null ? finalTaskType : "unknown"),
                                pyLatency);

                        adminChatService.completeStreamingMessage(
                                adminId, conversationId, content,
                                finalAnswer, finalTaskType, sourcesJson, traceId, agentSteps);

                        // 缓存非错误响应
                        if (finalAnswer != null && !finalAnswer.contains("AI服务") && !finalAnswer.contains("抱歉")) {
                            AiResponse cacheResponse = new AiResponse();
                            cacheResponse.setAnswer(finalAnswer);
                            cacheResponse.setTaskType(finalTaskType);
                            cacheResponse.setSources(sources);
                            String normalizedKey = CacheUtils.normalizeQuestion(content);
                            cacheService.set(CacheConfig.CacheConstants.CACHE_AI_ANSWER,
                                    "admin:" + normalizedKey, cacheResponse);
                            aiService.addToSemanticCache(normalizedKey);
                        }

                        agentTraceService.recordEvent("REQUEST_FINISHED", "HTTP", null,
                                Map.of("answerLength", finalAnswer.length(), "taskType", finalTaskType));

                        log.info("<<< [Admin Stream] Response generated successfully: traceId={}, answerLen={}, taskType={}, latencyMs={}",
                                traceId, finalAnswer.length(), finalTaskType, pyLatency);
                    } finally {
                        if (scopeHolder[0] != null) { scopeHolder[0].close(); }
                    }
                })
                .doOnError(e -> {
                    log.error("Admin stream error: traceId={}", traceId, e);
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

    private static String truncate(String text, int maxLen) {
        if (text == null) return null;
        if (text.length() <= maxLen) return text;
        return text.substring(0, maxLen) + "...[truncated]";
    }

    private String writeJson(Object obj) {
        try {
            return objectMapper.writeValueAsString(obj);
        } catch (Exception e) {
            return "\"\"";
        }
    }

    @GetMapping("/messages")
    public Result<List<AdminMessage>> getMessages(@RequestParam Long conversationId) {
        return Result.success(adminChatService.getMessages(conversationId));
    }

    @DeleteMapping("/conversations/{id}")
    public Result<String> deleteConversation(@PathVariable Long id) {
        adminChatService.deleteConversation(id);
        return Result.success("Conversation deleted");
    }

    @PutMapping("/conversations/{id}")
    public Result<AdminConversation> updateConversation(
            @PathVariable Long id,
            @RequestBody AdminConversation conversation) {
        return Result.success(adminChatService.updateConversation(id, conversation.getTitle(), conversation.getIsPinned()));
    }

    @PostMapping("/messages/feedback")
    public Result<AdminMessage> submitFeedback(@RequestBody FeedbackRequest request) {
        return Result.success(adminChatService.submitFeedback(request.getMessageId(), request.getFeedbackType()));
    }

    /**
     * 从缓存构建 SSE 流，模拟正常 streaming 响应
     */
    private Flux<String> buildCachedStreamFlux(AiResponse cached, Long adminId, Long conversationId,
                                                String question, String traceId) {
        return Flux.create(sink -> {
            try {
                String answer = cached.getAnswer();
                String taskType = cached.getTaskType();

                // 1. 发送 routed 事件
                if (taskType != null && !taskType.isEmpty()) {
                    Map<String, Object> routedEvent = new HashMap<>();
                    routedEvent.put("type", "routed");
                    routedEvent.put("task_type", taskType);
                    sink.next("data: " + objectMapper.writeValueAsString(routedEvent) + "\n\n");
                }

                // 2. 发送 token 事件（将答案分块模拟流式输出）
                int chunkSize = 10;
                for (int i = 0; i < answer.length(); i += chunkSize) {
                    int end = Math.min(i + chunkSize, answer.length());
                    String chunk = answer.substring(i, end);
                    Map<String, Object> tokenEvent = new HashMap<>();
                    tokenEvent.put("type", "token");
                    tokenEvent.put("content", chunk);
                    sink.next("data: " + objectMapper.writeValueAsString(tokenEvent) + "\n\n");
                    Thread.sleep(5); // 模拟流式延迟
                }

                // 3. 发送 sources 事件
                if (cached.getSources() != null && !cached.getSources().isEmpty()) {
                    Map<String, Object> sourcesEvent = new HashMap<>();
                    sourcesEvent.put("type", "sources");
                    sourcesEvent.put("content", cached.getSources());
                    sink.next("data: " + objectMapper.writeValueAsString(sourcesEvent) + "\n\n");
                }

                // 4. 发送 end 事件
                Map<String, Object> endEvent = new HashMap<>();
                endEvent.put("type", "end");
                endEvent.put("content", answer);
                endEvent.put("task_type", taskType);
                sink.next("data: " + objectMapper.writeValueAsString(endEvent) + "\n\n");

                // 5. 保存消息并缓存
                String sourcesJson = cached.getSources() != null && !cached.getSources().isEmpty()
                        ? objectMapper.writeValueAsString(cached.getSources()) : null;
                adminChatService.completeStreamingMessage(
                        adminId, conversationId, question, answer, taskType, sourcesJson, traceId);

                log.info("[Admin Stream] Cache hit, streamed {} chars for traceId={}", answer.length(), traceId);
                sink.complete();
            } catch (Exception e) {
                log.error("[Admin Stream] Error building cached flux for traceId={}", traceId, e);
                sink.error(e);
            }
        });
    }

}