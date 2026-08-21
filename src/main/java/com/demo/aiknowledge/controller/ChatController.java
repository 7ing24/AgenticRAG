package com.demo.aiknowledge.controller;

import com.demo.aiknowledge.common.Result;
import com.demo.aiknowledge.dto.ChatRequest;
import com.demo.aiknowledge.dto.FeedbackRequest;
import com.demo.aiknowledge.entity.Conversation;
import com.demo.aiknowledge.entity.Message;
import com.demo.aiknowledge.mapper.AdminMapper;
import com.demo.aiknowledge.mapper.ConversationMapper;
import com.demo.aiknowledge.mapper.MessageMapper;
import com.demo.aiknowledge.service.ChatService;
import com.demo.aiknowledge.service.UserService;
import com.demo.aiknowledge.config.CacheConfig;
import com.demo.aiknowledge.dto.AiResponse;
import com.demo.aiknowledge.common.AgentTraceContext;
import com.demo.aiknowledge.service.AgentTraceService;
import com.demo.aiknowledge.service.AiService;
import com.demo.aiknowledge.service.CacheService;
import com.demo.aiknowledge.service.ConversationContextService;
import com.demo.aiknowledge.utils.CacheUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.demo.aiknowledge.entity.Admin;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

import java.io.IOException;
import java.util.ArrayList;

import java.io.File;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
@Slf4j
public class ChatController {

    private final ChatService chatService;
    private final WebClient webClient;
    private final UserService userService;
    private final AdminMapper adminMapper;
    private final ConversationMapper conversationMapper;
    private final MessageMapper messageMapper;
    private final ConversationContextService conversationContextService;
    private final ObjectMapper objectMapper;
    private final AgentTraceService agentTraceService;
    private final CacheService cacheService;
    private final AiService aiService;

    @Value("${upload.dir}/temp")
    private String uploadTempDir;

    @PostMapping("/conversations")
    public Result<Conversation> createConversation(@RequestParam Long userId, @RequestParam(required = false) String title) {
        return Result.success(chatService.createConversation(userId, title));
    }

    @GetMapping("/conversations")
    public Result<List<Conversation>> getHistory(@RequestParam Long userId) {
        return Result.success(chatService.getHistory(userId));
    }

    @PostMapping("/messages")
    public Result<Message> sendMessage(@RequestBody ChatRequest request) {
        return Result.success(chatService.sendMessage(request.getUserId(), request.getConversationId(), request.getContent()));
    }

    /**
     * 流式 SSE 端点 —— 透传代理 + 消息落库 + 全链路追踪。
     *
     * <pre>
     * Servlet 线程：保存用户消息 → 返回 Flux
     * BoundedElastic 线程：打开 trace → 透传 Python SSE → 落库 → 关闭 trace
     * </pre>
     *
     * 用 .publishOn(Schedulers.boundedElastic()) 把 [doOnSubscribe..doOnComplete]
     * 固定在一个 worker 线程上，trace 的 ThreadLocal 自然可用，无需手动迁移。
     */
    @PostMapping(value = "/stream/messages", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> sendMessageStream(@RequestBody ChatRequest request) {
        Long userId = request.getUserId();
        Long conversationId = request.getConversationId();
        String content = request.getContent();
        String traceId = UUID.randomUUID().toString();

        log.info("Stream request received: userId={}, conversationId={}, question={}, traceId={}",
                userId, conversationId, content.length() > 50 ? content.substring(0, 50) + "..." : content, traceId);

        // ── Servlet 线程：保存用户消息、检查缓存 ──
        Message userMsg = new Message();
        userMsg.setConversationId(conversationId);
        userMsg.setRole("user");
        userMsg.setContent(content);
        userMsg.setCreateTime(LocalDateTime.now());
        messageMapper.insert(userMsg);

        conversationContextService.updateConversationContext(conversationId, userId, userMsg);

        String cacheKey = CacheUtils.normalizeQuestion(content);
        long cacheStart = System.nanoTime();
        AiResponse cachedResponse = cacheService.get(
                CacheConfig.CacheConstants.CACHE_AI_ANSWER, cacheKey, AiResponse.class);
        long cacheLatency = (System.nanoTime() - cacheStart) / 1_000_000;
        final boolean cacheHit = cachedResponse != null
                && cachedResponse.getAnswer() != null
                && !cachedResponse.getAnswer().contains("AI服务")
                && !cachedResponse.getAnswer().contains("抱歉");

        log.info("Cache {} for stream question: {} ({}ms)",
                cacheHit ? "HIT" : "MISS",
                content.length() > 50 ? content.substring(0, 50) + "..." : content,
                cacheLatency);

        if (cacheHit) {
            String cachedAnswer = cachedResponse.getAnswer();
            String cachedTaskType = cachedResponse.getTaskType();
            log.info("<<< [Stream] Cache HIT: traceId={}, answerLen={}, taskType={}",
                    traceId, cachedAnswer.length(), cachedTaskType);

            // 用数组持有 scope/ctx，doOnSubscribe 和 doOnComplete 可能在不同线程
            final AgentTraceService.TraceScope[] hitScope = {null};
            final AgentTraceContext[] hitCtx = {null};

            return Flux.just(
                    "{\"type\":\"start\"}",
                    "{\"type\":\"end\",\"content\":" + writeJson(cachedAnswer)
                            + ",\"task_type\":\"" + (cachedTaskType != null ? cachedTaskType : "unknown") + "\"}"
            ).doOnSubscribe(s -> {
                hitScope[0] = agentTraceService.openTrace(
                        traceId, conversationId.toString(), userId);
                hitCtx[0] = AgentTraceContext.current();
                agentTraceService.recordEvent("REQUEST_RECEIVED", "HTTP",
                        Map.of("content", content, "conversationId", conversationId), null);

                Long hitMsgCount = messageMapper.selectCount(
                        new LambdaQueryWrapper<Message>().eq(Message::getConversationId, conversationId));
                agentTraceService.recordEvent("CONVERSATION_LOADED", "DB",
                        Map.of("conversationId", conversationId),
                        Map.of("messageCount", hitMsgCount, "isNew", hitMsgCount <= 1));
                agentTraceService.recordEvent("USER_MESSAGE_SAVED", "DB",
                        Map.of("content", content), Map.of("messageId", userMsg.getId()));
                agentTraceService.recordEvent("CACHE_LOOKUP", "CACHE",
                        Map.of("cacheKey", cacheKey.length() > 60 ? cacheKey.substring(0, 60) + "..." : cacheKey),
                        Map.of("result", "HIT"));
                agentTraceService.recordEvent("CACHE_HIT_RETURN", "CACHE",
                        Map.of("question", content),
                        Map.of("answer", cachedAnswer));
                // 不关 scope — doOnComplete 里关
            }).doOnComplete(() -> {
                // 恢复 trace 上下文
                if (hitCtx[0] != null) { AgentTraceContext.set(hitCtx[0]); }
                try {
                    chatService.completeStreamingMessage(
                            userId, conversationId, content,
                            cachedAnswer, cachedTaskType, null, traceId, null);
                    agentTraceService.recordEvent("REQUEST_FINISHED", "HTTP", null,
                            Map.of("answerLength", cachedAnswer.length(), "taskType", cachedTaskType, "cached", true));
                } finally {
                    if (hitScope[0] != null) { hitScope[0].close(); }
                }
            });
        }

        // ── 精确未命中 → 尝试语义缓存 ──
        String semanticKey = aiService.semanticCacheLookup(content);
        if (semanticKey != null) {
            AiResponse semanticCached = cacheService.get(
                    CacheConfig.CacheConstants.CACHE_AI_ANSWER, semanticKey, AiResponse.class);
            if (semanticCached != null && semanticCached.getAnswer() != null
                    && !semanticCached.getAnswer().contains("AI服务")
                    && !semanticCached.getAnswer().contains("抱歉")) {
                String semAnswer = semanticCached.getAnswer();
                String semTaskType = semanticCached.getTaskType();
                log.info("<<< [Stream] Semantic cache HIT: traceId={}, key='{}', answerLen={}",
                        traceId, semanticKey, semAnswer.length());

                final AgentTraceService.TraceScope[] semScope = {null};
                final AgentTraceContext[] semCtx = {null};
                return Flux.just(
                        "{\"type\":\"start\"}",
                        "{\"type\":\"end\",\"content\":" + writeJson(semAnswer)
                                + ",\"task_type\":\"" + (semTaskType != null ? semTaskType : "unknown") + "\"}"
                ).doOnSubscribe(s -> {
                    semScope[0] = agentTraceService.openTrace(
                            traceId, conversationId.toString(), userId);
                    semCtx[0] = AgentTraceContext.current();
                    agentTraceService.recordEvent("CACHE_LOOKUP", "CACHE",
                            Map.of("cacheKey", cacheKey.length() > 60 ? cacheKey.substring(0, 60) + "..." : cacheKey),
                            Map.of("result", "SEMANTIC_HIT", "semanticKey", semanticKey));
                    agentTraceService.recordEvent("CACHE_HIT_RETURN", "CACHE",
                            Map.of("question", content), Map.of("answer", semAnswer));
                }).doOnComplete(() -> {
                    if (semCtx[0] != null) { AgentTraceContext.set(semCtx[0]); }
                    try {
                        chatService.completeStreamingMessage(
                                userId, conversationId, content,
                                semAnswer, semTaskType, null, traceId, null);
                        agentTraceService.recordEvent("REQUEST_FINISHED", "HTTP", null,
                                Map.of("answerLength", semAnswer.length(), "taskType", semTaskType, "cached", true));
                    } finally {
                        if (semScope[0] != null) { semScope[0].close(); }
                    }
                });
            }
        }

        // ── 缓存未命中：透传 Python SSE + 全链路追踪 ──

        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("question", content);
        String username = null;
        if (userId != null) {
            var user = userService.getById(userId);
            if (user != null) {
                username = user.getUsername();
                requestBody.put("username", username);
                log.info("Added username to stream request: {}", username);
            }
        }
        boolean isAdmin = username != null
                && adminMapper.selectCount(new LambdaQueryWrapper<Admin>().eq(Admin::getUsername, username)) > 0;
        requestBody.put("is_admin", isAdmin);
        if (userId != null) requestBody.put("user_id", userId.toString());
        requestBody.put("conversation_id", conversationId.toString());
        requestBody.put("trace_id", traceId);

        log.info("User is admin: {}, userId: {}, conversationId: {}, traceId: {}", isAdmin, userId, conversationId, traceId);
        log.info(">>> [Stream] Calling Python /api/ask/stream: traceId={}", traceId);

        final StringBuilder answer = new StringBuilder();
        final String[] taskType = {null};
        final List<Map<String, Object>> sources = new ArrayList<>();
        final List<Map<String, Object>> pythonTraces = new ArrayList<>();
        final List<Map<String, Object>> agentSteps = new ArrayList<>();
        final long[] pyStart = {0};
        final boolean[] firstToken = {true};

        // 持有 scope 和 context — doOnSubscribe 和 doOnComplete 可能在不同线程
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
                            traceId, conversationId.toString(), userId);
                    ctxHolder[0] = AgentTraceContext.current();  // 保存引用
                    agentTraceService.recordEvent("REQUEST_RECEIVED", "HTTP",
                            Map.of("content", content, "conversationId", conversationId), null);

                    Long msgCount = messageMapper.selectCount(
                            new LambdaQueryWrapper<Message>().eq(Message::getConversationId, conversationId));
                    agentTraceService.recordEvent("CONVERSATION_LOADED", "DB",
                            Map.of("conversationId", conversationId),
                            Map.of("messageCount", msgCount, "isNew", msgCount <= 1));
                    agentTraceService.recordEvent("USER_MESSAGE_SAVED", "DB",
                            Map.of("content", content), Map.of("messageId", userMsg.getId()));
                    agentTraceService.recordEvent("CACHE_LOOKUP", "CACHE",
                            Map.of("cacheKey", cacheKey.length() > 60 ? cacheKey.substring(0, 60) + "..." : cacheKey),
                            Map.of("result", "MISS"));
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
                                log.info("First token received: traceId={}, ttfbMs={}", traceId, ttfb);
                                firstToken[0] = false;
                            }
                            answer.append(event.get("content"));
                        } else if ("end".equals(type)) {
                            if (event.get("content") != null) { answer.setLength(0); answer.append((String) event.get("content")); }
                            if (event.get("task_type") != null) taskType[0] = (String) event.get("task_type");
                        } else if ("routed".equals(type) && event.get("task_type") != null) {
                            taskType[0] = (String) event.get("task_type");
                            log.info("Stream routed: traceId={}, taskType={}", traceId, taskType[0]);
                        } else if ("sources".equals(type) && event.get("content") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> srcs = (List<Map<String, Object>>) event.get("content");
                            sources.addAll(srcs);
                        } else if ("traces".equals(type) && event.get("traces") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> pts = (List<Map<String, Object>>) event.get("traces");
                            pythonTraces.addAll(pts);
                            log.info("<<< [Stream] Python trace events received: traceId={}, count={}", traceId, pts.size());
                        } else if ("steps".equals(type) && event.get("content") instanceof List) {
                            @SuppressWarnings("unchecked")
                            List<Map<String, Object>> stps = (List<Map<String, Object>>) event.get("content");
                            agentSteps.addAll(stps);
                        }
                    } catch (Exception ignored) {}
                })
                .doOnComplete(() -> {
                    // ⚠️ BoundedElastic Worker 不保证同一线程，手动迁移上下文
                    if (ctxHolder[0] != null) { AgentTraceContext.set(ctxHolder[0]); }
                    try {
                        long pyLatency = (System.nanoTime() - pyStart[0]) / 1_000_000;
                        String finalAnswer = answer.toString();
                        String finalTaskType = taskType[0];
                        String sourcesJson = sources.isEmpty() ? null : writeJson(sources);

                        // 先合并 Python trace（排在 PYTHON_CALL_END 前面）
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
                            chatService.completeStreamingMessage(
                                    userId, conversationId, content,
                                    finalAnswer, finalTaskType, sourcesJson, traceId,
                                    agentSteps.isEmpty() ? null : agentSteps);

                            agentTraceService.recordEvent("REQUEST_FINISHED", "HTTP", null,
                                    Map.of("answerLength", finalAnswer.length(), "taskType", finalTaskType));

                            log.info("<<< [Stream] Response generated successfully: traceId={}, answerLen={}, taskType={}, latencyMs={}, sourceCount={}",
                                    traceId, finalAnswer.length(), finalTaskType, pyLatency, sources.size());
                        } else {
                            agentTraceService.markFailed();
                            agentTraceService.recordEvent("REQUEST_FAILED", "HTTP", null,
                                    Map.of("error", "Empty answer from Python"));
                            log.warn("<<< [Stream] Empty answer, not persisted: traceId={}", traceId);
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

    @GetMapping("/messages")
    public Result<List<Message>> getMessages(@RequestParam Long conversationId) {
        return Result.success(chatService.getMessages(conversationId));
    }

    @DeleteMapping("/conversations/{id}")
    public Result<String> deleteConversation(@PathVariable Long id) {
        chatService.deleteConversation(id);
        return Result.success("Conversation deleted");
    }

    @PutMapping("/conversations/{id}")
    public Result<Conversation> updateConversation(@PathVariable Long id, @RequestBody Conversation conversation) {
        return Result.success(chatService.updateConversation(id, conversation.getTitle(), conversation.getIsPinned()));
    }

    @GetMapping("/test-auth")
    public Result<String> testAuth() {
        return Result.success("Authentication successful - you have USER role access");
    }

    // 临时图片上传API，用于用户端上传图片，不会添加到知识库
    @PostMapping("/upload/image")
    public Result<Map<String, Object>> uploadImage(@RequestParam("file") MultipartFile file) {
        String fileName = file.getOriginalFilename();
        if (fileName == null) fileName = "unknown";
        String uuid = UUID.randomUUID().toString();
        String savedFileName = uuid + "_" + fileName;
        String filePath;

        // 保存到本地临时目录
        try {
            File dir = new File(uploadTempDir);
            if (!dir.exists()) {
                dir.mkdirs();
            }
            File savedFile = new File(dir, savedFileName);
            file.transferTo(savedFile);
            filePath = savedFile.getAbsolutePath();

            // 返回文件信息
            Map<String, Object> result = new HashMap<>();
            result.put("id", uuid);
            result.put("name", fileName);
            result.put("path", filePath);
            result.put("url", "/api/chat/view/image/" + uuid);

            return Result.success(result);
        } catch (IOException e) {
            throw new RuntimeException("图片上传失败");
        }
    }

    // 查看临时图片
    @GetMapping("/view/image/{id}")
    public ResponseEntity<Resource> viewImage(@PathVariable String id) {
        try {
            File dir = new File(uploadTempDir);
            if (!dir.exists()) {
                return ResponseEntity.notFound().build();
            }

            // 查找对应ID的文件
            File[] files = dir.listFiles((d, name) -> name.startsWith(id + "_"));
            if (files == null || files.length == 0) {
                return ResponseEntity.notFound().build();
            }

            File file = files[0];
            Resource resource = new FileSystemResource(file);
            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(getContentType(file.getName())))
                    .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=" + file.getName())
                    .body(resource);
        } catch (Exception e) {
            return ResponseEntity.status(500).build();
        }
    }

    // 获取文件类型
    private String getContentType(String fileName) {
        String ext = fileName.substring(fileName.lastIndexOf('.')).toLowerCase();
        switch (ext) {
            case ".png": return "image/png";
            case ".jpg": case ".jpeg": return "image/jpeg";
            case ".gif": return "image/gif";
            case ".bmp": return "image/bmp";
            default: return "application/octet-stream";
        }
    }

    // 消息反馈接口
    @PostMapping("/messages/feedback")
    public Result<Message> submitFeedback(@RequestBody FeedbackRequest request) {
        return Result.success(chatService.submitFeedback(request.getMessageId(), request.getFeedbackType()));
    }

    // 清理临时文件（可选）
    @PostMapping("/cleanup/temp")
    public Result<String> cleanupTemp() {
        try {
            File dir = new File(uploadTempDir);
            if (dir.exists()) {
                File[] files = dir.listFiles();
                if (files != null) {
                    for (File file : files) {
                        // 删除24小时前的文件
                        if (System.currentTimeMillis() - file.lastModified() > 24 * 60 * 60 * 1000) {
                            file.delete();
                        }
                    }
                }
            }
            return Result.success("临时文件清理成功");
        } catch (Exception e) {
            return Result.error("临时文件清理失败");
        }
    }
}
