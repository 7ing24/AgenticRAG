package com.demo.aiknowledge.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.demo.aiknowledge.config.CacheConfig;
import com.demo.aiknowledge.dto.AiResponse;
import com.demo.aiknowledge.entity.AgentRun;
import com.demo.aiknowledge.entity.AgentStep;
import com.demo.aiknowledge.entity.Conversation;
import com.demo.aiknowledge.entity.Message;
import com.demo.aiknowledge.entity.QaLog;
import com.demo.aiknowledge.entity.ToolCall;
import com.demo.aiknowledge.mapper.AgentStepMapper;
import com.demo.aiknowledge.mapper.ConversationMapper;
import com.demo.aiknowledge.mapper.MessageMapper;
import com.demo.aiknowledge.mapper.QaLogMapper;
import com.demo.aiknowledge.service.AgentRunService;
import com.demo.aiknowledge.service.AiService;
import com.demo.aiknowledge.service.CacheService;
import com.demo.aiknowledge.service.ChatService;
import com.demo.aiknowledge.service.ConversationContextService;
import com.demo.aiknowledge.service.QaUnansweredService;
import com.demo.aiknowledge.service.ToolCallService;
import com.demo.aiknowledge.service.AgentTraceService;
import com.demo.aiknowledge.utils.CacheUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Service
@Slf4j
@RequiredArgsConstructor
public class ChatServiceImpl implements ChatService {

    private final ConversationMapper conversationMapper;
    private final MessageMapper messageMapper;
    private final QaLogMapper qaLogMapper;
    private final AiService aiService;
    private final QaUnansweredService qaUnansweredService;
    private final ConversationContextService conversationContextService;
    private final AgentRunService agentRunService;
    private final AgentStepMapper agentStepMapper;
    private final ToolCallService toolCallService;
    private final ObjectMapper objectMapper;
    private final CacheService cacheService;
    private final AgentTraceService agentTraceService;

    private static final String[] UNANSWERED_KEYWORDS = {
            "抱歉", "无法回答", "知识库中暂无", "暂无相关", "未找到相关", "没有找到相关", "无相关信息"
    };

    private boolean isUnanswered(String answer) {
        if (answer == null) {
            return false;
        }
        for (String keyword : UNANSWERED_KEYWORDS) {
            if (answer.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public Conversation createConversation(Long userId, String title) {
        Conversation conversation = new Conversation();
        conversation.setUserId(userId);
        conversation.setTitle(title != null ? title : "New Chat " + LocalDateTime.now());
        conversation.setCreateTime(LocalDateTime.now());
        conversationMapper.insert(conversation);
        return conversation;
    }

    @Override
    public List<Conversation> getHistory(Long userId) {
        return conversationMapper.selectList(new LambdaQueryWrapper<Conversation>()
                .eq(Conversation::getUserId, userId)
                .orderByDesc(Conversation::getIsPinned) // 先按置顶排序
                .orderByDesc(Conversation::getCreateTime)); // 再按时间排序
    }

    @Override
    public Conversation updateConversation(Long conversationId, String title, Boolean isPinned) {
        Conversation conversation = conversationMapper.selectById(conversationId);
        if (conversation != null) {
            if (title != null) {
                conversation.setTitle(title);
            }
            if (isPinned != null) {
                conversation.setIsPinned(isPinned);
            }
            conversationMapper.updateById(conversation);
        }
        return conversation;
    }

    @Override
    @Transactional
    public Message sendMessage(Long userId, Long conversationId, String content) {
        String traceId = java.util.UUID.randomUUID().toString();

        // ═══════ 开启全链路追踪 ═══════
        try (AgentTraceService.TraceScope scope = agentTraceService.openTrace(
                traceId, conversationId.toString(), userId)) {

            // ── 1. REQUEST_RECEIVED ──────────────────────
            agentTraceService.recordEvent("REQUEST_RECEIVED", "HTTP",
                    Map.of("content", content, "conversationId", conversationId),
                    null);

            try {

                // ── 2. CONVERSATION_LOADED ───────────────────
                Long msgCount = messageMapper.selectCount(new LambdaQueryWrapper<Message>()
                        .eq(Message::getConversationId, conversationId));
                agentTraceService.recordEvent("CONVERSATION_LOADED", "DB",
                        Map.of("conversationId", conversationId),
                        Map.of("messageCount", msgCount, "isNew", msgCount == 0));

                // 1. 保存用户消息
                Message userMsg = new Message();
                userMsg.setConversationId(conversationId);
                userMsg.setRole("user");
                userMsg.setContent(content);
                userMsg.setCreateTime(LocalDateTime.now());
                messageMapper.insert(userMsg);

                // ── 3. USER_MESSAGE_SAVED ────────────────────
                agentTraceService.recordEvent("USER_MESSAGE_SAVED", "DB",
                        Map.of("content", content),
                        Map.of("messageId", userMsg.getId()));

                // 1.1 更新对话上下文缓存（前端拉历史消息列表时用，不影响AI回答）
                conversationContextService.updateConversationContext(conversationId, userId, userMsg);

                // 检查是否为第一条消息，如果是则生成标题
                if (msgCount == 0) { // 保存前 count=0，保存后为第一条
                    // 异步生成标题，避免阻塞
                    aiService.generateTitle(conversationId, content);
                }

                // 2. 先查缓存，再决定是否调用 Python AI 服务
                String cacheKey = CacheUtils.normalizeQuestion(content);
                long cacheStart = System.nanoTime();
                AiResponse cachedResponse = cacheService.get(
                        CacheConfig.CacheConstants.CACHE_AI_ANSWER, cacheKey, AiResponse.class);
                long cacheLatency = (System.nanoTime() - cacheStart) / 1_000_000;
                boolean cacheHit = cachedResponse != null && cachedResponse.isValidAnswer();

                log.info("Cache {} for question: {} ({}ms)",
                        cacheHit ? "HIT" : "MISS",
                        content.length() > 50 ? content.substring(0, 50) + "..." : content,
                        cacheLatency);

                agentTraceService.recordEvent("CACHE_LOOKUP", "CACHE",
                        Map.of("cacheKey", cacheKey.length() > 60 ? cacheKey.substring(0, 60) + "..." : cacheKey),
                        Map.of("result", cacheHit ? "HIT" : "MISS", "latencyMs", cacheLatency));

                AiResponse aiResponse;
                String answer;
                String taskType;

                if (cacheHit) {
                    aiResponse = cachedResponse;
                    aiResponse.setCached(true);
                    answer = aiResponse.getAnswer();
                    taskType = aiResponse.getTaskType();
                    agentTraceService.recordEvent("CACHE_HIT_RETURN", "CACHE",
                            Map.of("question", content),
                            Map.of("answer", answer));
                } else {
                    // 精确未命中 → 尝试语义缓存
                    String semanticKey = aiService.semanticCacheLookup(content);
                    AiResponse semanticCached = null;
                    if (semanticKey != null) {
                        semanticCached = cacheService.get(
                                CacheConfig.CacheConstants.CACHE_AI_ANSWER, semanticKey, AiResponse.class);
                    }

                    if (semanticCached != null && semanticCached.isValidAnswer()) {
                        aiResponse = semanticCached;
                        aiResponse.setCached(true);
                        answer = aiResponse.getAnswer();
                        taskType = aiResponse.getTaskType();
                        log.info("Semantic cache HIT: key='{}'", semanticKey);
                    } else {
                        agentTraceService.recordEvent("PYTHON_CALL_START", "AI_CALL",
                                Map.of("question", content, "conversationId", conversationId),
                                null);

                        long pyStart = System.nanoTime();
                        aiResponse = aiService.ask(content, userId, conversationId, traceId);
                        long pyLatency = (System.nanoTime() - pyStart) / 1_000_000;

                        answer = aiResponse.getAnswer();
                        taskType = aiResponse.getTaskType();

                        if (aiResponse.getTraces() != null && !aiResponse.getTraces().isEmpty()) {
                            agentTraceService.mergePythonTraces(aiResponse.getTraces());
                        }

                        agentTraceService.recordEvent("PYTHON_CALL_END", "AI_CALL",
                                Map.of("question", content, "conversationId", conversationId),
                                Map.of("answer", answer, "taskType", taskType != null ? taskType : "unknown",
                                        "sourceCount", aiResponse.getSources() != null ? aiResponse.getSources().size() : 0),
                                pyLatency);
                    }
                }

                String sourcesJson = null;
                if (aiResponse.getSources() != null && !aiResponse.getSources().isEmpty()) {
                    try {
                        sourcesJson = objectMapper.writeValueAsString(aiResponse.getSources());
                    } catch (Exception e) {
                        log.error("Failed to serialize sources", e);
                    }
                } else {
                    if (isUnanswered(answer)) {
                        qaUnansweredService.recordUnansweredQuestion(content);
                    }
                }

                // 3. 保存 AI 回答
                Message aiMsg = new Message();
                aiMsg.setConversationId(conversationId);
                aiMsg.setRole("assistant");
                aiMsg.setContent(answer);
                aiMsg.setSources(sourcesJson);
                aiMsg.setTaskType(taskType);
                aiMsg.setCreateTime(LocalDateTime.now());
                messageMapper.insert(aiMsg);

                // ── 8. AI_MESSAGE_SAVED ──────────────────────
                agentTraceService.recordEvent("AI_MESSAGE_SAVED", "DB",
                        Map.of("answerLength", answer != null ? answer.length() : 0, "taskType", taskType),
                        Map.of("messageId", aiMsg.getId()));

                // 3.1 更新对话上下文（AI消息）
                conversationContextService.updateConversationContext(conversationId, userId, aiMsg);

                // 4. 记录 QA 日志
                QaLog qaLog = new QaLog();
                qaLog.setUserId(userId);
                qaLog.setQuestion(content);
                qaLog.setAnswer(answer);
                qaLog.setCreateTime(LocalDateTime.now());
                qaLogMapper.insert(qaLog);

                // ── 9. QA_LOG_RECORDED ───────────────────────
                agentTraceService.recordEvent("QA_LOG_RECORDED", "DB", null,
                        Map.of("qaLogId", qaLog.getId()));

                // ── 10. 记录 Agent 执行记录（供管理端查看） ──
                try {
                    String finalTraceId = aiResponse.getTraceId() != null ? aiResponse.getTraceId() : traceId;
                    java.util.List<AiResponse.AgentRunRecord> runs = aiResponse.getRuns();

                    if (runs != null && !runs.isEmpty()) {
                        for (AiResponse.AgentRunRecord runRecord : runs) {
                            AgentRun agentRun = new AgentRun();
                            agentRun.setId(java.util.UUID.randomUUID().toString());
                            agentRun.setRunId(runRecord.getRunId());
                            agentRun.setTraceId(finalTraceId);
                            agentRun.setParentRunId(runRecord.getParentRunId());
                            agentRun.setAgentType(runRecord.getAgentType());
                            agentRun.setConversationId(String.valueOf(conversationId));
                            agentRun.setUserId(String.valueOf(userId));
                            agentRun.setStatus("completed");
                            agentRun.setGoal("Answer: " + (content.length() > 100 ? content.substring(0, 100) + "..." : content));
                            String runInput = (runRecord.getParentRunId() != null && runRecord.getQuestion() != null)
                                    ? runRecord.getQuestion() : content;
                            agentRun.setInput(runInput);
                            if (runRecord.getParentRunId() == null) {
                                agentRun.setOutput(answer.length() > 500 ? answer.substring(0, 500) + "..." : answer);
                            }
                            agentRun.setStartTime(LocalDateTime.now());
                            agentRun.setEndTime(LocalDateTime.now());
                            agentRun.setCreatedAt(LocalDateTime.now());
                            agentRunService.saveAgentRun(agentRun);

                            saveAgentSteps(runRecord.getRunId(), runRecord.getSteps());
                        }
                    } else {
                        AgentRun agentRun = new AgentRun();
                        agentRun.setId(java.util.UUID.randomUUID().toString());
                        agentRun.setRunId(java.util.UUID.randomUUID().toString());
                        agentRun.setTraceId(finalTraceId);
                        agentRun.setConversationId(String.valueOf(conversationId));
                        agentRun.setUserId(String.valueOf(userId));
                        agentRun.setStatus("completed");
                        agentRun.setGoal("Answer: " + (content.length() > 100 ? content.substring(0, 100) + "..." : content));
                        agentRun.setInput(content);
                        agentRun.setOutput(answer.length() > 500 ? answer.substring(0, 500) + "..." : answer);
                        agentRun.setStartTime(LocalDateTime.now());
                        agentRun.setEndTime(LocalDateTime.now());
                        agentRun.setCreatedAt(LocalDateTime.now());
                        agentRunService.saveAgentRun(agentRun);

                        saveAgentSteps(agentRun.getRunId(), aiResponse.getSteps());
                    }
                } catch (Exception e) {
                    log.error("Failed to save agent run record", e);
                }

                agentTraceService.recordEvent("REQUEST_FINISHED", "HTTP", null,
                        Map.of("answerLength", answer != null ? answer.length() : 0, "taskType", taskType));

                // 缓存非错误响应
                if (!cacheHit && answer != null && !answer.contains("AI服务") && !answer.contains("抱歉")) {
                    AiResponse cacheResponse = new AiResponse();
                    cacheResponse.setAnswer(answer);
                    cacheResponse.setTaskType(taskType);
                    try {
                        if (sourcesJson != null) {
                            cacheResponse.setSources(objectMapper.readValue(sourcesJson, List.class));
                        }
                    } catch (Exception e) {
                        log.error("Failed to deserialize sources for cache", e);
                    }
                    String normalizedKey = CacheUtils.normalizeQuestion(content);
                    cacheService.set(CacheConfig.CacheConstants.CACHE_AI_ANSWER, normalizedKey, cacheResponse);
                    aiService.addToSemanticCache(normalizedKey);
                }

                return aiMsg;

            } catch (Exception e) {
                log.error("Request failed: traceId={}", traceId, e);
                agentTraceService.recordEvent("REQUEST_FAILED", "HTTP", null,
                        Map.of("error", e.getClass().getSimpleName() + ": " + (e.getMessage() != null ? e.getMessage().substring(0, Math.min(e.getMessage().length(), 200)) : "")));
                recordFailedAgentRun(conversationId, userId, content, traceId, e);
                throw e;
            }
        } // TraceScope.close(): flush all trace events + cleanup ThreadLocal
    }

    @Override
    @Transactional
    public Message completeStreamingMessage(Long userId, Long conversationId, String question,
                                            String answer, String taskType, String sourcesJson, String traceId,
                                            List<Map<String, Object>> steps) {

        // 如果已有 trace 上下文（来自 Controller 流式端点），复用；否则创建新的
        com.demo.aiknowledge.common.AgentTraceContext existingCtx =
                com.demo.aiknowledge.common.AgentTraceContext.current();
        final boolean ownsContext = (existingCtx == null);
        final AgentTraceService.TraceScope scope;

        if (ownsContext) {
            scope = agentTraceService.openTrace(traceId, conversationId.toString(), userId);
        } else {
            scope = null;
        }

        try {
            // 1. 保存 AI 回答
            Message aiMsg = new Message();
            aiMsg.setConversationId(conversationId);
            aiMsg.setRole("assistant");
            aiMsg.setContent(answer);
            aiMsg.setSources(sourcesJson);
            aiMsg.setTaskType(taskType);
            aiMsg.setCreateTime(LocalDateTime.now());
            messageMapper.insert(aiMsg);

            agentTraceService.recordEvent("AI_MESSAGE_SAVED", "DB",
                    Map.of("answerLength", answer != null ? answer.length() : 0, "taskType", taskType),
                    Map.of("messageId", aiMsg.getId()));

            // 2. 更新对话上下文
            conversationContextService.updateConversationContext(conversationId, userId, aiMsg);

            // 3. 记录 QA 日志
            QaLog qaLog = new QaLog();
            qaLog.setUserId(userId);
            qaLog.setQuestion(question);
            qaLog.setAnswer(answer);
            qaLog.setCreateTime(LocalDateTime.now());
            qaLogMapper.insert(qaLog);

            agentTraceService.recordEvent("QA_LOG_RECORDED", "DB", null,
                    Map.of("qaLogId", qaLog.getId()));

            // 4. 未回答问题记录（对齐非流式：仅在无参考来源时记录）
            if (sourcesJson == null && isUnanswered(answer)) {
                qaUnansweredService.recordUnansweredQuestion(question);
            }

            // 5. 记录 AgentRun（对齐非流式字段）
            try {
                AgentRun agentRun = new AgentRun();
                agentRun.setId(UUID.randomUUID().toString());
                agentRun.setRunId(UUID.randomUUID().toString());
                agentRun.setTraceId(traceId);
                agentRun.setAgentType(taskType != null ? taskType : "unknown");
                agentRun.setConversationId(String.valueOf(conversationId));
                agentRun.setUserId(String.valueOf(userId));
                agentRun.setStatus("completed");
                agentRun.setGoal("Answer: " + question);
                agentRun.setInput(question);
                agentRun.setOutput(answer);
                agentRun.setStartTime(LocalDateTime.now());
                agentRun.setEndTime(LocalDateTime.now());
                agentRun.setCreatedAt(LocalDateTime.now());
                agentRunService.saveAgentRun(agentRun);

                saveAgentSteps(agentRun.getRunId(), steps);
            } catch (Exception e) {
                log.error("Failed to save agent run record for streaming", e);
            }

            // 6. 缓存 AI 回答（仅缓存有效响应）
            if (answer != null && !answer.contains("AI服务") && !answer.contains("抱歉")) {
                String cacheKey = CacheUtils.normalizeQuestion(question);
                AiResponse cacheResponse = new AiResponse();
                cacheResponse.setAnswer(answer);
                cacheResponse.setTaskType(taskType);
                try {
                    if (sourcesJson != null) {
                        cacheResponse.setSources(objectMapper.readValue(sourcesJson, List.class));
                    }
                } catch (Exception e) {
                    log.error("Failed to deserialize sources for cache", e);
                }
                cacheService.set(CacheConfig.CacheConstants.CACHE_AI_ANSWER, cacheKey, cacheResponse);
                aiService.addToSemanticCache(cacheKey);
            }

            // 7. 检查是否首条消息，异步生成标题
            Long msgCount = messageMapper.selectCount(new LambdaQueryWrapper<Message>()
                    .eq(Message::getConversationId, conversationId));
            if (msgCount <= 2) {
                aiService.generateTitle(conversationId, question);
            }

            return aiMsg;

        } catch (Exception e) {
            log.error("Post-stream processing failed: traceId={}", traceId, e);
            agentTraceService.recordEvent("STREAM_POST_PROCESS_FAILED", "HTTP", null,
                    Map.of("error", e.getClass().getSimpleName()));
            recordFailedAgentRun(conversationId, userId, question, traceId, e);
            throw e;
        } finally {
            if (ownsContext && scope != null) {
                scope.close();
            }
        }
    }

    /**
     * 记录一次失败的 Agent 运行记录（独立事务，避免随外层异常回滚）。
     */
    private void recordFailedAgentRun(Long conversationId, Long userId, String question,
                                      String traceId, Exception e) {
        try {
            AgentRun agentRun = new AgentRun();
            agentRun.setId(java.util.UUID.randomUUID().toString());
            agentRun.setRunId(java.util.UUID.randomUUID().toString());
            agentRun.setTraceId(traceId);
            agentRun.setConversationId(String.valueOf(conversationId));
            agentRun.setUserId(String.valueOf(userId));
            agentRun.setStatus("failed");
            agentRun.setGoal("Answer: " + question);
            agentRun.setInput(question);
            String errMsg = e.getClass().getSimpleName() + ": "
                    + (e.getMessage() != null ? e.getMessage() : "");
            agentRun.setErrorMessage(errMsg.length() > 500 ? errMsg.substring(0, 500) : errMsg);
            agentRun.setStartTime(LocalDateTime.now());
            agentRun.setEndTime(LocalDateTime.now());
            agentRun.setCreatedAt(LocalDateTime.now());
            agentRunService.saveAgentRun(agentRun);
        } catch (Exception ex) {
            log.error("Failed to save failed agent run record", ex);
        }
    }

    @Override
    public List<Message> getMessages(Long conversationId) {
        // 使用对话上下文服务获取消息，支持滑动窗口和缓存
        return conversationContextService.getConversationContext(conversationId, 20);
    }

    @Override
    @Transactional
    public void deleteConversation(Long conversationId) {
        // 删除会话相关的消息
        messageMapper.delete(new LambdaQueryWrapper<Message>().eq(Message::getConversationId, conversationId));
        // 删除会话本身
        conversationMapper.deleteById(conversationId);
    }

    @Override
    @Transactional
    public Message submitFeedback(Long messageId, String feedbackType) {
        // 1. 查找消息
        Message message = messageMapper.selectById(messageId);
        if (message == null) {
            throw new RuntimeException("消息不存在");
        }

        // 2. 更新反馈字段
        message.setFeedbackType(feedbackType);
        message.setFeedbackTime(LocalDateTime.now());
        messageMapper.updateById(message);

        // 3. 清除该会话的缓存，确保下次获取时从数据库读取最新数据
        String cacheKey = CacheConfig.CacheConstants.KEY_CONVERSATION_CONTEXT + message.getConversationId();
        cacheService.delete(CacheConfig.CacheConstants.CACHE_CONVERSATION_CONTEXT, cacheKey);
        log.debug("Cleared conversation context cache for conversationId: {}", message.getConversationId());

        // 4. 如果是AI消息，同步更新QA日志的反馈
        if ("assistant".equals(message.getRole())) {
            QaLog qaLog = qaLogMapper.selectOne(new LambdaQueryWrapper<QaLog>()
                    .eq(QaLog::getAnswer, message.getContent())
                    .orderByDesc(QaLog::getCreateTime)
                    .last("LIMIT 1"));
            if (qaLog != null) {
                qaLog.setFeedbackType(feedbackType);
                qaLog.setFeedbackTime(LocalDateTime.now());
                qaLogMapper.updateById(qaLog);
            }
        }

        return message;
    }

    /** 截断过长文本，用于 trace 事件的 input/output 快照 */
    private static String truncate(String text, int maxLen) {
        if (text == null) return null;
        if (text.length() <= maxLen) return text;
        return text.substring(0, maxLen) + "...[truncated]";
    }

    /** 落库 AgentStep + ToolCall（供非流式和流式路径复用） */
    private void saveAgentSteps(String runId, List<Map<String, Object>> steps) {
        if (steps == null || steps.isEmpty()) return;
        int stepIdx = 0;
        for (Map<String, Object> stepData : steps) {
            AgentStep step = new AgentStep();
            step.setId(java.util.UUID.randomUUID().toString());
            step.setRunId(runId);
            step.setStepName(String.valueOf(stepData.getOrDefault("step_name", "unknown")));
            step.setStepType(String.valueOf(stepData.getOrDefault("step_type", "unknown")));
            step.setStatus(String.valueOf(stepData.getOrDefault("status", "completed")));
            if (stepData.containsKey("output")) {
                String output = String.valueOf(stepData.get("output"));
                step.setOutput(output.length() > 500 ? output.substring(0, 500) : output);
            }
            if (stepData.containsKey("input")) {
                try {
                    step.setInput(objectMapper.writeValueAsString(stepData.get("input")));
                } catch (Exception e) {
                    step.setInput(String.valueOf(stepData.get("input")));
                }
            }
            if (stepData.containsKey("error_message")) {
                step.setErrorMessage(String.valueOf(stepData.get("error_message")));
            }
            if (stepData.containsKey("tool_call_id")) {
                step.setToolCallId(String.valueOf(stepData.get("tool_call_id")));
            }
            step.setStartTime(LocalDateTime.now());
            step.setEndTime(LocalDateTime.now());
            step.setCreatedAt(LocalDateTime.now().plusSeconds(stepIdx++));
            step.setDurationMs(0L);
            agentStepMapper.insert(step);

            // 同步写入工具调用记录（供工具失败分析使用）
            if ("tool_call".equals(step.getStepType())) {
                ToolCall toolCall = new ToolCall();
                toolCall.setToolCallId(step.getToolCallId());
                toolCall.setRunId(runId);
                toolCall.setToolName(step.getStepName());
                toolCall.setInputParams(step.getInput());
                toolCall.setOutput(step.getOutput());
                toolCall.setStatus(step.getStatus());
                toolCall.setErrorMessage(step.getErrorMessage());
                toolCall.setDurationMs(step.getDurationMs());
                toolCall.setTimestamp(LocalDateTime.now());
                toolCallService.saveToolCall(toolCall);
            }
        }
    }

}