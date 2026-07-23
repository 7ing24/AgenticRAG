package com.demo.aiknowledge.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.demo.aiknowledge.config.CacheConfig;
import com.demo.aiknowledge.dto.AiResponse;
import com.demo.aiknowledge.entity.AgentRun;
import com.demo.aiknowledge.entity.AgentStep;
import com.demo.aiknowledge.entity.Conversation;
import com.demo.aiknowledge.entity.Message;
import com.demo.aiknowledge.entity.QaLog;
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
import com.demo.aiknowledge.service.AgentTraceService;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

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
    private final ObjectMapper objectMapper;
    private final CacheService cacheService;
    private final AgentTraceService agentTraceService;

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
                    Map.of("content", truncate(content, 200), "conversationId", conversationId),
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
                    Map.of("content", truncate(content, 100)),
                    Map.of("messageId", userMsg.getId()));

            // 1.1 更新对话上下文缓存（前端拉历史消息列表时用，不影响AI回答）
            conversationContextService.updateConversationContext(conversationId, userId, userMsg);

            // 检查是否为第一条消息，如果是则生成标题
            if (msgCount == 0) { // 保存前 count=0，保存后为第一条
                 // 异步生成标题，避免阻塞
                 aiService.generateTitle(conversationId, content);
            }

            // 2. 先查缓存，再决定是否调用 Python AI 服务
            String cacheKey = (userId != null ? userId : "anonymous") + ":" + content.trim().toLowerCase();
            long cacheStart = System.nanoTime();
            AiResponse cachedResponse = cacheService.get(
                    CacheConfig.CacheConstants.CACHE_AI_ANSWER, cacheKey, AiResponse.class);
            long cacheLatency = (System.nanoTime() - cacheStart) / 1_000_000;
            boolean cacheHit = cachedResponse != null
                    && cachedResponse.getAnswer() != null
                    && !cachedResponse.getAnswer().contains("AI服务");

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
                        Map.of("question", truncate(content, 100)),
                        Map.of("answer", truncate(answer, 100)));
            } else {
                agentTraceService.recordEvent("PYTHON_CALL_START", "AI_CALL",
                        Map.of("question", truncate(content, 200), "conversationId", conversationId),
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
                        Map.of("question", truncate(content, 200), "conversationId", conversationId),
                        Map.of("answer", truncate(answer, 200), "taskType", taskType != null ? taskType : "unknown",
                               "sourceCount", aiResponse.getSources() != null ? aiResponse.getSources().size() : 0),
                        pyLatency);
            }

            String sourcesJson = null;
            if (aiResponse.getSources() != null && !aiResponse.getSources().isEmpty()) {
                try {
                    sourcesJson = objectMapper.writeValueAsString(aiResponse.getSources());
                } catch (Exception e) {
                    log.error("Failed to serialize sources", e);
                }
            } else {
                if (answer.contains("抱歉") || answer.contains("无法回答")) {
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
                        agentRun.setStatus("COMPLETED");
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

                        java.util.List<java.util.Map<String, Object>> steps = runRecord.getSteps();
                        if (steps != null && !steps.isEmpty()) {
                            int stepIdx = 0;
                            for (java.util.Map<String, Object> stepData : steps) {
                                AgentStep step = new AgentStep();
                                step.setId(java.util.UUID.randomUUID().toString());
                                step.setRunId(runRecord.getRunId());
                                step.setStepName(String.valueOf(stepData.getOrDefault("step_name", "unknown")));
                                step.setStepType(String.valueOf(stepData.getOrDefault("step_type", "unknown")));
                                step.setStatus(String.valueOf(stepData.getOrDefault("status", "completed")));
                                if (stepData.containsKey("output")) {
                                    String output = String.valueOf(stepData.get("output"));
                                    step.setOutput(output.length() > 500 ? output.substring(0, 500) : output);
                                }
                                step.setStartTime(LocalDateTime.now());
                                step.setEndTime(LocalDateTime.now());
                                step.setCreatedAt(LocalDateTime.now().plusSeconds(stepIdx++));
                                step.setDurationMs(0L);
                                agentStepMapper.insert(step);
                            }
                        }
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

                    java.util.List<java.util.Map<String, Object>> steps = aiResponse.getSteps();
                    if (steps != null && !steps.isEmpty()) {
                        int fallbackStepIdx = 0;
                        for (java.util.Map<String, Object> stepData : steps) {
                            AgentStep step = new AgentStep();
                            step.setId(java.util.UUID.randomUUID().toString());
                            step.setRunId(agentRun.getRunId());
                            step.setStepName(String.valueOf(stepData.getOrDefault("step_name", "unknown")));
                            step.setStepType(String.valueOf(stepData.getOrDefault("step_type", "unknown")));
                            step.setStatus(String.valueOf(stepData.getOrDefault("status", "completed")));
                            if (stepData.containsKey("output")) {
                                String output = String.valueOf(stepData.get("output"));
                                step.setOutput(output.length() > 500 ? output.substring(0, 500) : output);
                            }
                            step.setStartTime(LocalDateTime.now());
                            step.setEndTime(LocalDateTime.now());
                            step.setCreatedAt(LocalDateTime.now().plusSeconds(fallbackStepIdx++));
                            step.setDurationMs(0L);
                            agentStepMapper.insert(step);
                        }
                    }
                }
            } catch (Exception e) {
                log.error("Failed to save agent run record", e);
            }

            agentTraceService.recordEvent("REQUEST_FINISHED", "HTTP", null,
                    Map.of("answerLength", answer != null ? answer.length() : 0, "taskType", taskType));
            return aiMsg;

            } catch (Exception e) {
                log.error("Request failed: traceId={}", traceId, e);
                agentTraceService.recordEvent("REQUEST_FAILED", "HTTP", null,
                        Map.of("error", e.getClass().getSimpleName() + ": " + (e.getMessage() != null ? e.getMessage().substring(0, Math.min(e.getMessage().length(), 200)) : "")));
                throw e;
            }
        } // TraceScope.close(): flush all trace events + cleanup ThreadLocal
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
}