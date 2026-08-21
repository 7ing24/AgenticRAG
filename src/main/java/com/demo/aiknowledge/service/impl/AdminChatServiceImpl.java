package com.demo.aiknowledge.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.demo.aiknowledge.config.CacheConfig;
import com.demo.aiknowledge.dto.AiResponse;
import com.demo.aiknowledge.entity.AdminConversation;
import com.demo.aiknowledge.entity.AdminMessage;
import com.demo.aiknowledge.entity.AgentStep;
import com.demo.aiknowledge.entity.ToolCall;
import com.demo.aiknowledge.mapper.AdminConversationMapper;
import com.demo.aiknowledge.mapper.AdminMessageMapper;
import com.demo.aiknowledge.mapper.AgentStepMapper;
import com.demo.aiknowledge.service.AdminChatService;
import com.demo.aiknowledge.service.AgentTraceService;
import com.demo.aiknowledge.service.AiService;
import com.demo.aiknowledge.service.CacheService;
import com.demo.aiknowledge.service.ToolCallService;
import com.demo.aiknowledge.utils.CacheUtils;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@Slf4j
@RequiredArgsConstructor
public class AdminChatServiceImpl implements AdminChatService {

    private final AdminConversationMapper adminConversationMapper;
    private final AdminMessageMapper adminMessageMapper;
    private final RestTemplate restTemplate;
    private final ObjectMapper objectMapper;
    private final AiService aiService;
    private final AgentTraceService agentTraceService;
    private final CacheService cacheService;
    private final AgentStepMapper agentStepMapper;
    private final ToolCallService toolCallService;

    @Override
    public AdminConversation createConversation(Long adminId, String title) {
        AdminConversation conversation = new AdminConversation();
        conversation.setAdminId(adminId);
        conversation.setTitle(title != null ? title : "新建会话 " + LocalDateTime.now());
        conversation.setCreateTime(LocalDateTime.now());
        conversation.setIsPinned(false);
        adminConversationMapper.insert(conversation);
        return conversation;
    }

    @Override
    public List<AdminConversation> getHistory(Long adminId) {
        return adminConversationMapper.selectList(new LambdaQueryWrapper<AdminConversation>()
                .eq(AdminConversation::getAdminId, adminId)
                .orderByDesc(AdminConversation::getIsPinned)
                .orderByDesc(AdminConversation::getCreateTime));
    }

    @Override
    public AdminConversation updateConversation(Long conversationId, String title, Boolean isPinned) {
        AdminConversation conversation = adminConversationMapper.selectById(conversationId);
        if (conversation != null) {
            if (title != null) {
                conversation.setTitle(title);
            }
            if (isPinned != null) {
                conversation.setIsPinned(isPinned);
            }
            adminConversationMapper.updateById(conversation);
        }
        return conversation;
    }

    @Override
    @Transactional
    public AdminMessage sendMessage(Long adminId, Long conversationId, String content) {
        String traceId = java.util.UUID.randomUUID().toString();

        try (AgentTraceService.TraceScope scope = agentTraceService.openTrace(
                traceId, "admin_" + conversationId, adminId)) {

            agentTraceService.recordEvent("REQUEST_RECEIVED", "HTTP",
                    Map.of("content", content.length() > 200 ? content.substring(0, 200) + "..." : content,
                           "conversationId", conversationId, "adminId", adminId),
                    null);

            AdminConversation conversation = adminConversationMapper.selectById(conversationId);
            if (conversation == null || !conversation.getAdminId().equals(adminId)) {
                throw new RuntimeException("会话不存在或无权访问");
            }

            AdminMessage userMsg = new AdminMessage();
            userMsg.setConversationId(conversationId);
            userMsg.setRole("user");
            userMsg.setContent(content);
            userMsg.setCreateTime(LocalDateTime.now());
            adminMessageMapper.insert(userMsg);

            agentTraceService.recordEvent("USER_MESSAGE_SAVED", "DB",
                    Map.of("content", content.length() > 100 ? content.substring(0, 100) + "..." : content),
                    Map.of("messageId", userMsg.getId()));

            // 缓存查询
            String cacheKey = CacheUtils.normalizeQuestion(content);
            long cacheStart = System.nanoTime();
            AiResponse cachedResponse = cacheService.get(
                    CacheConfig.CacheConstants.CACHE_AI_ANSWER, "admin:" + cacheKey, AiResponse.class);
            long cacheLatency = (System.nanoTime() - cacheStart) / 1_000_000;
            boolean cacheHit = cachedResponse != null
                    && cachedResponse.getAnswer() != null
                    && !cachedResponse.getAnswer().contains("AI服务");

            log.info("Admin cache {} for question: {} ({}ms)",
                    cacheHit ? "HIT" : "MISS",
                    content.length() > 50 ? content.substring(0, 50) + "..." : content,
                    cacheLatency);

            AiResponse aiResponse;
            String answer;
            String taskType;

            if (cacheHit) {
                aiResponse = cachedResponse;
                aiResponse.setCached(true);
                answer = aiResponse.getAnswer();
                taskType = aiResponse.getTaskType();
            } else {
                // 精确未命中 → 尝试语义缓存
                String semanticKey = aiService.semanticCacheLookup(content);
                AiResponse semanticCached = null;
                if (semanticKey != null) {
                    semanticCached = cacheService.get(
                            CacheConfig.CacheConstants.CACHE_AI_ANSWER, "admin:" + semanticKey, AiResponse.class);
                }

                if (semanticCached != null && semanticCached.getAnswer() != null
                        && !semanticCached.getAnswer().contains("AI服务")) {
                    aiResponse = semanticCached;
                    aiResponse.setCached(true);
                    answer = aiResponse.getAnswer();
                    taskType = aiResponse.getTaskType();
                    log.info("Admin semantic cache HIT: key='{}'", semanticKey);
                } else {
                    agentTraceService.recordEvent("PYTHON_CALL_START", "AI_CALL",
                            Map.of("question", content.length() > 200 ? content.substring(0, 200) + "..." : content,
                                   "conversationId", conversationId),
                            null);

                    long pyStart = System.nanoTime();
                    aiResponse = callAdminAgent(content, adminId, conversationId, traceId);
                    long pyLatency = (System.nanoTime() - pyStart) / 1_000_000;

                    answer = aiResponse.getAnswer();
                    taskType = aiResponse.getTaskType();

                    agentTraceService.recordEvent("PYTHON_CALL_END", "AI_CALL",
                            Map.of("question", content.length() > 200 ? content.substring(0, 200) + "..." : content),
                            Map.of("answer", answer != null && answer.length() > 200 ? answer.substring(0, 200) + "..." : answer,
                                   "taskType", taskType),
                            pyLatency);
                }
            }

            if (aiResponse.getTraces() != null && !aiResponse.getTraces().isEmpty()) {
                agentTraceService.mergePythonTraces(aiResponse.getTraces());
            }

            // 缓存非错误响应
            if (!cacheHit && answer != null && !answer.contains("AI服务") && !answer.contains("抱歉")) {
                AiResponse cacheResponse = new AiResponse();
                cacheResponse.setAnswer(answer);
                cacheResponse.setTaskType(taskType);
                cacheService.set(CacheConfig.CacheConstants.CACHE_AI_ANSWER, "admin:" + cacheKey, cacheResponse);
                aiService.addToSemanticCache(cacheKey);
            }

            String sourcesJson = null;
            if (aiResponse.getSources() != null && !aiResponse.getSources().isEmpty()) {
                try {
                    sourcesJson = objectMapper.writeValueAsString(aiResponse.getSources());
                } catch (Exception e) {
                    log.error("Failed to serialize sources", e);
                }
            }

            AdminMessage aiMsg = new AdminMessage();
            aiMsg.setConversationId(conversationId);
            aiMsg.setRole("assistant");
            aiMsg.setContent(answer);
            aiMsg.setSources(sourcesJson);
            aiMsg.setTaskType(taskType);
            aiMsg.setCreateTime(LocalDateTime.now());
            adminMessageMapper.insert(aiMsg);

            agentTraceService.recordEvent("AI_MESSAGE_SAVED", "DB",
                    Map.of("answerLength", answer != null ? answer.length() : 0, "taskType", taskType),
                    Map.of("messageId", aiMsg.getId()));

            String title = conversation.getTitle();
            if (title == null || title.isEmpty() || title.startsWith("新对话") || title.startsWith("新建会话")) {
                String newTitle = content.length() > 30 ? content.substring(0, 30) + "..." : content;
                conversation.setTitle(newTitle);
                adminConversationMapper.updateById(conversation);
            }

            agentTraceService.recordEvent("REQUEST_FINISHED", "HTTP", null,
                    Map.of("answerLength", answer != null ? answer.length() : 0, "taskType", taskType));
            return aiMsg;

        } catch (Exception e) {
            log.error("Admin request failed: traceId={}", traceId, e);
            agentTraceService.recordEvent("REQUEST_FAILED", "HTTP", null,
                    Map.of("error", e.getClass().getSimpleName() + ": " +
                           (e.getMessage() != null ? e.getMessage().substring(0, Math.min(e.getMessage().length(), 200)) : "")));
            throw e;
        }
    }

    @Override
    public List<AdminMessage> getMessages(Long conversationId) {
        return adminMessageMapper.selectList(new LambdaQueryWrapper<AdminMessage>()
                .eq(AdminMessage::getConversationId, conversationId)
                .orderByAsc(AdminMessage::getCreateTime));
    }

    @Override
    @Transactional
    public void deleteConversation(Long conversationId) {
        adminMessageMapper.delete(new LambdaQueryWrapper<AdminMessage>()
                .eq(AdminMessage::getConversationId, conversationId));
        adminConversationMapper.deleteById(conversationId);
    }

    @Override
    @Transactional
    public AdminMessage completeStreamingMessage(Long adminId, Long conversationId, String question,
                                                  String answer, String taskType, String sourcesJson, String traceId,
                                                  List<Map<String, Object>> steps) {

        // 如果已有 trace 上下文（来自 Controller），复用；否则创建新的
        com.demo.aiknowledge.common.AgentTraceContext existingCtx =
                com.demo.aiknowledge.common.AgentTraceContext.current();
        final boolean ownsContext = (existingCtx == null);
        final AgentTraceService.TraceScope scope;

        if (ownsContext) {
            scope = agentTraceService.openTrace(traceId, "admin_" + conversationId, adminId);
        } else {
            scope = null;
        }

        try {
            // 1. 保存 AI 回答
            AdminMessage aiMsg = new AdminMessage();
            aiMsg.setConversationId(conversationId);
            aiMsg.setRole("assistant");
            aiMsg.setContent(answer);
            aiMsg.setSources(sourcesJson);
            aiMsg.setTaskType(taskType);
            aiMsg.setCreateTime(LocalDateTime.now());
            adminMessageMapper.insert(aiMsg);

            // 保存 Agent 步骤
            saveAgentSteps(traceId, steps);

            agentTraceService.recordEvent("AI_MESSAGE_SAVED", "DB",
                    Map.of("answerLength", answer != null ? answer.length() : 0, "taskType", taskType),
                    Map.of("messageId", aiMsg.getId()));

            // 2. 更新标题（首条消息）
            AdminConversation conversation = adminConversationMapper.selectById(conversationId);
            if (conversation != null) {
                String title = conversation.getTitle();
                if (title == null || title.isEmpty() || title.startsWith("新对话") || title.startsWith("新建会话")) {
                    String newTitle = question.length() > 30 ? question.substring(0, 30) + "..." : question;
                    conversation.setTitle(newTitle);
                    adminConversationMapper.updateById(conversation);
                }
            }

            return aiMsg;

        } catch (Exception e) {
            log.error("Admin post-stream processing failed: traceId={}", traceId, e);
            agentTraceService.recordEvent("STREAM_POST_PROCESS_FAILED", "HTTP", null,
                    Map.of("error", e.getClass().getSimpleName()));
            throw e;
        } finally {
            if (ownsContext && scope != null) {
                scope.close();
            }
        }
    }

    @Override
    @Transactional
    public AdminMessage submitFeedback(Long messageId, String feedbackType) {
        AdminMessage message = adminMessageMapper.selectById(messageId);
        if (message == null) {
            throw new RuntimeException("消息不存在");
        }
        message.setCreateTime(LocalDateTime.now());
        adminMessageMapper.updateById(message);
        return message;
    }


    /** 落库 AgentStep + ToolCall */
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

    private AiResponse callAdminAgent(String question, Long adminId, Long conversationId, String traceId) {
        try {
            Map<String, Object> response = aiService.askForAdmin(question, adminId, conversationId, traceId);
            AiResponse aiResponse = new AiResponse();
            aiResponse.setAnswer((String) response.get("answer"));
            aiResponse.setTaskType((String) response.get("task_type"));
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> sources = (List<Map<String, Object>>) response.get("sources");
            aiResponse.setSources(sources);
            @SuppressWarnings("unchecked")
            List<Map<String, Object>> traces = (List<Map<String, Object>>) response.get("traces");
            aiResponse.setTraces(traces);
            return aiResponse;
        } catch (Exception e) {
            log.error("Failed to call admin agent", e);
            AiResponse aiResponse = new AiResponse();
            aiResponse.setAnswer("抱歉，服务暂时不可用，请稍后再试。");
            aiResponse.setSources(null);
            aiResponse.setTaskType("unknown");
            return aiResponse;
        }
    }

}