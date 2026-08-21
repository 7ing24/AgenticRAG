package com.demo.aiknowledge.service;

import com.demo.aiknowledge.entity.AdminConversation;
import com.demo.aiknowledge.entity.AdminMessage;
import java.util.List;
import java.util.Map;

public interface AdminChatService {
    AdminConversation createConversation(Long adminId, String title);
    List<AdminConversation> getHistory(Long adminId);
    AdminMessage sendMessage(Long adminId, Long conversationId, String content);
    List<AdminMessage> getMessages(Long conversationId);
    void deleteConversation(Long conversationId);
    AdminConversation updateConversation(Long conversationId, String title, Boolean isPinned);
    AdminMessage submitFeedback(Long messageId, String feedbackType);

    /**
     * 流式回答完成后，持久化 AI 消息等
     */
    AdminMessage completeStreamingMessage(Long adminId, Long conversationId, String question,
                                          String answer, String taskType, String sourcesJson, String traceId,
                                          List<Map<String, Object>> steps);
}