package com.demo.aiknowledge.service;

import com.demo.aiknowledge.entity.Conversation;
import com.demo.aiknowledge.entity.Message;
import java.util.List;
import java.util.Map;

public interface ChatService {
    Conversation createConversation(Long userId, String title);
    List<Conversation> getHistory(Long userId);
    Message sendMessage(Long userId, Long conversationId, String content);
    List<Message> getMessages(Long conversationId);
    void deleteConversation(Long conversationId);
    Conversation updateConversation(Long conversationId, String title, Boolean isPinned);
    Message submitFeedback(Long messageId, String feedbackType);

    /**
     * 流式回答完成后，持久化 AI 消息、更新上下文、记录日志等
     */
    Message completeStreamingMessage(Long userId, Long conversationId, String question,
                                     String answer, String taskType, String sourcesJson, String traceId,
                                     List<Map<String, Object>> steps);
}
