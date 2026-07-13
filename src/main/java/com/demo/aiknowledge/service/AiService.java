package com.demo.aiknowledge.service;

import com.demo.aiknowledge.dto.AiResponse;

public interface AiService {
    void parseDocument(String filePath, Long docId);

    /**
     * 回答问题（上下文由 Python MemoryAgent 统一管理）
     * @param question 用户问题
     * @param userId 用户ID
     * @param conversationId 会话ID
     * @return AI回答对象
     */
    AiResponse ask(String question, Long userId, Long conversationId);

    /**
     * 生成会话标题并更新数据库
     * @param conversationId 会话ID
     * @param question 用户问题
     */
    void generateTitle(Long conversationId, String question);

    /**
     * 删除文档向量索引
     */
    void deleteDoc(Long docId);

    /**
     * 管理端AI助手问答（上下文由 Python MemoryAgent 统一管理）
     * @param question 用户问题
     * @param adminId 管理员ID
     * @param conversationId 会话ID
     * @return AI回答对象
     */
    java.util.Map<String, Object> askForAdmin(String question, Long adminId, Long conversationId);
}
